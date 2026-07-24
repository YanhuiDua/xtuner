# Copyright (c) OpenMMLab. All rights reserved.

import re
from itertools import accumulate
from pathlib import Path
from typing import Literal, Self, cast

import torch
from pydantic import Field, model_validator
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import BaseLossContext
from xtuner.v1.model.base import BaseModel, ModelOutputs, TransformerConfig
from xtuner.v1.module import (
    CheckpointRMSNorm,
    ConceptQuantizer,
    ConceptRoute,
    ConceptStage,
    DecoderStage,
    DepthDD,
    EncoderStage,
    LMHead,
    MHAConfig,
    ResidualRoute,
)
from xtuner.v1.module.decoder_layer import ConceptOlmoDecoderLayer
from xtuner.v1.module.linear import build_linear
from xtuner.v1.module.rope import RopeParametersConfig


class ConceptLMV22VQ(BaseModel):
    """XTuner implementation of the ConceptLM V2.2 residual-VQ model."""

    config: "ConceptLMV22VQConfig"

    def __init__(self, config: "ConceptLMV22VQConfig"):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = self.build_rotary_embedding(config)

        self.encoder_stages = nn.ModuleList(
            [
                EncoderStage(
                    self._build_transformer(layer_idx),
                    DepthDD(config.hidden_size, layer_idx, use_softmax=False, float8_cfg=config.float8_cfg),
                )
                for layer_idx in range(config.concept_encoder_layers)
            ]
        )
        self.concept_stages = nn.ModuleList(
            [
                ConceptStage(
                    self._build_transformer(layer_idx),
                    DepthDD(config.hidden_size, layer_idx, use_softmax=False, float8_cfg=config.float8_cfg),
                    ResidualRoute(
                        config.hidden_size,
                        config.concept_encoder_layers - 1,
                        use_softmax=True,
                        float8_cfg=config.float8_cfg,
                    ),
                )
                for layer_idx in range(config.concept_special_layers)
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [
                DecoderStage(
                    self._build_transformer(layer_idx),
                    DepthDD(
                        config.hidden_size,
                        layer_idx,
                        use_softmax=config.concept_dd_two_route_add_decoder_use_softmax,
                        float8_cfg=config.float8_cfg,
                    ),
                    ConceptRoute(config.hidden_size, config.rms_norm_eps),
                    ResidualRoute(
                        config.hidden_size,
                        config.concept_encoder_layers,
                        use_softmax=True,
                        float8_cfg=config.float8_cfg,
                    ),
                    ResidualRoute(
                        config.hidden_size,
                        config.concept_special_layers,
                        use_softmax=True,
                        float8_cfg=config.float8_cfg,
                    ),
                )
                for layer_idx in range(config.concept_decoder_layers)
            ]
        )

        self.concept_vq_input_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.concept_quantizer = ConceptQuantizer(
            config.hidden_size,
            config.concept_v22_vq_num_codebooks,
            config.concept_v22_vq_codebook_size,
        )
        self.concept_final_norm = CheckpointRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.prediction_heads = nn.ModuleList(
            [
                build_linear(
                    config.hidden_size,
                    config.concept_v22_vq_codebook_size,
                    bias=True,
                    float8_cfg=config.float8_cfg,
                )
                for _ in range(config.concept_v22_vq_num_codebooks)
            ]
        )
        self.concept_read_encoder_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.fusion_tok_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fusion_hl_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fusion_norm_alpha = nn.Parameter(torch.tensor(config.concept_fusion_norm_alpha_init))
        self.decoder_read_encoder_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_read_concept_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_read_concept_gate_logits = nn.Parameter(torch.zeros(config.concept_decoder_layers, 2))
        self.final_norm = CheckpointRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        self._init_load_spec()
        self._maybe_enable_compile(self.compile_cfg)

    def _build_transformer(self, layer_idx: int) -> ConceptOlmoDecoderLayer:
        config = self.config
        layer_type: Literal["full_attention", "sliding_attention"]
        if (layer_idx + 1) % config.window_attn_skip_freq == 0:
            layer_type = "full_attention"
        else:
            layer_type = "sliding_attention"
        return ConceptOlmoDecoderLayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            num_attention_heads=config.attention.num_attention_heads,
            head_dim=config.attention.head_dim,
            attention_dropout=config.attention.dropout,
            rms_norm_eps=config.rms_norm_eps,
            attn_impl=config.attention.attn_impl,
            sliding_window=config.window_size[0],
            layer_type=layer_type,
            float8_cfg=config.float8_cfg,
        )

    def to_hf_key_list(self, key: str) -> list[str]:
        mappings = (
            (r"^embed_tokens\.", "embedding.word_embeddings."),
            (r"^encoder_stages\.(\d+)\.transformer\.", r"encoder.layers.\1."),
            (r"^encoder_stages\.(\d+)\.depth_dd\.", r"dd_encoder_self_dd.depth_dds.\1."),
            (r"^concept_stages\.(\d+)\.transformer\.", r"concept_predictor.hlm_block.layers.\1."),
            (r"^concept_stages\.(\d+)\.depth_dd\.", r"concept_predictor.concept_self_dd.depth_dds.\1."),
            (
                r"^concept_stages\.(\d+)\.read_encoder_route\.",
                r"concept_predictor.concept_read_encoder_routes.\1.",
            ),
            (r"^decoder_stages\.(\d+)\.transformer\.", r"decoder.layers.\1."),
            (r"^decoder_stages\.(\d+)\.decoder_dd\.", r"dd_two_route_add.decoder_dds.\1."),
            (r"^decoder_stages\.(\d+)\.concept_route\.", r"dd_two_route_add.concept_routes.\1."),
            (
                r"^decoder_stages\.(\d+)\.read_encoder_route\.",
                r"decoder_read_encoder_routes.\1.",
            ),
            (
                r"^decoder_stages\.(\d+)\.read_concept_route\.",
                r"decoder_read_concept_routes.\1.",
            ),
            (r"^concept_final_norm\.", "concept_predictor.hlm_block.final_layernorm."),
            (r"^prediction_heads\.", "concept_predictor.prediction_heads."),
            (
                r"^concept_read_encoder_shared_source_norm\.",
                "concept_predictor.concept_read_encoder_shared_source_norm.",
            ),
            (r"^final_norm\.", "decoder.final_layernorm."),
            (r"^lm_head\.", "output_layer."),
        )
        for pattern, replacement in mappings:
            if re.search(pattern, key):
                return [re.sub(pattern, replacement, key)]
        return [key]

    @staticmethod
    def _lengths(seq_ctx: SequenceContext) -> list[int]:
        lengths = seq_ctx.cu_seq_lens_q[1:] - seq_ctx.cu_seq_lens_q[:-1]
        return [int(length) for length in lengths.detach().cpu().tolist()]

    def _merge_chunks(
        self,
        hidden_states: torch.Tensor,
        token_lengths: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        chunks: list[torch.Tensor] = []
        concept_lengths: list[int] = []
        start = 0
        chunk_size = self.config.concept_chunk_size
        for seq_len in token_lengths:
            state = hidden_states[:, start : start + seq_len]
            start += seq_len
            if seq_len < chunk_size:
                merged = state.mean(dim=1, keepdim=True)
            else:
                usable = (seq_len // chunk_size) * chunk_size
                reshaped = state[:, :usable].reshape(1, usable // chunk_size, chunk_size, *state.shape[2:])
                if self.config.concept_chunk_merge_method == "first":
                    merged = reshaped[:, :, 0]
                elif self.config.concept_chunk_merge_method == "last":
                    merged = reshaped[:, :, -1]
                else:
                    merged = reshaped.mean(dim=2)
            chunks.append(merged)
            concept_lengths.append(merged.shape[1])
        return torch.cat(chunks, dim=1), concept_lengths

    def _repeat_shift(
        self,
        concept_states: torch.Tensor,
        concept_lengths: list[int],
        token_lengths: list[int],
        prepend_zero: bool = True,
    ) -> torch.Tensor:
        repeated_docs: list[torch.Tensor] = []
        start = 0
        chunk_size = self.config.concept_chunk_size
        for concept_len, seq_len in zip(concept_lengths, token_lengths, strict=True):
            state = concept_states[:, start : start + concept_len]
            start += concept_len
            if prepend_zero:
                shifted = torch.cat((torch.zeros_like(state[:, :1]), state), dim=1)
            else:
                shifted = state
            repeated = shifted.repeat_interleave(chunk_size, dim=1)[:, : seq_len + 1]
            if self.config.concept_shift_feature:
                repeated = repeated[:, 1:]
            else:
                repeated = repeated[:, :seq_len]
            if repeated.shape[1] < seq_len:
                padding = repeated.new_zeros(1, seq_len - repeated.shape[1], *repeated.shape[2:])
                repeated = torch.cat((repeated, padding), dim=1)
            repeated_docs.append(repeated[:, :seq_len])
        return torch.cat(repeated_docs, dim=1)

    @staticmethod
    def _prepend_zero_chunks(
        concept_states: torch.Tensor,
        concept_lengths: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        chunks: list[torch.Tensor] = []
        source_lengths: list[int] = []
        start = 0
        for concept_len in concept_lengths:
            state = concept_states[:, start : start + concept_len]
            start += concept_len
            chunks.append(torch.cat((torch.zeros_like(state[:, :1]), state), dim=1))
            source_lengths.append(concept_len + 1)
        return torch.cat(chunks, dim=1), source_lengths

    @staticmethod
    def _build_concept_context(
        concept_hidden: torch.Tensor,
        concept_lengths: list[int],
        chunk_size: int,
    ) -> SequenceContext:
        device = concept_hidden.device
        cu_seq_lens = torch.tensor(
            [0, *accumulate(concept_lengths)],
            dtype=torch.int32,
            device=device,
        )
        position_ids = torch.cat(
            [torch.arange(length, device=device) * chunk_size for length in concept_lengths]
        ).unsqueeze(0)
        return SequenceContext(
            input_ids=None,
            inputs_embeds=cast(torch.FloatTensor, concept_hidden),
            cu_seq_lens_q=cast(torch.IntTensor, cu_seq_lens),
            cu_seq_lens_k=cast(torch.IntTensor, cu_seq_lens),
            max_length_q=max(concept_lengths),
            max_length_k=max(concept_lengths),
            position_ids=cast(torch.LongTensor, position_ids),
            device=device,
        )

    def forward(
        self,
        seq_ctx: SequenceContext,
        loss_ctx: dict[str, BaseLossContext | list[BaseLossContext]] | None = None,
    ) -> ModelOutputs:
        if seq_ctx.input_ids is not None:
            hidden_states = self.embed_tokens(seq_ctx.input_ids)
        elif seq_ctx.inputs_embeds is not None:
            hidden_states = seq_ctx.inputs_embeds
        else:
            raise ValueError("SequenceContext must provide input_ids or inputs_embeds")
        if hidden_states.shape[0] != 1:
            raise ValueError("Concept OLMo expects XTuner packed input with batch dimension 1")
        if seq_ctx.position_ids is None:
            raise ValueError("SequenceContext.position_ids is required")

        self._mark_dynamic(seq_ctx)
        token_lengths = self._lengths(seq_ctx)
        token_position_embeddings = self.rotary_emb(hidden_states, seq_ctx.position_ids)

        encoder_history = [hidden_states]
        encoder_raw_states: list[torch.Tensor] = []
        for stage in self.encoder_stages:
            hidden_states, raw = stage(
                hidden_states,
                token_position_embeddings,
                seq_ctx,
                encoder_history,
            )
            encoder_raw_states.append(raw)
            encoder_history.append(raw)

        concept_hidden, concept_lengths = self._merge_chunks(hidden_states, token_lengths)
        concept_hidden = self.concept_vq_input_norm(concept_hidden)
        merged_encoder_states = [self._merge_chunks(state, token_lengths)[0] for state in encoder_raw_states[:-1]]
        encoder_concept_states = self.concept_read_encoder_shared_source_norm(
            torch.stack(merged_encoder_states, dim=2)
        )
        concept_ctx = self._build_concept_context(
            concept_hidden,
            concept_lengths,
            self.config.concept_chunk_size,
        )
        assert concept_ctx.position_ids is not None
        concept_position_embeddings = self.rotary_emb(concept_hidden, concept_ctx.position_ids)

        concept_history = [concept_hidden]
        concept_raw_states: list[torch.Tensor] = []
        for stage in self.concept_stages:
            concept_hidden, raw = stage(
                concept_hidden,
                concept_position_embeddings,
                concept_ctx,
                concept_history,
                encoder_concept_states,
            )
            concept_raw_states.append(raw)
            concept_history.append(raw)
        concept_hidden = self.concept_final_norm(concept_hidden)
        concept_logits = torch.stack([head(concept_hidden) for head in self.prediction_heads], dim=2)
        codebook = self.concept_quantizer.transformed_codebook().to(concept_logits.dtype)
        predicted_vectors = torch.einsum("bsck,ckd->bscd", concept_logits, codebook).reshape(
            concept_logits.shape[0], concept_logits.shape[1], self.config.hidden_size
        )
        repeated_concepts = self._repeat_shift(predicted_vectors, concept_lengths, token_lengths)

        hidden_states = self.fusion_tok_norm(hidden_states) + self.fusion_norm_alpha.to(
            hidden_states.dtype
        ) * self.fusion_hl_norm(repeated_concepts.to(hidden_states.dtype))
        decoder_encoder_states = self.decoder_read_encoder_shared_source_norm(torch.stack(encoder_raw_states, dim=2))
        packed_concept_states, decoder_concept_lengths = self._prepend_zero_chunks(
            torch.stack(concept_raw_states, dim=2),
            concept_lengths,
        )
        packed_concept_states = self.decoder_read_concept_shared_source_norm(packed_concept_states)
        decoder_concept_states = self._repeat_shift(
            packed_concept_states,
            decoder_concept_lengths,
            token_lengths,
            prepend_zero=False,
        )

        decoder_history = [hidden_states]
        for layer_idx, stage in enumerate(self.decoder_stages):
            gate = self.final_read_concept_gate_logits[layer_idx].float().softmax(dim=-1)
            hidden_states, raw = stage(
                hidden_states,
                token_position_embeddings,
                seq_ctx,
                decoder_history,
                repeated_concepts,
                decoder_encoder_states,
                decoder_concept_states,
                gate,
            )
            decoder_history.append(raw)
        hidden_states = self.final_norm(hidden_states)

        output: dict = {}
        if self.config.return_hidden_states:
            output["hidden_states"] = [hidden_states]
        if loss_ctx is None:
            _, (logits, _) = self.lm_head(hidden_states, None)
            output["logits"] = logits
        else:
            loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx["lm"])  # type: ignore[call-overload]
            output["loss"] = loss
            output["logits"] = logits
            output["extra_info"] = extra_info
        return ModelOutputs(**output)


class ConceptLMV22VQConfig(TransformerConfig):
    model_type: str | None = "conceptlm_v22_vq"
    bos_token_id: int = 100257
    eos_token_id: int = 100257
    pad_token_id: int | None = 100277
    concept_encoder_layers: int
    concept_decoder_layers: int
    concept_special_layers: int
    concept_chunk_size: int
    concept_shift_feature: bool
    concept_chunk_merge_method: Literal["meanpooling", "first", "last"]
    concept_v22_vq_codebook_size: int
    concept_v22_vq_num_codebooks: int
    concept_fusion_norm_alpha_init: float
    concept_dd_two_route_add_decoder_use_softmax: bool
    window_size: tuple[int, int]
    window_attn_skip_freq: int

    @model_validator(mode="after")
    def _validate_supported_variant(self) -> Self:
        if self.num_hidden_layers != self.concept_encoder_layers + self.concept_decoder_layers:
            raise ValueError("num_hidden_layers must equal encoder layers plus decoder layers")
        if self.attention.num_attention_heads != self.attention.num_key_value_heads:
            raise ValueError("Concept OLMo grouped-query attention is not supported")
        if self.attention.head_dim * self.attention.num_attention_heads != self.hidden_size:
            raise ValueError("Concept OLMo requires a full hidden-sized Q/K/V projection")
        if self.window_size[1] != 0:
            raise ValueError("Concept OLMo only supports a left causal attention window")
        return self

    def build(self) -> ConceptLMV22VQ:
        return ConceptLMV22VQ(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        from transformers import AutoConfig, AutoTokenizer

        hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        if hf_config.model_type != "conceptlm_v22_vq":
            raise ValueError(f"Expected conceptlm_v22_vq, got {hf_config.model_type}")
        unsupported = {
            "num_query_groups": hf_config.num_query_groups == hf_config.num_attention_heads,
            "rotary_percent": hf_config.rotary_percent == 1.0,
            "rotary_interleaved": not hf_config.rotary_interleaved,
            "position_embedding_type": hf_config.position_embedding_type == "rope",
            "gated_linear_unit": hf_config.gated_linear_unit,
            "activation_function": hf_config.activation_function == "silu",
            "concept_v22_vq_merge_mode": hf_config.concept_v22_vq_merge_mode == "raw_logits",
            "concept_layer_norm_option": hf_config.concept_layer_norm_option == "normed_add",
            "concept_dd_two_route_add": hf_config.concept_dd_two_route_add,
            "concept_enable_concept_read_encoder": hf_config.concept_enable_concept_read_encoder,
            "concept_enable_decoder_read_encoder": hf_config.concept_enable_decoder_read_encoder,
            "concept_enable_decoder_read_concept": hf_config.concept_enable_decoder_read_concept,
            "concept_final_read_concept_gate": hf_config.concept_final_read_concept_gate,
        }
        disabled = [name for name, enabled in unsupported.items() if not enabled]
        if disabled:
            raise ValueError(f"Unsupported ConceptLM V2.2 options: {disabled}")

        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        return cls(
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            bos_token_id=getattr(hf_config, "bos_token_id", None) or tokenizer.bos_token_id,
            eos_token_id=getattr(hf_config, "eos_token_id", None) or tokenizer.eos_token_id,
            pad_token_id=getattr(hf_config, "pad_token_id", None) or tokenizer.pad_token_id,
            num_hidden_layers=hf_config.num_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.ffn_hidden_size,
            rms_norm_eps=hf_config.layernorm_epsilon,
            hidden_act=hf_config.activation_function,
            attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=hf_config.num_query_groups,
                head_dim=hf_config.kv_channels,
                dropout=hf_config.attention_dropout,
                qk_norm=True,
                rms_norm_eps=hf_config.layernorm_epsilon,
                sliding_window=int(hf_config.window_size[0]),
            ),
            rope_parameters_cfg=RopeParametersConfig.from_hf_config(hf_config),
            tie_word_embeddings=hf_config.tie_word_embeddings,
            use_sliding_window=True,
            concept_encoder_layers=hf_config.concept_encoder_layers,
            concept_decoder_layers=hf_config.concept_decoder_layers,
            concept_special_layers=hf_config.concept_special_layers,
            concept_chunk_size=hf_config.concept_chunk_size,
            concept_shift_feature=hf_config.concept_shift_feature,
            concept_chunk_merge_method=hf_config.concept_chunk_merge_method,
            concept_v22_vq_codebook_size=hf_config.concept_v22_vq_codebook_size,
            concept_v22_vq_num_codebooks=hf_config.concept_v22_vq_num_codebooks,
            concept_fusion_norm_alpha_init=hf_config.concept_fusion_norm_alpha_init,
            concept_dd_two_route_add_decoder_use_softmax=(hf_config.concept_dd_two_route_add_decoder_use_softmax),
            window_size=tuple(hf_config.window_size),
            window_attn_skip_freq=hf_config.window_attn_skip_freq,
        )

    @property
    def hf_config(self):
        return None


class ConceptOlmoStage2V1Config(ConceptLMV22VQConfig):
    vocab_size: int = 100278
    max_position_embeddings: int = 8192
    num_hidden_layers: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    attention: MHAConfig = MHAConfig(
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        dropout=0.0,
        qk_norm=True,
        rms_norm_eps=1e-6,
        sliding_window=4096,
    )
    rope_parameters_cfg: RopeParametersConfig = Field(
        default_factory=lambda: RopeParametersConfig(rope_theta=500000.0)
    )
    tie_word_embeddings: bool = False
    use_sliding_window: bool = True
    concept_encoder_layers: int = 16
    concept_decoder_layers: int = 16
    concept_special_layers: int = 8
    concept_chunk_size: int = 4
    concept_shift_feature: bool = True
    concept_chunk_merge_method: Literal["meanpooling", "first", "last"] = "meanpooling"
    concept_v22_vq_codebook_size: int = 128
    concept_v22_vq_num_codebooks: int = 32
    concept_fusion_norm_alpha_init: float = 0.1
    concept_dd_two_route_add_decoder_use_softmax: bool = True
    window_size: tuple[int, int] = (4096, 0)
    window_attn_skip_freq: int = 4
