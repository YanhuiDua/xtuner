# Copyright (c) OpenMMLab. All rights reserved.

import os
import re
from contextlib import AbstractContextManager, nullcontext
from itertools import accumulate
from pathlib import Path
from typing import Literal, Self, cast

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
from pydantic import Field, model_validator
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from tqdm import tqdm
from typing_extensions import override

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.float8_handler import Float8Handler
from xtuner.v1.loss import BaseLossContext
from xtuner.v1.model.base import (
    DEFAULT_FLOAT8_CFG,
    BaseModel,
    ModelOutputs,
    TorchCompileOption,
    TransformerConfig,
)
from xtuner.v1.model.utils import checkpoint_wrapper
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
from xtuner.v1.utils import get_device, log_rank0
from xtuner.v1.utils.activation_offload import async_save_on_cpu


DEVICE = get_device()
CONCEPT_OLMO_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.concept_olmo.EncoderStage.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.concept_olmo.ConceptStage.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.concept_olmo.DecoderStage.forward": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


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
                    DepthDD(config.hidden_size, layer_idx, use_softmax=False),
                )
                for layer_idx in range(config.concept_encoder_layers)
            ]
        )
        self.concept_stages = nn.ModuleList(
            [
                ConceptStage(
                    self._build_transformer(layer_idx),
                    DepthDD(config.hidden_size, layer_idx, use_softmax=False),
                    ResidualRoute(
                        config.hidden_size,
                        config.concept_encoder_layers - 1,
                        use_softmax=True,
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
                    ),
                    ConceptRoute(config.hidden_size, config.rms_norm_eps),
                    ResidualRoute(
                        config.hidden_size,
                        config.concept_encoder_layers,
                        use_softmax=True,
                    ),
                    ResidualRoute(
                        config.hidden_size,
                        config.concept_special_layers,
                        use_softmax=True,
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
        self.fusion_norm_alpha = nn.Parameter(torch.tensor([config.concept_fusion_norm_alpha_init]))
        self.decoder_read_encoder_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_read_concept_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_read_concept_gate_logits = nn.Parameter(torch.zeros(config.concept_decoder_layers, 2))
        self.final_norm = CheckpointRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)

        self._init_load_spec()
        self._maybe_enable_compile(self.compile_cfg)
        self.offload_stream: torch.cuda.Stream | None = None

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

    @override
    def safetensors_to_params(
        self,
        safetensors: list[torch.Tensor],
        local_tensor: torch.Tensor,
        param_name: str,
        start: int | None,
        end: int | None,
        dim: int | None,
    ) -> None:
        if param_name == "fusion_norm_alpha":
            safetensors = [safetensors[0].reshape(1)]
        super().safetensors_to_params(
            safetensors=safetensors,
            local_tensor=local_tensor,
            param_name=param_name,
            start=start,
            end=end,
            dim=dim,
        )

    @override
    def param_to_safetensor(self, safetensor: torch.Tensor, hf_param_name: str) -> torch.Tensor:
        if hf_param_name == "fusion_norm_alpha":
            return safetensor.squeeze(0)
        return safetensor

    @staticmethod
    def _lengths(seq_ctx: SequenceContext) -> list[int]:
        lengths = seq_ctx.cu_seq_lens_q[1:] - seq_ctx.cu_seq_lens_q[:-1]
        return [int(length) for length in lengths.detach().cpu().tolist()]

    @staticmethod
    def _gather_sequence(hidden_states: torch.Tensor, sp_mesh: DeviceMesh | None) -> torch.Tensor:
        if sp_mesh is None or sp_mesh.size() == 1:
            return hidden_states
        gathered = distF.all_gather(hidden_states.contiguous(), group=sp_mesh.get_group())
        return torch.cat(gathered, dim=1).contiguous()

    @staticmethod
    def _pad_and_split_sequence(
        hidden_states: torch.Tensor,
        sp_mesh: DeviceMesh | None,
    ) -> tuple[torch.Tensor, int]:
        if sp_mesh is None or sp_mesh.size() == 1:
            return hidden_states, 0
        pad_size = (-hidden_states.shape[1]) % sp_mesh.size()
        if pad_size:
            padding = hidden_states.new_zeros(hidden_states.shape[0], pad_size, *hidden_states.shape[2:])
            hidden_states = torch.cat((hidden_states, padding), dim=1)
        shard_size = hidden_states.shape[1] // sp_mesh.size()
        shard_start = sp_mesh.get_local_rank() * shard_size
        return hidden_states[:, shard_start : shard_start + shard_size].contiguous(), pad_size

    @classmethod
    def _apply_sequence_module(
        cls,
        hidden_states: torch.Tensor,
        module: nn.Module,
        sp_mesh: DeviceMesh | None,
    ) -> torch.Tensor:
        """Apply a token-wise module once across SP ranks and restore the full
        sequence."""
        valid_length = hidden_states.shape[1]
        local_states, _ = cls._pad_and_split_sequence(hidden_states, sp_mesh)
        local_states = module(local_states)
        return cls._gather_sequence(local_states, sp_mesh)[:, :valid_length]

    @staticmethod
    def _local_token_shard(
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        shard_size: int,
    ) -> torch.Tensor:
        sp_mesh = seq_ctx.sequence_parallel_mesh
        if sp_mesh is None or sp_mesh.size() == 1:
            return hidden_states
        if seq_ctx._shard_size:
            shard_start = seq_ctx._shard_start
        else:
            shard_start = seq_ctx.sp_rank * shard_size
        return hidden_states[:, shard_start : shard_start + shard_size].contiguous()

    def _activation_offload_context(
        self,
        hidden_states: torch.Tensor,
        block_idx: int,
        group: str,
    ) -> AbstractContextManager:
        if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) != 1:
            return nullcontext()
        if hidden_states.device.type != "cuda":
            raise RuntimeError("XTUNER_ACTIVATION_OFFLOAD=1 requires CUDA activations")
        if self.offload_stream is None:
            self.offload_stream = torch.cuda.Stream()
        return async_save_on_cpu(
            h2d_stream=self.offload_stream,
            d2h_stream=self.offload_stream,
            block_idx=block_idx,
            group=f"concept_olmo_{group}",
            custom_check_fn=lambda tensor: tensor.data_ptr() == hidden_states.data_ptr(),
            prefetch=True,
            reserve_pin_memory=True,
        )

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
        sp_mesh: DeviceMesh | None = None,
        concept_padding: int = 0,
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
        shard_start = 0
        shard_size = concept_hidden.shape[1]
        num_padding = 0
        if sp_mesh is not None and sp_mesh.size() > 1:
            shard_start = sp_mesh.get_local_rank() * shard_size
            position_ids = position_ids[:, shard_start : shard_start + shard_size].contiguous()
            valid_length = int(cu_seq_lens[-1]) - concept_padding
            num_padding = max(0, min(shard_size, shard_start + shard_size - valid_length))
        return SequenceContext(
            input_ids=None,
            inputs_embeds=cast(torch.FloatTensor, concept_hidden),
            cu_seq_lens_q=cast(torch.IntTensor, cu_seq_lens),
            cu_seq_lens_k=cast(torch.IntTensor, cu_seq_lens),
            max_length_q=max(concept_lengths),
            max_length_k=max(concept_lengths),
            position_ids=cast(torch.LongTensor, position_ids),
            device=device,
            num_padding=num_padding,
            sequence_parallel_mesh=sp_mesh,
            shard_start=shard_start,
            shard_size=shard_size,
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

        encoder_history = hidden_states.unsqueeze(2)
        encoder_raw_states: list[torch.Tensor] = []
        for stage_idx, stage in enumerate(self.encoder_stages):
            with self._activation_offload_context(hidden_states, stage_idx, "encoder"):
                hidden_states, raw = stage(
                    hidden_states,
                    token_position_embeddings,
                    seq_ctx,
                    encoder_history,
                )
            encoder_raw_states.append(raw)
            encoder_history = torch.cat((encoder_history, raw.unsqueeze(2)), dim=2)

        sp_mesh = seq_ctx.sequence_parallel_mesh
        full_encoder_hidden = self._gather_sequence(hidden_states, sp_mesh)
        full_encoder_raw_states = [self._gather_sequence(state, sp_mesh) for state in encoder_raw_states]
        concept_hidden, concept_lengths = self._merge_chunks(full_encoder_hidden, token_lengths)
        valid_concept_length = concept_hidden.shape[1]
        concept_hidden, concept_padding = self._pad_and_split_sequence(concept_hidden, sp_mesh)
        concept_hidden = self.concept_vq_input_norm(concept_hidden)
        merged_encoder_state_list = [
            self._merge_chunks(state, token_lengths)[0] for state in full_encoder_raw_states[:-1]
        ]
        merged_encoder_states = torch.stack(merged_encoder_state_list, dim=2)
        merged_encoder_states, encoder_concept_padding = self._pad_and_split_sequence(
            merged_encoder_states,
            sp_mesh,
        )
        if encoder_concept_padding != concept_padding:
            raise RuntimeError("Concept hidden states and encoder sources received different SP padding")
        encoder_concept_states = self.concept_read_encoder_shared_source_norm(merged_encoder_states)
        padded_concept_lengths = [*concept_lengths, concept_padding] if concept_padding else concept_lengths
        concept_ctx = self._build_concept_context(
            concept_hidden,
            padded_concept_lengths,
            self.config.concept_chunk_size,
            sp_mesh,
            concept_padding,
        )
        assert concept_ctx.position_ids is not None
        concept_position_embeddings = self.rotary_emb(concept_hidden, concept_ctx.position_ids)

        concept_history = concept_hidden.unsqueeze(2)
        concept_raw_states: list[torch.Tensor] = []
        for stage_idx, stage in enumerate(self.concept_stages):
            with self._activation_offload_context(concept_hidden, stage_idx, "concept"):
                concept_hidden, raw = stage(
                    concept_hidden,
                    concept_position_embeddings,
                    concept_ctx,
                    concept_history,
                    encoder_concept_states,
                )
            concept_raw_states.append(raw)
            concept_history = torch.cat((concept_history, raw.unsqueeze(2)), dim=2)
        concept_hidden = self.concept_final_norm(concept_hidden)
        concept_logits = torch.stack([head(concept_hidden) for head in self.prediction_heads], dim=2)
        codebook = self.concept_quantizer.transformed_codebook().to(concept_logits.dtype)
        predicted_vectors = torch.einsum("bsck,ckd->bscd", concept_logits, codebook).reshape(
            concept_logits.shape[0], concept_logits.shape[1], self.config.hidden_size
        )
        predicted_vectors = self._gather_sequence(predicted_vectors, sp_mesh)[:, :valid_concept_length]
        repeated_concepts = self._repeat_shift(predicted_vectors, concept_lengths, token_lengths)
        repeated_concepts = self._local_token_shard(repeated_concepts, seq_ctx, hidden_states.shape[1])

        hidden_states = self.fusion_tok_norm(hidden_states) + self.fusion_norm_alpha.to(
            hidden_states.dtype
        ) * self.fusion_hl_norm(repeated_concepts.to(hidden_states.dtype))
        decoder_encoder_states = self.decoder_read_encoder_shared_source_norm(torch.stack(encoder_raw_states, dim=2))
        full_concept_raw_states = [
            self._gather_sequence(state, sp_mesh)[:, :valid_concept_length] for state in concept_raw_states
        ]
        packed_concept_states, decoder_concept_lengths = self._prepend_zero_chunks(
            torch.stack(full_concept_raw_states, dim=2),
            concept_lengths,
        )
        packed_concept_states = self._apply_sequence_module(
            packed_concept_states,
            self.decoder_read_concept_shared_source_norm,
            sp_mesh,
        )
        decoder_concept_states = self._repeat_shift(
            packed_concept_states,
            decoder_concept_lengths,
            token_lengths,
            prepend_zero=False,
        )
        decoder_concept_states = self._local_token_shard(decoder_concept_states, seq_ctx, hidden_states.shape[1])

        decoder_history = hidden_states.unsqueeze(2)
        for layer_idx, stage in enumerate(self.decoder_stages):
            with self._activation_offload_context(hidden_states, layer_idx, "decoder"):
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
            decoder_history = torch.cat((decoder_history, raw.unsqueeze(2)), dim=2)
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

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        return CONCEPT_OLMO_COMPILE_CFG

    @override
    def fully_shard(self, fsdp_config: FSDPConfig) -> Self:
        self.fsdp_config = fsdp_config
        self._init_device_mesh(fsdp_config)
        assert self.fsdp_mesh is not None

        if self.config.float8_cfg is not None:
            Float8Handler.pad_for_fsdp(self, self.fsdp_mesh, callback_after_pad=self._init_load_spec)

        checkpoint_preserve_rng_state = fsdp_config.checkpoint_preserve_rng_state
        if not checkpoint_preserve_rng_state and self.config.attention.dropout > 0.0:
            checkpoint_preserve_rng_state = True
            log_rank0.warning("checkpoint_preserve_rng_state was enabled because attention dropout is nonzero")

        if fsdp_config.requires_grad:
            for module in self.modules():
                for parameter_name, parameter in module.named_parameters(recurse=False):
                    if parameter.requires_grad:
                        setattr(module, parameter_name, nn.Parameter(parameter.to(dtype=torch.float32)))
        else:
            for parameter in self.parameters():
                parameter.requires_grad = False

        mp_policy = MixedPrecisionPolicy(
            param_dtype=fsdp_config.param_dtype,
            reduce_dtype=fsdp_config.reduce_dtype,
        )
        lm_head_policy = (
            MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
            if fsdp_config.fp32_lm_head
            else mp_policy
        )
        shard_mesh = self.fsdp_mesh if self.hsdp_mesh is None else self.hsdp_mesh
        offload_policy = CPUOffloadPolicy() if fsdp_config.cpu_offload else None

        stage_groups = (self.encoder_stages, self.concept_stages, self.decoder_stages)
        stage_locations = [(group, stage_idx) for group in stage_groups for stage_idx in range(len(group))]
        num_recompute_stages = int(len(stage_locations) * fsdp_config.recompute_ratio)
        generator = torch.Generator()
        generator.manual_seed(dist.get_rank())
        shuffled_indices = torch.randperm(len(stage_locations), generator=generator)
        for global_idx_tensor in tqdm(shuffled_indices, desc="[Concept OLMo FSDP Sharding]"):
            global_idx = int(global_idx_tensor)
            group, stage_idx = stage_locations[global_idx]
            stage = group[stage_idx]
            if global_idx < num_recompute_stages:
                stage = checkpoint_wrapper(
                    stage,
                    preserve_rng_state=checkpoint_preserve_rng_state,
                    checkpoint_impl=CheckpointImpl.REENTRANT,
                )
            group[stage_idx] = stage
            self._fully_shard(
                mesh=shard_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=fsdp_config.reshard_after_forward,
                offload_policy=offload_policy,
                module=stage,
            )

        ordered_stages = [stage for group in stage_groups for stage in group]
        for current_stage, next_stage in zip(ordered_stages[:-1], ordered_stages[1:], strict=True):
            current_stage.set_modules_to_forward_prefetch([next_stage])  # type: ignore[attr-defined]

        shared_modules: tuple[nn.Module, ...] = (
            self.embed_tokens,
            self.concept_vq_input_norm,
            self.concept_final_norm,
            *self.prediction_heads,
            self.concept_read_encoder_shared_source_norm,
            self.fusion_tok_norm,
            self.fusion_hl_norm,
            self.decoder_read_encoder_shared_source_norm,
            self.decoder_read_concept_shared_source_norm,
            self.final_norm,
        )
        for module in shared_modules:
            self._fully_shard(
                mesh=shard_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=fsdp_config.reshard_after_forward,
                offload_policy=offload_policy,
                module=module,
            )
        self._fully_shard(
            mesh=shard_mesh,
            mp_policy=lm_head_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=offload_policy,
            module=self.lm_head,
        )
        self._fully_shard(
            mesh=shard_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=offload_policy,
        )
        self.set_modules_to_forward_prefetch([self.embed_tokens, self.encoder_stages[0]])  # type: ignore[operator]
        self._to_empty_meta()
        return self

    def _init_device_mesh(self, fsdp_config: FSDPConfig) -> None:
        world_size = dist.get_world_size()
        prefix = self.config.mesh_prefix
        if fsdp_config.hsdp_sharding_size is None:
            model_mesh = init_device_mesh(
                DEVICE,
                (world_size, 1),
                mesh_dim_names=(f"{prefix}.fsdp", f"{prefix}.others"),
            )
            self.fsdp_mesh = model_mesh[f"{prefix}.fsdp"]
        else:
            self.hsdp_mesh = init_device_mesh(
                DEVICE,
                (
                    world_size // fsdp_config.hsdp_sharding_size,
                    fsdp_config.hsdp_sharding_size,
                ),
                mesh_dim_names=(f"{prefix}.hsdp_replicate", f"{prefix}.hsdp_shard"),
            )
            self.fsdp_mesh = self.hsdp_mesh[f"{prefix}.hsdp_shard"]


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
