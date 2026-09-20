# Copyright (c) OpenMMLab. All rights reserved.

import os
import re
from contextlib import AbstractContextManager, nullcontext
from itertools import accumulate
from pathlib import Path
from typing import Any, Literal, Self, cast

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
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
from xtuner.v1.model.utils import apply_activation_checkpointing
from xtuner.v1.module import (
    CheckpointRMSNorm,
    ConceptQuantizer,
    ConceptRoute,
    ConceptStage,
    ConceptStageGroup,
    DecoderStage,
    DecoderStageGroup,
    DepthDD,
    EncoderStage,
    EncoderStageGroup,
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
    "xtuner.v1.loss.ce_loss.LMHeadLossContext.loss_fn": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}
CONCEPT_OLMO_GROUPED_COMPILE_CFG: dict[str, TorchCompileOption] = {
    "xtuner.v1.module.concept_olmo.EncoderStageGroup.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.concept_olmo.ConceptStageGroup.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.module.concept_olmo.DecoderStageGroup.forward": TorchCompileOption(fullgraph=True),
    "xtuner.v1.loss.ce_loss.LMHeadLossContext.loss_fn": TorchCompileOption(fullgraph=True),
    **DEFAULT_FLOAT8_CFG,
}


class ConceptLMV22VQ(BaseModel):
    """XTuner implementation of the ConceptLM V2.2 residual-VQ model."""

    config: "ConceptLMV22VQConfig"

    def __init__(self, config: "ConceptLMV22VQConfig"):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.rotary_emb = self.build_rotary_embedding(config)

        encoder_stages = [
            EncoderStage(
                self._build_transformer(layer_idx),
                DepthDD(config.hidden_size, layer_idx, use_softmax=False),
            )
            for layer_idx in range(config.concept_encoder_layers)
        ]
        concept_stages = [
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
        decoder_stages = [
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
        if config.stage_group_size == 1:
            self.encoder_stages = nn.ModuleList(encoder_stages)
            self.concept_stages = nn.ModuleList(concept_stages)
            self.decoder_stages = nn.ModuleList(decoder_stages)
        else:
            group_size = config.stage_group_size
            self.encoder_stage_groups = nn.ModuleList(
                [
                    EncoderStageGroup(list(enumerate(encoder_stages[start : start + group_size], start=start)))
                    for start in range(0, len(encoder_stages), group_size)
                ]
            )
            self.concept_stage_groups = nn.ModuleList(
                [
                    ConceptStageGroup(list(enumerate(concept_stages[start : start + group_size], start=start)))
                    for start in range(0, len(concept_stages), group_size)
                ]
            )
            self.decoder_stage_groups = nn.ModuleList(
                [
                    DecoderStageGroup(list(enumerate(decoder_stages[start : start + group_size], start=start)))
                    for start in range(0, len(decoder_stages), group_size)
                ]
            )

        self.concept_vq_input_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.concept_quantizer = ConceptQuantizer(
            config.hidden_size,
            config.concept_v22_vq_num_codebooks,
            config.concept_v22_vq_codebook_size,
        )
        self.concept_final_norm = CheckpointRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.prediction_heads = build_linear(
            config.hidden_size,
            config.concept_v22_vq_num_codebooks * config.concept_v22_vq_codebook_size,
            bias=True,
            float8_cfg=config.float8_cfg,
        )
        self.concept_read_encoder_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.fusion_tok_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fusion_hl_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fusion_norm_alpha = nn.Parameter(torch.tensor([config.concept_fusion_norm_alpha_init]))
        self.decoder_read_encoder_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_read_concept_shared_source_norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_read_concept_gate_logits = nn.Parameter(torch.zeros(config.concept_decoder_layers, 2))
        self.final_norm = CheckpointRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        lm_head_vocab_size = (
            (config.vocab_size + config.lm_head_vocab_padding_multiple - 1)
            // config.lm_head_vocab_padding_multiple
            * config.lm_head_vocab_padding_multiple
        )
        self.lm_head = LMHead(
            config.hidden_size,
            lm_head_vocab_size,
            bias=False,
            logical_out_features=config.vocab_size,
        )

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

    def _predict_concept_logits(self, concept_hidden: torch.Tensor) -> torch.Tensor:
        num_codebooks = self.config.concept_v22_vq_num_codebooks
        codebook_size = self.config.concept_v22_vq_codebook_size

        # A single physical parameter keeps prediction-head FSDP communication
        # in one collective. Use its single large GEMM in the optimized/FP8
        # path; for eager BF16 inference, retain the 32 independent HF GEMMs so
        # checkpoint parity keeps the original accumulation order.
        if self._compile_cfg or self.config.float8_cfg is not None:
            return self.prediction_heads(concept_hidden).reshape(
                *concept_hidden.shape[:-1],
                num_codebooks,
                codebook_size,
            )

        projection = cast(nn.Linear, self.prediction_heads)
        weight = projection.weight
        bias = projection.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
            if bias is not None:
                assert isinstance(bias, DTensor)
                bias = bias.to_local()
        assert bias is not None
        return torch.stack(
            [
                F.linear(concept_hidden, head_weight, head_bias)
                for head_weight, head_bias in zip(
                    weight.chunk(num_codebooks),
                    bias.chunk(num_codebooks),
                    strict=True,
                )
            ],
            dim=2,
        )

    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.model_type == "ncp_olmo3":
            if key.endswith("._extra_state"):
                return []
            if key == "concept_quantizer.codebook":
                return [
                    f"concept_quantizer.codebook.{index}"
                    for index in range(self.config.concept_v22_vq_num_codebooks)
                ]
            if key in {"prediction_heads.weight", "prediction_heads.bias"}:
                suffix = key.rsplit(".", 1)[-1]
                return [
                    f"concept_predictor.prediction_heads.{index}.{suffix}"
                    for index in range(self.config.concept_v22_vq_num_codebooks)
                ]
            mappings = (
                (r"^embed_tokens\.", "embed_tokens."),
                (r"^encoder_stages\.(\d+)\.transformer\.", r"encoder.layers.\1."),
                (r"^concept_stages\.(\d+)\.transformer\.", r"concept_predictor.hlm_block.layers.\1."),
                (r"^decoder_stages\.(\d+)\.transformer\.", r"decoder.layers.\1."),
                (r"^encoder_stages\.(\d+)\.depth_dd\.", r"dd_encoder_self_dd.depth_dds.\1."),
                (r"^concept_stages\.(\d+)\.depth_dd\.", r"concept_predictor.concept_self_dd.depth_dds.\1."),
                (r"^concept_stages\.(\d+)\.read_encoder_route\.", r"concept_predictor.concept_read_encoder_routes.\1."),
                (r"^decoder_stages\.(\d+)\.decoder_dd\.", r"dd_two_route_add.decoder_dds.\1."),
                (r"^decoder_stages\.(\d+)\.concept_route\.", r"dd_two_route_add.concept_routes.\1."),
                (r"^decoder_stages\.(\d+)\.read_encoder_route\.", r"decoder_read_encoder_routes.\1."),
                (r"^decoder_stages\.(\d+)\.read_concept_route\.", r"decoder_read_concept_routes.\1."),
                (r"^concept_final_norm\.", "concept_predictor.hlm_block.norm."),
                (r"^concept_read_encoder_shared_source_norm\.", "concept_predictor.concept_read_encoder_shared_source_norm."),
                (r"^final_norm\.", "decoder.norm."),
                (r"^lm_head\.", "lm_head."),
                (r"^decoder_read_encoder_shared_source_norm\.", "decoder_read_encoder_shared_source_norm."),
                (r"^decoder_read_concept_shared_source_norm\.", "decoder_read_concept_shared_source_norm."),
                (r"^fusion_tok_norm\.", "fusion_tok_norm."),
                (r"^fusion_hl_norm\.", "fusion_hl_norm."),
                (r"^fusion_norm_alpha$", "fusion_norm_alpha"),
                (r"^final_read_concept_gate_logits$", "final_read_concept_gate_logits"),
            )
            for pattern, replacement in mappings:
                if re.search(pattern, key):
                    mapped = re.sub(pattern, replacement, key)
                    mapped = mapped.replace("self_attention.", "self_attn.")
                    mapped = mapped.replace("q_layernorm.", "q_norm.")
                    mapped = mapped.replace("k_layernorm.", "k_norm.")
                    mapped = mapped.replace("linear_proj.", "o_proj.")
                    if mapped.endswith("linear_qkv.weight"):
                        prefix = mapped.removesuffix("linear_qkv.weight")
                        return [
                            prefix + "q_proj.weight",
                            prefix + "k_proj.weight",
                            prefix + "v_proj.weight",
                        ]
                    if mapped.endswith("linear_fc1.weight"):
                        prefix = mapped.removesuffix("linear_fc1.weight")
                        return [prefix + "gate_proj.weight", prefix + "up_proj.weight"]
                    return [mapped.replace("linear_fc2.", "down_proj.")]
            return [key]
        if key == "concept_quantizer.codebook":
            return [f"concept_quantizer.codebook.{index}" for index in range(self.config.concept_v22_vq_num_codebooks)]
        if key in {"prediction_heads.weight", "prediction_heads.bias"}:
            suffix = key.rsplit(".", 1)[-1]
            return [
                f"concept_predictor.prediction_heads.{index}.{suffix}"
                for index in range(self.config.concept_v22_vq_num_codebooks)
            ]
        mappings = (
            (r"^embed_tokens\.", "embedding.word_embeddings."),
            (r"^encoder_stage_groups\.\d+\.(\d+)\.transformer\.", r"encoder.layers.\1."),
            (r"^encoder_stage_groups\.\d+\.(\d+)\.depth_dd\.", r"dd_encoder_self_dd.depth_dds.\1."),
            (
                r"^concept_stage_groups\.\d+\.(\d+)\.transformer\.",
                r"concept_predictor.hlm_block.layers.\1.",
            ),
            (
                r"^concept_stage_groups\.\d+\.(\d+)\.depth_dd\.",
                r"concept_predictor.concept_self_dd.depth_dds.\1.",
            ),
            (
                r"^concept_stage_groups\.\d+\.(\d+)\.read_encoder_route\.",
                r"concept_predictor.concept_read_encoder_routes.\1.",
            ),
            (r"^decoder_stage_groups\.\d+\.(\d+)\.transformer\.", r"decoder.layers.\1."),
            (
                r"^decoder_stage_groups\.\d+\.(\d+)\.decoder_dd\.",
                r"dd_two_route_add.decoder_dds.\1.",
            ),
            (
                r"^decoder_stage_groups\.\d+\.(\d+)\.concept_route\.",
                r"dd_two_route_add.concept_routes.\1.",
            ),
            (
                r"^decoder_stage_groups\.\d+\.(\d+)\.read_encoder_route\.",
                r"decoder_read_encoder_routes.\1.",
            ),
            (
                r"^decoder_stage_groups\.\d+\.(\d+)\.read_concept_route\.",
                r"decoder_read_concept_routes.\1.",
            ),
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
    def hf_tensor_to_canonical(self, name: str, loaded_tensor: torch.Tensor) -> torch.Tensor:
        if self.config.model_type == "ncp_olmo3" and name.endswith("self_attention.linear_qkv.weight"):
            hidden_size = self.config.hidden_size
            query, key, value = loaded_tensor.split(hidden_size, dim=0)
            num_heads = self.config.attention.num_attention_heads
            head_dim = self.config.attention.head_dim
            query = query.view(num_heads, head_dim, hidden_size)
            key = key.view(num_heads, head_dim, hidden_size)
            value = value.view(num_heads, head_dim, hidden_size)
            return torch.stack((query, key, value), dim=2).reshape_as(loaded_tensor)
        if name == "fusion_norm_alpha":
            return loaded_tensor.reshape(1)
        return loaded_tensor

    @override
    def _init_load_spec(self) -> None:
        super()._init_load_spec()
        if self.lm_head.out_features == self.config.vocab_size:
            return

        load_spec = self.load_spec_mapping["lm_head.weight"]
        assert load_spec.global_shape[0] == self.lm_head.out_features
        origin_shape = load_spec.origin_shape or load_spec.global_shape
        assert self.config.vocab_size <= origin_shape[0] <= load_spec.global_shape[0]
        load_spec.origin_shape = (self.config.vocab_size, *origin_shape[1:])

    @override
    def param_to_safetensor(self, safetensor: torch.Tensor, hf_param_name: str) -> torch.Tensor:
        if hf_param_name == "fusion_norm_alpha":
            return safetensor.squeeze(0)
        if hf_param_name == "output_layer.weight" and safetensor.shape[0] > self.config.vocab_size:
            return safetensor[: self.config.vocab_size]
        return safetensor

    @staticmethod
    def _lengths(seq_ctx: SequenceContext) -> list[int]:
        if seq_ctx.cu_seq_lens_q.numel() == 2 and seq_ctx.num_padding == 0:
            # SequenceContext keeps max_length_q on CPU specifically so the
            # common single-sequence case does not need a CUDA-to-CPU copy.
            return [int(seq_ctx.max_length_q.item())]
        lengths = [
            int(length)
            for length in (seq_ctx.cu_seq_lens_q[1:] - seq_ctx.cu_seq_lens_q[:-1]).detach().cpu().tolist()
        ]
        if seq_ctx.num_padding <= 0:
            return lengths

        # SequenceContext keeps SP padding in the physical packed tensor and
        # records it separately. Concept pooling must only see real tokens.
        physical_tokens = sum(lengths)
        valid_tokens = (
            int(seq_ctx.valid_sequence_length)
            if getattr(seq_ctx, "_valid_sequence_length_explicit", False)
            else physical_tokens - int(seq_ctx.num_padding)
        )
        if valid_tokens > physical_tokens:
            raise ValueError("SequenceContext.valid_sequence_length exceeds the physical sequence length")
        if valid_tokens < 0:
            raise ValueError("SequenceContext.num_padding exceeds the physical sequence length")
        valid_lengths: list[int] = []
        remaining = valid_tokens
        for length in lengths:
            valid_length = min(length, remaining)
            if valid_length > 0:
                valid_lengths.append(valid_length)
            remaining -= valid_length
            if remaining == 0:
                break
        if remaining != 0:
            raise ValueError("SequenceContext cumulative lengths do not contain all valid tokens")
        return valid_lengths

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
        if self.config.stage_group_size > 1:
            raise RuntimeError("stage_group_size > 1 currently does not support activation offload")
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

    @staticmethod
    def _cat_if_multiple(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        """Concatenate tensors without copying an already-contiguous
        singleton."""
        if len(tensors) == 1:
            return tensors[0].contiguous()
        return torch.cat(tensors, dim=dim)

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
            usable = (seq_len // chunk_size) * chunk_size
            if usable == 0:
                # The HF reference emits no concept when a document does not
                # contain a complete source chunk. Keep an empty segment so
                # packed token/concept alignment remains document-local.
                merged = state[:, :0]
            else:
                reshaped = state[:, :usable].reshape(1, usable // chunk_size, chunk_size, *state.shape[2:])
                if self.config.concept_chunk_merge_method == "first":
                    merged = reshaped[:, :, 0]
                elif self.config.concept_chunk_merge_method == "last":
                    merged = reshaped[:, :, -1]
                else:
                    merged = reshaped.mean(dim=2)
            chunks.append(merged)
            concept_lengths.append(merged.shape[1])
        if not chunks:
            return hidden_states.new_empty(1, 0, *hidden_states.shape[2:]), []
        return self._cat_if_multiple(chunks, dim=1), concept_lengths

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
        if not repeated_docs:
            return concept_states.new_empty(1, 0, *concept_states.shape[2:])
        return self._cat_if_multiple(repeated_docs, dim=1)

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
            if concept_len == 0:
                chunks.append(concept_states.new_zeros(1, 1, *concept_states.shape[2:]))
            else:
                chunks.append(torch.cat((torch.zeros_like(state[:, :1]), state), dim=1))
            source_lengths.append(concept_len + 1)
        if not chunks:
            return concept_states.new_empty(1, 0, *concept_states.shape[2:]), []
        return ConceptLMV22VQ._cat_if_multiple(chunks, dim=1), source_lengths

    @staticmethod
    def _pad_sequence_tail(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
        if hidden_states.shape[1] >= target_length:
            return hidden_states[:, :target_length]
        padding = hidden_states.new_zeros(
            hidden_states.shape[0],
            target_length - hidden_states.shape[1],
            *hidden_states.shape[2:],
        )
        return torch.cat((hidden_states, padding), dim=1)

    def _concept_auxiliary_losses(
        self,
        continuous_concepts: torch.Tensor,
        predicted_concepts: torch.Tensor,
        concept_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the paper's next-concept and product-VQ objectives.

        ``continuous_concepts`` contains the encoder target at concept
        resolution.  ``predicted_concepts[i]`` predicts the next concept, so
        the last concept of every packed document has no NCP target.  Padding
        and incomplete documents are absent from ``concept_lengths``/the
        tensors passed here.
        """
        zero = continuous_concepts.new_zeros(())
        ncp_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        start = 0
        for length in concept_lengths:
            if length > 1:
                ncp_parts.append(predicted_concepts[:, start : start + length - 1])
                target_parts.append(continuous_concepts[:, start + 1 : start + length])
            start += length

        if ncp_parts:
            ncp_target = torch.cat(target_parts, dim=1)
            if self.config.concept_vq_detach_hlm_target:
                ncp_target = ncp_target.detach()
            if self.config.concept_hlm_loss_type != "mse":
                raise ValueError(f"Unsupported concept HLM loss type: {self.config.concept_hlm_loss_type}")
            ncp_loss = F.mse_loss(torch.cat(ncp_parts, dim=1).float(), ncp_target.float())
        else:
            ncp_loss = zero

        if continuous_concepts.shape[1] == 0:
            return ncp_loss, zero

        codebook = self.concept_quantizer.transformed_codebook().to(continuous_concepts.dtype)
        target = continuous_concepts.view(
            continuous_concepts.shape[0],
            continuous_concepts.shape[1],
            self.config.concept_v22_vq_num_codebooks,
            self.concept_quantizer.code_dim,
        )
        distances = (target.unsqueeze(3) - codebook.unsqueeze(0).unsqueeze(0)).pow(2).sum(dim=-1)
        indices = distances.argmin(dim=-1)
        quantized = torch.gather(
            codebook.unsqueeze(0).unsqueeze(0).expand(target.shape[0], target.shape[1], -1, -1, -1),
            dim=3,
            index=indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, self.concept_quantizer.code_dim),
        ).squeeze(3)
        quantized = quantized.reshape_as(continuous_concepts)
        codebook_loss = F.mse_loss(quantized.float(), continuous_concepts.detach().float())
        commitment_loss = F.mse_loss(continuous_concepts.float(), quantized.detach().float())
        vq_loss = codebook_loss + self.config.concept_vq_commitment_cost * commitment_loss
        return ncp_loss, vq_loss

    @staticmethod
    def _build_concept_context(
        concept_hidden: torch.Tensor,
        concept_lengths: list[int],
        chunk_size: int,
        sp_mesh: DeviceMesh | None = None,
        concept_padding: int = 0,
    ) -> SequenceContext:
        device = concept_hidden.device
        use_pinned_metadata = device.type == "cuda"
        cu_seq_lens_cpu = torch.tensor(
            [0, *accumulate(concept_lengths)],
            dtype=torch.int32,
            pin_memory=use_pinned_metadata,
        )
        cu_seq_lens = cu_seq_lens_cpu.to(device, non_blocking=use_pinned_metadata)
        # Concept positions index the pooled concept sequence, not the source
        # token sequence.  The reference NCP implementation therefore uses
        # 0, 1, 2, ... for each document even when each concept represents
        # ``chunk_size`` source tokens.  Keep document-local resets for packed
        # inputs; sequence-parallel sharding is applied below.
        position_id_chunks = [torch.arange(length, device=device) for length in concept_lengths]
        position_ids = ConceptLMV22VQ._cat_if_multiple(position_id_chunks, dim=0).unsqueeze(0)
        shard_start = 0
        shard_size = concept_hidden.shape[1]
        num_padding = 0
        if sp_mesh is not None and sp_mesh.size() > 1:
            shard_start = sp_mesh.get_local_rank() * shard_size
            position_ids = position_ids[:, shard_start : shard_start + shard_size].contiguous()
            valid_length = sum(concept_lengths) - concept_padding
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

        encoder_history: tuple[torch.Tensor, ...] = (hidden_states,)
        encoder_raw_states: list[torch.Tensor] = []
        if self.config.stage_group_size == 1:
            for stage_idx, stage in enumerate(self.encoder_stages):
                with self._activation_offload_context(hidden_states, stage_idx, "encoder"):
                    hidden_states, raw = stage(
                        hidden_states,
                        token_position_embeddings,
                        seq_ctx,
                        *encoder_history,
                    )
                encoder_raw_states.append(raw)
                encoder_history = (*encoder_history, raw)
        else:
            for group_idx, group in enumerate(self.encoder_stage_groups):
                stage_idx = group_idx * self.config.stage_group_size
                with self._activation_offload_context(hidden_states, stage_idx, "encoder"):
                    group_outputs = group(
                        hidden_states,
                        token_position_embeddings,
                        seq_ctx,
                        *encoder_history,
                    )
                hidden_states, *raw_states = group_outputs
                encoder_raw_states.extend(raw_states)
                encoder_history = (*encoder_history, *raw_states)

        sp_mesh = seq_ctx.sequence_parallel_mesh
        full_encoder_hidden = self._gather_sequence(hidden_states, sp_mesh)
        full_encoder_raw_states = [self._gather_sequence(state, sp_mesh) for state in encoder_raw_states]
        concept_hidden, concept_lengths = self._merge_chunks(full_encoder_hidden, token_lengths)
        continuous_concepts = concept_hidden
        valid_concept_length = concept_hidden.shape[1]
        active_concept_lengths = [length for length in concept_lengths if length > 0]
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
        concept_raw_states: list[torch.Tensor] = []
        if valid_concept_length > 0:
            padded_concept_lengths = (
                [*active_concept_lengths, concept_padding] if concept_padding else active_concept_lengths
            )
            concept_ctx = self._build_concept_context(
                concept_hidden,
                padded_concept_lengths,
                self.config.concept_chunk_size,
                sp_mesh,
                concept_padding,
            )
            assert concept_ctx.position_ids is not None
            concept_position_embeddings = self.rotary_emb(concept_hidden, concept_ctx.position_ids)

            concept_history: tuple[torch.Tensor, ...] = (concept_hidden,)
            if self.config.stage_group_size == 1:
                for stage_idx, stage in enumerate(self.concept_stages):
                    with self._activation_offload_context(concept_hidden, stage_idx, "concept"):
                        concept_hidden, raw = stage(
                            concept_hidden,
                            concept_position_embeddings,
                            concept_ctx,
                            encoder_concept_states,
                            *concept_history,
                        )
                    concept_raw_states.append(raw)
                    concept_history = (*concept_history, raw)
            else:
                for group_idx, group in enumerate(self.concept_stage_groups):
                    stage_idx = group_idx * self.config.stage_group_size
                    with self._activation_offload_context(concept_hidden, stage_idx, "concept"):
                        group_outputs = group(
                            concept_hidden,
                            concept_position_embeddings,
                            concept_ctx,
                            encoder_concept_states,
                            *concept_history,
                        )
                    concept_hidden, *raw_states = group_outputs
                    concept_raw_states.extend(raw_states)
                    concept_history = (*concept_history, *raw_states)
            concept_hidden = self.concept_final_norm(concept_hidden)
            concept_logits = self._predict_concept_logits(concept_hidden)
            codebook = self.concept_quantizer.transformed_codebook().to(concept_logits.dtype)
            predicted_vectors = torch.einsum("bsck,ckd->bscd", concept_logits, codebook).reshape(
                concept_logits.shape[0], concept_logits.shape[1], self.config.hidden_size
            )
        else:
            # Match the HF path for sequences without a complete concept
            # chunk: no Concept Module computation and an all-zero feedback
            # path. Raw states are still materialized for decoder routing.
            predicted_vectors = concept_hidden.new_empty(1, 0, self.config.hidden_size)
            concept_raw_states = [
                concept_hidden.new_empty(1, 0, self.config.hidden_size)
                for _ in range(self.config.concept_special_layers)
            ]
        predicted_vectors = self._gather_sequence(predicted_vectors, sp_mesh)[:, :valid_concept_length]
        ncp_loss: torch.Tensor | None = None
        vq_loss: torch.Tensor | None = None
        if loss_ctx is not None and (
            self.config.concept_hlm_loss_weight > 0 or self.config.concept_vq_loss_weight > 0
        ):
            ncp_loss, vq_loss = self._concept_auxiliary_losses(
                continuous_concepts,
                predicted_vectors,
                active_concept_lengths,
            )
        repeated_concepts = self._repeat_shift(predicted_vectors, concept_lengths, token_lengths)
        repeated_concepts = self._pad_sequence_tail(repeated_concepts, full_encoder_hidden.shape[1])
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
        decoder_concept_states = self._pad_sequence_tail(decoder_concept_states, full_encoder_hidden.shape[1])
        decoder_concept_states = self._local_token_shard(decoder_concept_states, seq_ctx, hidden_states.shape[1])

        decoder_history: tuple[torch.Tensor, ...] = (hidden_states,)
        if self.config.stage_group_size == 1:
            for layer_idx, stage in enumerate(self.decoder_stages):
                with self._activation_offload_context(hidden_states, layer_idx, "decoder"):
                    gate = self.final_read_concept_gate_logits[layer_idx].float().softmax(dim=-1)
                    hidden_states, raw = stage(
                        hidden_states,
                        token_position_embeddings,
                        seq_ctx,
                        repeated_concepts,
                        decoder_encoder_states,
                        decoder_concept_states,
                        gate,
                        *decoder_history,
                    )
                decoder_history = (*decoder_history, raw)
        else:
            for group_idx, group in enumerate(self.decoder_stage_groups):
                layer_idx = group_idx * self.config.stage_group_size
                with self._activation_offload_context(hidden_states, layer_idx, "decoder"):
                    group_outputs = group(
                        hidden_states,
                        token_position_embeddings,
                        seq_ctx,
                        repeated_concepts,
                        decoder_encoder_states,
                        decoder_concept_states,
                        self.final_read_concept_gate_logits,
                        *decoder_history,
                    )
                hidden_states, *raw_states = group_outputs
                decoder_history = (*decoder_history, *raw_states)
        hidden_states = self.final_norm(hidden_states)

        output: dict = {}
        if self.config.return_hidden_states:
            output["hidden_states"] = [hidden_states]
        if loss_ctx is None:
            _, (logits, _) = self.lm_head(hidden_states, None)
            output["logits"] = logits
        else:
            loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx["lm"])  # type: ignore[call-overload]
            total_loss = loss
            if ncp_loss is not None:
                total_loss = total_loss + self.config.concept_hlm_loss_weight * ncp_loss
            if vq_loss is not None:
                total_loss = total_loss + self.config.concept_vq_loss_weight * vq_loss
            output["loss"] = total_loss
            output["ncp_loss"] = ncp_loss
            output["vq_loss"] = vq_loss
            output["logits"] = logits
            if ncp_loss is not None:
                extra_info["ncp_loss"] = ncp_loss.detach()
            if vq_loss is not None:
                extra_info["vq_loss"] = vq_loss.detach()
            output["extra_info"] = extra_info
        return ModelOutputs(**output)

    @property
    @override
    def default_compile_cfg(self) -> dict[str, TorchCompileOption]:
        if self.config.stage_group_size > 1:
            return CONCEPT_OLMO_GROUPED_COMPILE_CFG
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

        if self.config.stage_group_size > 1:
            if fsdp_config.recompute_ratio > 0:
                raise ValueError("stage_group_size > 1 currently requires recompute_ratio=0")
            if fsdp_config.reshard_after_forward:
                raise ValueError("stage_group_size > 1 currently requires reshard_after_forward=False")
            if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
                raise ValueError("stage_group_size > 1 currently does not support activation offload")
            stage_groups = (
                *self.encoder_stage_groups,
                *self.concept_stage_groups,
                *self.decoder_stage_groups,
            )
            for group in stage_groups:
                self._fully_shard(
                    mesh=shard_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                    offload_policy=offload_policy,
                    module=group,
                )
        else:
            stage_stacks = (self.encoder_stages, self.concept_stages, self.decoder_stages)
            stage_locations = [(stack, stage_idx) for stack in stage_stacks for stage_idx in range(len(stack))]
            num_recompute_stages = int(len(stage_locations) * fsdp_config.recompute_ratio)
            generator = torch.Generator()
            generator.manual_seed(dist.get_rank())
            shuffled_indices = torch.randperm(len(stage_locations), generator=generator)
            for global_idx_tensor in tqdm(shuffled_indices, desc="[Concept OLMo FSDP Sharding]"):
                global_idx = int(global_idx_tensor)
                stack, stage_idx = stage_locations[global_idx]
                stage = stack[stage_idx]
                if global_idx < num_recompute_stages:
                    stage = apply_activation_checkpointing(
                        stage,
                        preserve_rng_state=checkpoint_preserve_rng_state,
                    )
                stack[stage_idx] = stage
                self._fully_shard(
                    mesh=shard_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=fsdp_config.reshard_after_forward,
                    offload_policy=offload_policy,
                    module=stage,
                )

        # Keep large shared modules independently shardable/prefetchable. Tiny
        # norms are intentionally left to the root FSDP group so their
        # parameters share one collective instead of eight latency-bound ones.
        separately_sharded_modules: tuple[nn.Module, ...] = (
            self.embed_tokens,
            self.prediction_heads,
        )
        for module in separately_sharded_modules:
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
        if self.config.stage_group_size > 1:
            ordered_forward_modules = [
                *self.encoder_stage_groups,
                *self.concept_stage_groups,
                self.prediction_heads,
                *self.decoder_stage_groups,
                self.lm_head,
            ]
        else:
            ordered_forward_modules = [
                *self.encoder_stages,
                *self.concept_stages,
                self.prediction_heads,
                *self.decoder_stages,
                self.lm_head,
            ]
        for current_module, next_module in zip(ordered_forward_modules[:-1], ordered_forward_modules[1:], strict=True):
            current_module.set_modules_to_forward_prefetch([next_module])  # type: ignore[attr-defined]
        first_encoder_module = (
            self.encoder_stage_groups[0] if self.config.stage_group_size > 1 else self.encoder_stages[0]
        )
        cast(Any, self).set_modules_to_forward_prefetch([self.embed_tokens, first_encoder_module])
        self._init_load_spec()
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
    concept_hlm_loss_weight: float = 0.0
    concept_vq_loss_weight: float = 0.0
    concept_vq_commitment_cost: float = 0.25
    concept_vq_detach_hlm_target: bool = True
    concept_hlm_loss_type: Literal["mse"] = "mse"
    concept_fusion_norm_alpha_init: float
    concept_dd_two_route_add_decoder_use_softmax: bool
    window_size: tuple[int, int]
    window_attn_skip_freq: int
    lm_head_vocab_padding_multiple: int = 1
    stage_group_size: int = 1

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
        if self.lm_head_vocab_padding_multiple <= 0:
            raise ValueError("lm_head_vocab_padding_multiple must be positive")
        if self.stage_group_size <= 0:
            raise ValueError("stage_group_size must be positive")
        if self.concept_hlm_loss_weight < 0 or self.concept_vq_loss_weight < 0:
            raise ValueError("concept auxiliary loss weights must be non-negative")
        if self.concept_vq_commitment_cost < 0:
            raise ValueError("concept_vq_commitment_cost must be non-negative")
        return self

    def build(self) -> ConceptLMV22VQ:
        return ConceptLMV22VQ(self)

    @classmethod
    def from_hf(cls, hf_path: str | Path) -> Self:
        from transformers import AutoConfig, AutoTokenizer

        hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        if hf_config.model_type not in {"conceptlm_v22_vq", "ncp_olmo3"}:
            raise ValueError(f"Expected conceptlm_v22_vq or ncp_olmo3, got {hf_config.model_type}")
        def get_field(name: str, legacy: str, default: Any = None) -> Any:
            return getattr(hf_config, name, getattr(hf_config, legacy, default))
        unsupported = {
            "num_query_groups": get_field("num_query_groups", "num_attention_heads") == hf_config.num_attention_heads,
            "rotary_percent": getattr(hf_config, "rotary_percent", 1.0) == 1.0,
            "rotary_interleaved": not getattr(hf_config, "rotary_interleaved", False),
            "position_embedding_type": getattr(hf_config, "position_embedding_type", "rope") in {"rope", "yarn"},
            "gated_linear_unit": getattr(hf_config, "gated_linear_unit", True),
            "activation_function": getattr(hf_config, "activation_function", "silu") == "silu",
            "concept_v22_vq_merge_mode": get_field("conceptlm_v22_vq_merge_mode", "concept_v22_vq_merge_mode") == "raw_logits",
            "concept_layer_norm_option": get_field("conceptlm_layer_norm_option", "concept_layer_norm_option") == "normed_add",
            "concept_dd_two_route_add": get_field("conceptlm_v21_dd_two_route_add", "concept_dd_two_route_add", True),
            "concept_enable_concept_read_encoder": get_field("conceptlm_v21_enable_concept_read_encoder", "concept_enable_concept_read_encoder", True),
            "concept_enable_decoder_read_encoder": get_field("conceptlm_v21_enable_decoder_read_encoder", "concept_enable_decoder_read_encoder", True),
            "concept_enable_decoder_read_concept": get_field("conceptlm_v21_enable_decoder_read_concept", "concept_enable_decoder_read_concept", True),
            "concept_final_read_concept_gate": get_field("conceptlm_v21_final_read_concept_gate", "concept_final_read_concept_gate", True),
        }
        disabled = [name for name, enabled in unsupported.items() if not enabled]
        if disabled:
            raise ValueError(f"Unsupported ConceptLM V2.2 options: {disabled}")

        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        return cls(
            model_type=hf_config.model_type,
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            bos_token_id=getattr(hf_config, "bos_token_id", None) or tokenizer.bos_token_id,
            eos_token_id=getattr(hf_config, "eos_token_id", None) or tokenizer.eos_token_id,
            pad_token_id=getattr(hf_config, "pad_token_id", None) or tokenizer.pad_token_id,
            num_hidden_layers=hf_config.num_layers,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.ffn_hidden_size,
            rms_norm_eps=hf_config.layernorm_epsilon,
            hidden_act=getattr(hf_config, "activation_function", "silu"),
            attention=MHAConfig(
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=get_field("num_query_groups", "num_attention_heads"),
                head_dim=hf_config.kv_channels,
                dropout=hf_config.attention_dropout,
                qk_norm=True,
                rms_norm_eps=hf_config.layernorm_epsilon,
                sliding_window=int(hf_config.window_size[0]),
            ),
            rope_parameters_cfg=RopeParametersConfig.from_hf_config(hf_config),
            tie_word_embeddings=hf_config.tie_word_embeddings,
            use_sliding_window=True,
            concept_encoder_layers=get_field("conceptlm_encoder_layers", "concept_encoder_layers"),
            concept_decoder_layers=get_field("conceptlm_decoder_layers", "concept_decoder_layers"),
            concept_special_layers=get_field("conceptlm_special_layers", "concept_special_layers"),
            concept_chunk_size=get_field("conceptlm_chunk_size", "concept_chunk_size"),
            concept_shift_feature=get_field("conceptlm_shift_feature", "concept_shift_feature"),
            concept_chunk_merge_method=get_field("conceptlm_chunk_merge_method", "concept_chunk_merge_method"),
            concept_v22_vq_codebook_size=get_field("conceptlm_v22_vq_codebook_size", "concept_v22_vq_codebook_size"),
            concept_v22_vq_num_codebooks=get_field("conceptlm_v22_vq_num_codebooks", "concept_v22_vq_num_codebooks"),
            concept_hlm_loss_weight=get_field("conceptlm_hlm_loss_weight", "concept_hlm_loss_weight", 0.0),
            concept_vq_loss_weight=get_field("conceptlm_vq_loss_weight", "concept_vq_loss_weight", 0.0),
            concept_vq_commitment_cost=get_field(
                "conceptlm_v22_vq_commitment_cost", "concept_v22_vq_commitment_cost", 0.25
            ),
            concept_vq_detach_hlm_target=get_field(
                "conceptlm_v22_vq_detach_hlm_target", "concept_v22_vq_detach_hlm_target", True
            ),
            concept_hlm_loss_type=get_field("conceptlm_v22_vq_hlm_loss_type", "concept_hlm_loss_type", "mse"),
            concept_fusion_norm_alpha_init=get_field("conceptlm_fusion_norm_alpha_init", "concept_fusion_norm_alpha_init"),
            concept_dd_two_route_add_decoder_use_softmax=get_field(
                "conceptlm_v21_dd_two_route_add_decoder_use_softmax",
                "concept_dd_two_route_add_decoder_use_softmax",
            ),
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
