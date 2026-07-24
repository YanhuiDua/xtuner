# Copyright (c) OpenMMLab. All rights reserved.

import os
from typing import Callable, Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops import AttnOpOutputs, get_apply_rotary_emb, get_attn_impl_fn
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import XTUNER_DETERMINISTIC

from ..linear import build_linear
from ..rms_norm import RMSNorm
from .attn_outputs import AttnOutputs


class ConceptOlmoRMSNorm(RMSNorm):
    """RMSNorm with an exact remote-code parity path."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps)
        self.reference_impl = os.getenv("XTUNER_HF_IMPL") == "true"

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.reference_impl:
            return super().forward(hidden_states)
        if isinstance(self.weight, DTensor):
            weight = self.weight.to_local()
        else:
            weight = self.weight
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (values * weight.float()).to(input_dtype)


class ConceptOlmoAttention(nn.Module):
    """OLMo attention with head-interleaved fused QKV storage.

    Unlike the common XTuner MHA module, the Q/K RMSNorms cover the complete hidden vector before it is split into
    heads. This is the layout used by the ConceptLM V2.2 checkpoint and therefore must not be rewritten as separate
    Q/K/V projections.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        attn_impl: Literal["flash_attention", "flex_attention", "eager_attention"] = "flash_attention",
        sliding_window: int = -1,
        layer_type: Literal["full_attention", "sliding_attention"] = "full_attention",
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        if hidden_size != num_attention_heads * head_dim:
            raise ValueError("Concept OLMo requires hidden_size == num_attention_heads * head_dim")

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.scaling = head_dim**-0.5
        self.window_size = (sliding_window, 0) if layer_type == "sliding_attention" else (-1, -1)

        self.linear_qkv = build_linear(
            hidden_size,
            3 * hidden_size,
            bias=False,
            float8_cfg=float8_cfg,
        )
        self.linear_proj = build_linear(
            hidden_size,
            hidden_size,
            bias=False,
            float8_cfg=float8_cfg,
        )
        self.q_layernorm = ConceptOlmoRMSNorm(hidden_size, eps=rms_norm_eps)
        self.k_layernorm = ConceptOlmoRMSNorm(hidden_size, eps=rms_norm_eps)
        self.apply_rotary_emb = get_apply_rotary_emb()
        self.reference_sdpa = os.getenv("XTUNER_HF_IMPL") == "true"
        self.attn_impl_func: Callable[..., AttnOpOutputs] = get_attn_impl_fn(attn_impl)  # type: ignore[assignment]

    def _reference_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_ctx: SequenceContext,
    ) -> AttnOpOutputs:
        """Run the checkpoint's native SDPA path for numerical parity."""
        boundaries = [int(value) for value in seq_ctx.cu_seq_lens_q.detach().cpu().tolist()]
        outputs: list[torch.Tensor] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            query = query_states[:, :, start:end]
            key = key_states[:, :, start:end]
            value = value_states[:, :, start:end]
            seq_len = end - start
            use_window = self.window_size != (-1, -1) and self.window_size[0] < seq_len - 1
            if use_window:
                positions = torch.arange(seq_len, device=query.device)
                blocked = (positions[None, :] > positions[:, None]) | (
                    positions[:, None] - positions[None, :] > self.window_size[0]
                )
                mask = torch.zeros((seq_len, seq_len), device=query.device, dtype=query.dtype).masked_fill(
                    blocked, float("-inf")
                )
                attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=self.dropout)
            else:
                attended = F.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=self.dropout)
            outputs.append(attended.transpose(1, 2))
        return {"raw_output": torch.cat(outputs, dim=1).contiguous()}

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.linear_qkv(hidden_states).view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            3,
            self.head_dim,
        )
        query_states = qkv[..., 0, :].reshape(batch_size, seq_len, self.hidden_size)
        key_states = qkv[..., 1, :].reshape(batch_size, seq_len, self.hidden_size)
        value_states = qkv[..., 2, :]

        query_states = self.q_layernorm(query_states).view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        )
        key_states = self.k_layernorm(key_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_emb(query_states, key_states, cos, sin)

        sp_mesh = seq_ctx.sequence_parallel_mesh
        if sp_mesh is not None and sp_mesh.size() > 1:
            query_states = ulysses_all_to_all(query_states, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
            key_states = ulysses_all_to_all(key_states, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
            value_states = ulysses_all_to_all(value_states, scatter_dim=1, gather_dim=2, mesh=sp_mesh)

        if query_states.size(0) != 1:
            raise ValueError("Concept OLMo attention expects XTuner's packed batch dimension to be 1")

        if self.reference_sdpa:
            attn_op_outputs = self._reference_attention(query_states, key_states, value_states, seq_ctx)
        else:
            attn_op_outputs = self.attn_impl_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=seq_ctx.cu_seq_lens_q,
                cu_seqlens_k=seq_ctx.cu_seq_lens_k,
                max_seqlen_q=seq_ctx.max_length_q,
                max_seqlen_k=seq_ctx.max_length_k,
                window_size=self.window_size,
                dropout_p=self.dropout,
                softmax_scale=self.scaling,
                causal=True,
                deterministic=XTUNER_DETERMINISTIC,
            )
        raw_output = attn_op_outputs["raw_output"]

        if sp_mesh is not None and sp_mesh.size() > 1:
            raw_output = ulysses_all_to_all(raw_output, scatter_dim=1, gather_dim=2, mesh=sp_mesh)

        raw_output = raw_output.reshape(batch_size, seq_len, self.hidden_size).contiguous()
        projected_output = self.linear_proj(raw_output)
        return {"projected_output": projected_output, **attn_op_outputs}
