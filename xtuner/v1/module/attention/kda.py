# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Delta Attention (KDA) — distinct from Qwen3.5 GatedDeltaNet weight layout."""

from __future__ import annotations

import math
from typing import Annotated, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch.distributed.tensor import DTensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops.gated_deltanet import get_causal_conv1d_fn
from xtuner.v1.ops.gated_deltanet.gen_seq_idx import gen_seq_idx
from xtuner.v1.utils import get_logger, init_params

from ..linear import build_linear
from .attn_outputs import AttnOutputs


logger = get_logger()

_fused_rms_norm_gated_import_error: ImportError | None = None
try:
    from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated

    class FusedRMSNormGated(FLA_FusedRMSNormGated):
        def forward(
            self,
            x: torch.Tensor,
            g: torch.Tensor,
            residual: torch.Tensor | None = None,
            prenorm: bool = False,
            residual_in_fp32: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            from ...ops.gated_deltanet.rms_norm_gated import rms_norm_gated

            weight = self.weight
            if isinstance(weight, DTensor):
                weight = weight.to_local()
            return rms_norm_gated(
                x,
                g,
                weight,
                self.bias,
                self.activation,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )

except (ImportError, ModuleNotFoundError) as e:
    has_fused_rms_norm_gated = False
    FusedRMSNormGated = None  # type: ignore
    _fused_rms_norm_gated_import_error = e
else:
    has_fused_rms_norm_gated = True
    _fused_rms_norm_gated_import_error = None


def _get_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda

        return chunk_kda
    except (ImportError, ModuleNotFoundError):
        return None


class KDAConfig(BaseModel):
    model_config = ConfigDict(title="Kimi Delta Attention config", extra="forbid")
    num_heads: Annotated[int, Parameter(group="attention")]
    head_dim: Annotated[int, Parameter(group="attention")]
    conv_kernel_size: Annotated[int, Parameter(group="attention")] = 4
    use_full_rank_gate: Annotated[bool, Parameter(group="attention")] = True
    gate_lower_bound: Annotated[float | None, Parameter(group="attention")] = -5.0
    rms_norm_eps: Annotated[float, Parameter(group="attention")] = 1e-5

    def build(
        self,
        hidden_size: int,
        float8_cfg: Float8Config | None = None,
        layer_idx: int = 0,
        **kwargs: Any,
    ) -> "KimiDeltaAttention":
        return KimiDeltaAttention(
            **self.model_dump(),
            hidden_size=hidden_size,
            float8_cfg=float8_cfg,
            layer_idx=layer_idx,
        )


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention matching HF `KimiDeltaAttention` parameter names."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        conv_kernel_size: int = 4,
        use_full_rank_gate: bool = True,
        gate_lower_bound: float | None = -5.0,
        rms_norm_eps: float = 1e-5,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
    ) -> None:
        super().__init__()
        self.name = f"layers.{layer_idx}.self_attn"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.conv_size = conv_kernel_size
        self.use_full_rank_gate = use_full_rank_gate
        self.gate_lower_bound = gate_lower_bound
        self.layer_idx = layer_idx
        self.float8_cfg = float8_cfg

        projection_size = head_dim * num_heads
        self.q_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)
        self.k_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)
        self.v_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)

        self.q_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            kernel_size=conv_kernel_size,
            groups=projection_size,
            padding=conv_kernel_size - 1,
            bias=False,
        )
        self.k_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            kernel_size=conv_kernel_size,
            groups=projection_size,
            padding=conv_kernel_size - 1,
            bias=False,
        )
        self.v_conv1d = nn.Conv1d(
            projection_size,
            projection_size,
            kernel_size=conv_kernel_size,
            groups=projection_size,
            padding=conv_kernel_size - 1,
            bias=False,
        )

        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.f_a_proj = build_linear(hidden_size, head_dim, bias=False, float8_cfg=float8_cfg)
        self.f_b_proj = build_linear(head_dim, projection_size, bias=False, float8_cfg=float8_cfg)
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))
        self.b_proj = build_linear(hidden_size, num_heads, bias=False, float8_cfg=float8_cfg)

        if use_full_rank_gate:
            self.g_proj = build_linear(hidden_size, projection_size, bias=False, float8_cfg=float8_cfg)
        else:
            self.g_a_proj = build_linear(hidden_size, head_dim, bias=False, float8_cfg=float8_cfg)
            self.g_b_proj = build_linear(head_dim, projection_size, bias=False, float8_cfg=float8_cfg)

        if not has_fused_rms_norm_gated:
            assert _fused_rms_norm_gated_import_error is not None
            raise _fused_rms_norm_gated_import_error
        assert FusedRMSNormGated is not None
        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = build_linear(projection_size, hidden_size, bias=False, float8_cfg=float8_cfg)
        self.causal_conv1d_fn = get_causal_conv1d_fn()
        self.chunk_kda = _get_chunk_kda()

    @torch.no_grad()
    def init_extra_weights(self) -> set[str]:
        def init_a_log(param: torch.Tensor) -> None:
            nn.init.uniform_(param, a=1.0, b=16.0)
            param.log_()

        def init_dt_bias(param: torch.Tensor) -> None:
            nn.init.zeros_(param)

        init_params(self.A_log, init_a_log)
        init_params(self.dt_bias, init_dt_bias)
        return {"A_log", "dt_bias"}

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> conv expects (B, C, L)
        weight = conv.weight.squeeze(1)
        bias = conv.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if bias is not None and isinstance(bias, DTensor):
            bias = bias.to_local()
        x_t = x.transpose(1, 2).contiguous()
        y = self.causal_conv1d_fn(x_t, weight, bias, activation="silu", seq_idx=seq_idx)
        return y.transpose(1, 2).contiguous()

    def _fallback_kda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Shape-correct fallback when ``fla.ops.kda`` is unavailable (smoke only)."""
        # q/k/v/g: (B, T, H, D); beta: (B, T, H)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bthd,bshd->bhts", q, k) * scale
        attn = attn.softmax(dim=-1)
        o = torch.einsum("bhts,bshd->bthd", attn, v)
        gate = torch.sigmoid(g)
        beta_gate = torch.sigmoid(beta).unsqueeze(-1)
        return o * gate * beta_gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        mudd_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> AttnOutputs:
        del position_embeddings, mudd_inputs
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "KDA currently supports packed batch size 1"

        if seq_ctx.seq_idx is None:
            seq_idx = gen_seq_idx(seq_len, seq_ctx.cu_seq_lens_q)
            seq_ctx.seq_idx = seq_idx  # type: ignore[assignment]
        else:
            seq_idx = seq_ctx.seq_idx

        q = self._causal_conv(self.q_conv1d, self.q_proj(hidden_states), seq_idx)
        k = self._causal_conv(self.k_conv1d, self.k_proj(hidden_states), seq_idx)
        v = self._causal_conv(self.v_conv1d, self.v_proj(hidden_states), seq_idx)

        g_decay = self.f_b_proj(self.f_a_proj(hidden_states))
        g_decay = g_decay.view(batch_size, seq_len, self.num_heads, self.head_dim)
        beta = self.b_proj(hidden_states).float()

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        if self.chunk_kda is not None:
            o, _ = self.chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g_decay,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=self.gate_lower_bound is not None,
                lower_bound=self.gate_lower_bound,
                transpose_state_layout=True,
                cu_seqlens=seq_ctx.cu_seq_lens_q,
            )
        else:
            logger.warning_once("fla.ops.kda.chunk_kda unavailable; using smoke fallback for KDA")
            o = self._fallback_kda(q, k, v, g_decay, beta)

        if self.use_full_rank_gate:
            g_out = self.g_proj(hidden_states)
        else:
            g_out = self.g_b_proj(self.g_a_proj(hidden_states))
        g_out = g_out.view(batch_size, seq_len, self.num_heads, self.head_dim)
        o = self.o_norm(o, g_out)
        raw_output = o.reshape(batch_size, seq_len, -1)
        projected_output = self.o_proj(raw_output)
        return {
            "raw_output": raw_output,
            "projected_output": projected_output,
            "softmax_lse": None,
        }
