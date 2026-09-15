# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Delta Attention (KDA) — distinct from Qwen3.5 GatedDeltaNet weight layout."""

from __future__ import annotations

from typing import Annotated, Any

import torch
import torch.nn as nn
from cyclopts import Parameter
from einops import rearrange
from pydantic import BaseModel, ConfigDict
from torch.distributed.tensor import DTensor

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.utils import init_params

from ..linear import build_linear
from .attn_outputs import AttnOutputs


try:
    from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated
    from fla.modules import ShortConvolution as FLAShortConvolution
    from fla.ops.kda import chunk_kda
except ImportError as e:
    raise ImportError("Please run `pip install -U fla-core`") from e


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


class KDAShortConvolution(FLAShortConvolution):
    """HF ``ShortConvolution`` with FSDP DTensor weights gathered for the kernel."""

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(self.weight, DTensor) and not isinstance(self.bias, DTensor):
            return super().forward(
                x,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )

        from fla.modules.conv.causal_conv1d import causal_conv1d

        weight = self.weight.full_tensor() if isinstance(self.weight, DTensor) else self.weight
        bias = self.bias
        if bias is not None and isinstance(bias, DTensor):
            bias = bias.full_tensor()
        return causal_conv1d(
            x=x,
            weight=rearrange(weight, "d 1 w -> d w"),
            bias=bias,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=getattr(self, "backend", "triton"),
            cu_seqlens=cu_seqlens,
            **kwargs,
        )


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

        self.q_conv1d = KDAShortConvolution(
            hidden_size=projection_size,
            kernel_size=conv_kernel_size,
            activation="silu",
        )
        self.k_conv1d = KDAShortConvolution(
            hidden_size=projection_size,
            kernel_size=conv_kernel_size,
            activation="silu",
        )
        self.v_conv1d = KDAShortConvolution(
            hidden_size=projection_size,
            kernel_size=conv_kernel_size,
            activation="silu",
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

        self.o_norm = FusedRMSNormGated(head_dim, eps=rms_norm_eps, activation="sigmoid")
        self.o_proj = build_linear(projection_size, hidden_size, bias=False, float8_cfg=float8_cfg)

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

    def _maybe_full(self, param: torch.Tensor) -> torch.Tensor:
        return param.full_tensor() if isinstance(param, DTensor) else param

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        **kwargs: Any,
    ) -> AttnOutputs:
        # Decoder layers may still pass rotary / mudd kwargs used by MLA.
        del kwargs
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "KDA currently supports packed batch size 1"

        cu_seqlens = seq_ctx.cu_seq_lens_q
        q, _ = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        k, _ = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        v, _ = self.v_conv1d(
            x=self.v_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )

        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim)
        beta = self.b_proj(hidden_states).float()

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self._maybe_full(self.A_log),
            dt_bias=self._maybe_full(self.dt_bias),
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            transpose_state_layout=True,
            cu_seqlens=cu_seqlens,
        )

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
