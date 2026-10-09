# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Delta Attention (KDA) — distinct from Qwen3.5 GatedDeltaNet weight
layout."""

from __future__ import annotations

from typing import Annotated, Any

import torch
import torch.nn as nn
from cyclopts import Parameter
from einops import rearrange
from pydantic import BaseModel, ConfigDict

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.ops.gated_deltanet.dtensor import dtensor_full_tensor, materialize_depthwise_conv1d_weight_bias
from xtuner.v1.ops.gated_deltanet.fused_rms_norm_gated import (
    FusedRMSNormGated,
    _fused_rms_norm_gated_import_error,
    has_fused_rms_norm_gated,
)
from xtuner.v1.utils import init_params

from ..linear import build_linear
from .attn_outputs import AttnOutputs


# Separate call-site wrappers so Dynamo caches each SP collective independently (same as GDN).
def _all_to_all_conv_pre_q(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_conv_pre_k(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_conv_pre_v(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_g(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_beta(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_out(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


_fla_kda_import_error: BaseException | None = None
_fla_causal_conv1d = None  # type: ignore[assignment]


def _unavailable_chunk_kda(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
    del args, kwargs
    raise RuntimeError("Please run `pip install -U fla-core`")


chunk_kda = _unavailable_chunk_kda


class KDAShortConvolution(nn.Module):
    """Typed stand-in; the ``try`` block replaces it with the FLA subclass."""

    weight: nn.Parameter
    bias: nn.Parameter | None
    activation: str

    def __init__(self, hidden_size: int, kernel_size: int, activation: str = "silu", **kwargs: Any) -> None:
        super().__init__()
        del hidden_size, kernel_size, kwargs
        self.activation = activation

    def materialize_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise RuntimeError("Please run `pip install -U fla-core`")

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del x, cache, output_final_state, cu_seqlens, weight, bias, kwargs
        raise RuntimeError("Please run `pip install -U fla-core`")


try:
    from fla.modules import ShortConvolution as FLAShortConvolution
    from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d_impl
    from fla.ops.kda import chunk_kda as _chunk_kda

    if not has_fused_rms_norm_gated or FusedRMSNormGated is None:
        raise ImportError("FusedRMSNormGated unavailable") from _fused_rms_norm_gated_import_error

    _fla_causal_conv1d = _fla_causal_conv1d_impl

    class KDAShortConvolution(FLAShortConvolution):  # type: ignore[no-redef]
        """HF ``ShortConvolution``; always routes through FLA
        ``causal_conv1d``."""

        def materialize_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
            return materialize_depthwise_conv1d_weight_bias(self.weight, self.bias)

        def forward(  # type: ignore[override]
            self,
            x: torch.Tensor,
            cache: torch.Tensor | None = None,
            output_final_state: bool = False,
            cu_seqlens: torch.Tensor | None = None,
            weight: torch.Tensor | None = None,
            bias: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            assert _fla_causal_conv1d is not None
            if weight is None:
                weight, bias = self.materialize_weight_bias()
            elif weight.ndim == 3:
                weight = rearrange(weight, "d 1 w -> d w")
            return _fla_causal_conv1d(
                x=x,
                weight=weight,
                bias=bias,
                initial_state=cache,
                output_final_state=output_final_state,
                activation=self.activation,
                backend=getattr(self, "backend", "triton"),
                cu_seqlens=cu_seqlens,
                **kwargs,
            )

    chunk_kda = _chunk_kda
except (ImportError, ModuleNotFoundError) as e:
    has_fla_kda = False
    _fla_kda_import_error = e
else:
    has_fla_kda = True
    _fla_kda_import_error = None


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
    ) -> KimiDeltaAttention:
        return KimiDeltaAttention(
            **self.model_dump(),
            hidden_size=hidden_size,
            float8_cfg=float8_cfg,
            layer_idx=layer_idx,
        )


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention matching HF `KimiDeltaAttention` parameter
    names."""

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
        if not has_fla_kda:
            assert _fla_kda_import_error is not None
            raise ImportError("Please run `pip install -U fla-core`") from _fla_kda_import_error
        assert FusedRMSNormGated is not None
        assert KDAShortConvolution is not None
        assert chunk_kda is not None

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

        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
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
            nn.init.zeros_(param)

        def init_dt_bias(param: torch.Tensor) -> None:
            # inverse_softplus(LogUniform[0.001, 0.1])
            log_low = torch.log(torch.tensor(0.001, device=param.device, dtype=param.dtype))
            log_high = torch.log(torch.tensor(0.1, device=param.device, dtype=param.dtype))
            dt = torch.empty_like(param).uniform_(log_low.item(), log_high.item()).exp()
            param.copy_(dt + torch.log(-torch.expm1(-dt)))

        init_params(self.A_log, init_a_log)
        init_params(self.dt_bias, init_dt_bias)
        return {"A_log", "dt_bias"}

    def _maybe_full(self, param: torch.Tensor) -> torch.Tensor:
        return dtensor_full_tensor(param)

    def _sp_short_conv(
        self,
        conv: KDAShortConvolution,
        x_local: torch.Tensor,
        *,
        all_to_all_fn,
        sp_mesh,
        sp_rank: int,
        sp_size: int,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        """Gather full sequence / shard channels, run the same short-conv entry
        as non-SP."""
        # (B, L/sp, C) -> (B, C, L/sp) -> all_to_all -> (B, C/sp, L) -> (B, L, C/sp)
        x = x_local.transpose(1, 2)
        x = all_to_all_fn(x, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
        x = x.transpose(1, 2).contiguous()

        weight, bias = conv.materialize_weight_bias()
        weight = weight.chunk(sp_size, dim=0)[sp_rank]
        if bias is not None:
            bias = bias.chunk(sp_size, dim=0)[sp_rank]

        out, _ = conv(
            x,
            cache=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            weight=weight,
            bias=bias,
        )
        return out

    def forward_for_sp(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
    ) -> AttnOutputs:
        """Sequence-parallel KDA aligned with GatedDeltaNet
        ``forward_for_sp``."""
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "KDA currently supports packed batch size 1"
        assert seq_ctx.sequence_parallel_mesh is not None, "sequence_parallel_mesh is required for forward_for_sp"
        sp_mesh = seq_ctx.sequence_parallel_mesh
        sp_size = sp_mesh.size()
        sp_rank = seq_ctx.sp_rank
        assert self.num_heads % sp_size == 0, (
            f"KDA num_heads ({self.num_heads}) must be divisible by sp_size ({sp_size})"
        )

        cu_seqlens = seq_ctx.cu_seq_lens_q
        projection_size = self.num_heads * self.head_dim

        q = self._sp_short_conv(
            self.q_conv1d,
            self.q_proj(hidden_states),
            all_to_all_fn=_all_to_all_conv_pre_q,
            sp_mesh=sp_mesh,
            sp_rank=sp_rank,
            sp_size=sp_size,
            cu_seqlens=cu_seqlens,
        )
        k = self._sp_short_conv(
            self.k_conv1d,
            self.k_proj(hidden_states),
            all_to_all_fn=_all_to_all_conv_pre_k,
            sp_mesh=sp_mesh,
            sp_rank=sp_rank,
            sp_size=sp_size,
            cu_seqlens=cu_seqlens,
        )
        v = self._sp_short_conv(
            self.v_conv1d,
            self.v_proj(hidden_states),
            all_to_all_fn=_all_to_all_conv_pre_v,
            sp_mesh=sp_mesh,
            sp_rank=sp_rank,
            sp_size=sp_size,
            cu_seqlens=cu_seqlens,
        )

        # Local seq, full heads → all_to_all → full seq, head shard (same as GDN g/beta).
        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = g.view(batch_size, seq_len, projection_size).transpose(1, 2)  # (B, H*D, L/sp)
        g = _all_to_all_g(g, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
        g = g.transpose(1, 2).view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)

        beta = self.b_proj(hidden_states).float().transpose(1, 2)  # (B, H, L/sp)
        beta = _all_to_all_beta(beta, scatter_dim=1, gather_dim=2, mesh=sp_mesh)
        beta = beta.transpose(1, 2)  # (B, L, H/sp)

        q = q.view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)
        k = k.view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)
        v = v.view(batch_size, seq_len * sp_size, self.num_heads // sp_size, self.head_dim)

        a_log = self._maybe_full(self.A_log).chunk(sp_size, dim=0)[sp_rank]
        dt_bias = (
            self._maybe_full(self.dt_bias)
            .view(self.num_heads, self.head_dim)
            .chunk(sp_size, dim=0)[sp_rank]
            .reshape(-1)
        )

        o, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=a_log,
            dt_bias=dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )

        # (B, L, H/sp, D) -> (B, L/sp, H, D)
        o = _all_to_all_out(o, scatter_dim=1, gather_dim=2, mesh=sp_mesh)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        **kwargs: Any,
    ) -> AttnOutputs:
        # Decoder layers may still pass rotary / mudd kwargs used by MLA.
        del kwargs
        if seq_ctx.sequence_parallel_mesh is not None and seq_ctx.sequence_parallel_mesh.size() > 1:
            return self.forward_for_sp(hidden_states, seq_ctx)

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
            state_v_first=True,
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
