# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Annotated, cast

import torch
import torch.nn.functional as F
from cyclopts import Parameter
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.ops.comm.all_to_all import ulysses_all_to_all
from xtuner.v1.utils import get_logger, init_params

from ...ops.gated_deltanet import get_causal_conv1d_fn, get_chunk_gated_delta_rule_fn
from ...ops.gated_deltanet.gen_seq_idx import gen_seq_idx
from ..linear import build_linear
from .attn_outputs import AttnOutputs


# Temporary solution: use separate function objects for each call site, Dynamo will cache them separately
def _all_to_all_conv_pre_qk(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_conv_pre_v(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_gb(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


def _all_to_all_out(x, scatter_dim, gather_dim, mesh):
    return ulysses_all_to_all(x, scatter_dim=scatter_dim, gather_dim=gather_dim, mesh=mesh)


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

logger = get_logger()


class GatedDeltaNetConfig(BaseModel):
    model_config = ConfigDict(title="Base attention config for xtuner", extra="forbid")
    num_value_heads: Annotated[int, Parameter(group="attention")]
    num_key_heads: Annotated[int, Parameter(group="attention")]
    key_head_dim: Annotated[int, Parameter(group="attention")]
    value_head_dim: Annotated[int, Parameter(group="attention")]
    conv_kernel_dim: Annotated[int, Parameter(group="attention")]
    hidden_act: Annotated[str, Parameter(group="model")]  # key defined in `transformers.activations.ACT2CLS`
    rms_norm_eps: Annotated[float, Parameter(group="attention")]

    def build(
        self,
        hidden_size: int,
        float8_cfg: Float8Config | None = None,
        **kwargs,
    ) -> "GatedDeltaNet":
        return GatedDeltaNet(
            **self.model_dump(),
            hidden_size=hidden_size,
            float8_cfg=float8_cfg,
        )


class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_value_heads: int,
        num_key_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        hidden_act: str,
        rms_norm_eps: float,
        layer_idx: int = 0,
        float8_cfg: Float8Config | None = None,
    ) -> None:
        super().__init__()
        self.name = f"layers.{layer_idx}.gate_deltanet"
        self.float8_cfg = float8_cfg

        self.hidden_size = hidden_size
        self.num_v_heads = num_value_heads
        self.num_k_heads = num_key_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = hidden_act
        self.rms_norm_eps = rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Resolved at build time so `XTUNER_HF_IMPL` selects the HF-native fla call patterns
        # (same convention as `get_attn_impl_fn`).
        if not has_fused_rms_norm_gated:
            assert _fused_rms_norm_gated_import_error is not None
            raise _fused_rms_norm_gated_import_error
        assert FusedRMSNormGated is not None
        self.causal_conv1d_fn = get_causal_conv1d_fn()
        self.chunk_gated_delta_rule = get_chunk_gated_delta_rule_fn()
        self.norm = FusedRMSNormGated(self.head_v_dim, eps=self.rms_norm_eps, activation=self.activation)

        self.out_proj = build_linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            float8_cfg=self.float8_cfg,
        )

        self.in_proj_qkv = build_linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
        self.in_proj_z = build_linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            float8_cfg=self.float8_cfg,
        )
        self.in_proj_b = build_linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = build_linear(self.hidden_size, self.num_v_heads, bias=False)

    @torch.no_grad()
    def init_extra_weights(self) -> set[str]:
        """Initialize Qwen3.5 GatedDeltaNet parameters that are not standard
        weights or biases."""

        def init_a_log(param: torch.Tensor) -> None:
            nn.init.uniform_(param, a=0.0, b=16.0)
            param.log_()

        def init_dt_bias(param: torch.Tensor) -> None:
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            nn.init.uniform_(param)
            dt = torch.exp(param * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt.clamp_(min=dt_init_floor)
            param.copy_(dt + torch.log(-torch.expm1(-dt)))

        init_params(self.A_log, init_a_log)
        init_params(self.dt_bias, init_dt_bias)
        return {"A_log", "dt_bias"}

    def get_muon_split_sizes(self) -> dict[nn.Parameter, tuple[int, ...]]:
        """Return the logical Q, K, and V row blocks for Muon
        orthogonalization.

        Returns:
            dict[nn.Parameter, tuple[int, ...]]: Q, K, and V row split sizes keyed by the fused parameter.
        """
        return {
            cast(nn.Parameter, self.in_proj_qkv.weight): (
                self.key_dim,
                self.key_dim,
                self.value_dim,
            )
        }

    def _mudd_qkv_projection(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor,
        value_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``in_proj_qkv`` is laid out as [Q | K | V] and each routed source feeds
        # exactly one block. Projecting every source with the whole fused weight
        # (the previous behavior) computed all three blocks three times over and
        # discarded two thirds of each result; slicing the weight rows first keeps
        # the total projection work equal to the non-MUDD path. The per-source
        # gradients still accumulate into the single fused parameter.
        weight = self.in_proj_qkv.weight
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        # Row slicing is only equivalent for the plain bf16 linear (the float8
        # variants apply their scaling inside the module) over an unsharded fused
        # weight; otherwise keep the previous whole-weight projections.
        if weight.shape[0] == self.key_dim * 2 + self.value_dim and (
            self.float8_cfg is None or self.float8_cfg.scaling_granularity_gemm is None
        ):
            return (
                F.linear(query_input, weight[: self.key_dim]),
                F.linear(key_input, weight[self.key_dim : 2 * self.key_dim]),
                F.linear(value_input, weight[2 * self.key_dim :]),
            )
        return (
            self.in_proj_qkv(query_input)[..., : self.key_dim],
            self.in_proj_qkv(key_input)[..., self.key_dim : 2 * self.key_dim],
            self.in_proj_qkv(value_input)[..., 2 * self.key_dim :],
        )

    def forward_for_sp(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # not used
        mudd_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> AttnOutputs:
        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "Only batch size of 1 is supported for now in GateDeltaNet"
        query_input, key_input, value_input = mudd_inputs if mudd_inputs is not None else (hidden_states,) * 3
        if mudd_inputs is None:
            mixed_qkv = self.in_proj_qkv(query_input)
        else:
            query, key, value = self._mudd_qkv_projection(query_input, key_input, value_input)

        z = self.in_proj_z(query_input)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(query_input)
        a = self.in_proj_a(query_input)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if bias and isinstance(bias, DTensor):
            bias = bias.to_local()

        assert seq_ctx.sequence_parallel_mesh is not None, "sequence_parallel_mesh is required for forward_for_sp"
        sp_rank = seq_ctx.sp_rank
        sp_size = seq_ctx.sequence_parallel_mesh.size()
        if seq_ctx.seq_idx is None:
            # SP restores the full packed sequence before convolution, so seq_idx uses the global length.
            seq_idx = gen_seq_idx(seq_len * sp_size, seq_ctx.cu_seq_lens_q)
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        if mudd_inputs is None:
            query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.transpose(1, 2)  # (1, dim, L/sp_size)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = _all_to_all_conv_pre_qk(
            query,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )
        key = _all_to_all_conv_pre_qk(
            key,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )
        value = _all_to_all_conv_pre_v(
            value,
            scatter_dim=1,
            gather_dim=2,
            mesh=seq_ctx.sequence_parallel_mesh,
        )

        query_weight, key_weight, value_weight = torch.split(
            weight,  # (8192, 4)
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=0,
        )
        query_weight = query_weight.chunk(sp_size, dim=0)[sp_rank]
        key_weight = key_weight.chunk(sp_size, dim=0)[sp_rank]
        value_weight = value_weight.chunk(sp_size, dim=0)[sp_rank]
        if bias is not None:
            bias = bias.chunk(sp_size, dim=0)[sp_rank]

        # The local causal-conv custom op consumes channel-last tensors when seq_idx is set.
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        query = self.causal_conv1d_fn(
            x=query,
            weight=query_weight,
            bias=bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        key = self.causal_conv1d_fn(
            x=key,
            weight=key_weight,
            bias=bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )
        value = self.causal_conv1d_fn(
            x=value,
            weight=value_weight,
            bias=bias,
            activation=self.activation,
            seq_idx=seq_idx,
        )

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A_log = self.A_log
        dt_bias = self.dt_bias
        if isinstance(A_log, DTensor):
            A_log = A_log.to_local()
        if isinstance(dt_bias, DTensor):
            dt_bias = dt_bias.to_local()

        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        query = query.reshape(batch_size, seq_len * sp_size, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len * sp_size, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len * sp_size, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            g = g.transpose(1, 2)
            beta = beta.transpose(1, 2)

            g = _all_to_all_gb(
                g,  # (1, num_v_heads, L/sp_size)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            beta = _all_to_all_gb(
                beta,  # (1, num_v_heads, L/sp_size)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )
            g = g.transpose(1, 2)
            beta = beta.transpose(1, 2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
        )

        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            core_attn_out = _all_to_all_out(
                core_attn_out,  # (1, L, num_v_head/sp_size, head_dim)
                scatter_dim=1,
                gather_dim=2,
                mesh=seq_ctx.sequence_parallel_mesh,
            )

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        attn_outputs: AttnOutputs = {
            "projected_output": output,
        }
        return attn_outputs

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # not used
        mudd_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> AttnOutputs:
        if seq_ctx.sequence_parallel_mesh and seq_ctx.sequence_parallel_mesh.size() > 1:
            return self.forward_for_sp(hidden_states, seq_ctx, position_embeddings, mudd_inputs)

        batch_size, seq_len, _ = hidden_states.shape
        assert batch_size == 1, "Only batch size of 1 is supported for now in GateDeltaNet"
        query_input, key_input, value_input = mudd_inputs if mudd_inputs is not None else (hidden_states,) * 3
        if mudd_inputs is None:
            mixed_qkv = self.in_proj_qkv(query_input)
        else:
            query, key, value = self._mudd_qkv_projection(query_input, key_input, value_input)

        z = self.in_proj_z(query_input)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(query_input)
        a = self.in_proj_a(query_input)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias
        if isinstance(weight, DTensor):
            weight = weight.to_local()
        if bias and isinstance(bias, DTensor):
            bias = bias.to_local()

        if seq_ctx.seq_idx is None:
            # Keep seq_idx generation in a custom op so full-graph compile avoids Python tensor construction.
            seq_idx = gen_seq_idx(seq_len, seq_ctx.cu_seq_lens_q)
            seq_ctx.seq_idx = cast(torch.IntTensor, seq_idx)
        else:
            seq_idx = seq_ctx.seq_idx

        if mudd_inputs is None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=weight,
                bias=bias,
                activation=self.activation,
                seq_idx=seq_idx,
            )
            query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        else:
            query = self.causal_conv1d_fn(
                x=query,
                weight=weight[: self.key_dim],
                bias=bias[: self.key_dim] if bias is not None else None,
                activation=self.activation,
                seq_idx=seq_idx,
            )
            key = self.causal_conv1d_fn(
                x=key,
                weight=weight[self.key_dim : 2 * self.key_dim],
                bias=bias[self.key_dim : 2 * self.key_dim] if bias is not None else None,
                activation=self.activation,
                seq_idx=seq_idx,
            )
            value = self.causal_conv1d_fn(
                x=value,
                weight=weight[2 * self.key_dim :],
                bias=bias[2 * self.key_dim :] if bias is not None else None,
                activation=self.activation,
                seq_idx=seq_idx,
            )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A_log = self.A_log
        dt_bias = self.dt_bias
        if isinstance(A_log, DTensor):
            A_log = A_log.to_local()
        if isinstance(dt_bias, DTensor):
            dt_bias = dt_bias.to_local()

        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=seq_ctx.cu_seq_lens_q,
        )
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        attn_outputs: AttnOutputs = {
            "projected_output": output,
        }
        return attn_outputs

    @overload  # type: ignore
    def __call__(  # type: ignore
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        mudd_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> AttnOutputs: ...

    __call__ = nn.Module.__call__
