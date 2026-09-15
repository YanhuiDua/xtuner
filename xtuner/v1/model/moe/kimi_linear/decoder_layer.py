# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Linear MoE decoder: LatentMoE + SiTU + AttnRes cross-layer residual."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing_extensions import override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module import AttnOutputs, RMSNorm, RouterResults
from xtuner.v1.module.decoder_layer.dense_decoder_layer import (
    DenseDecoderLayer,
    DenseDecoderLayerMicroBatchOutput,
    DenseDecoderLayerOutput,
    DenseMLP,
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEActFnConfig,
    MoEBlock,
    MoEDecoderLayer,
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
    MoEMLP,
)
from xtuner.v1.module.attention.kda import KimiDeltaAttention
from xtuner.v1.module.linear import build_linear
from xtuner.v1.utils import ForwardState


def _run_self_attn(
    attn: nn.Module,
    hidden_states: torch.Tensor,
    seq_ctx: SequenceContext,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> AttnOutputs:
    if isinstance(attn, KimiDeltaAttention):
        return attn(hidden_states=hidden_states, seq_ctx=seq_ctx)
    return attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        seq_ctx=seq_ctx,
    )


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: RMSNorm,
) -> torch.Tensor:
    """HF ``_apply_attn_res`` — mix current prefix with stored block residuals.

    Args:
        prefix_sum: ``(num_tokens, hidden_size)``
        block_residual: ``(num_tokens, num_blocks, hidden_size)``
        proj: linear ``hidden -> 1`` score projection
        norm: RMSNorm providing ``weight`` and ``variance_epsilon``
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


class KimiDenseDecoderLayerOutput(DenseDecoderLayerOutput):
    block_residual: torch.Tensor | None


class KimiDenseDecoderLayerMicroBatchOutput(DenseDecoderLayerMicroBatchOutput):
    block_residual: list[torch.Tensor | None]


class KimiMoEDecoderLayerOutput(MoEDecoderLayerOutput):
    block_residual: torch.Tensor | None


class KimiMoEDecoderLayerMicroBatchOutput(MoEDecoderLayerMicroBatchOutput):
    block_residual: list[torch.Tensor | None]


class KimiDenseDecoderLayer(DenseDecoderLayer):
    """Dense first-k layer with SiTU MLP and AttnRes cross-layer residual."""

    def __init__(
        self,
        *args,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
        attn_res_block_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.layer_idx = int(kwargs.get("layer_idx", 0))
        self.attn_res_block_size = attn_res_block_size
        hidden_act = kwargs["hidden_act"]
        if hidden_act == "situ":
            self.mlp = DenseMLP(
                hidden_size=kwargs["hidden_size"],
                intermediate_size=kwargs["intermediate_size"],
                bias=kwargs.get("mlp_bias", False),
                hidden_act="situ",
                float8_cfg=kwargs.get("float8_cfg"),
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
        if attn_res_block_size is not None:
            eps = kwargs.get("rms_norm_eps", 1e-6)
            hs = kwargs["hidden_size"]
            self.self_attention_res_norm = RMSNorm(hs, eps=eps)
            self.mlp_res_norm = RMSNorm(hs, eps=eps)
            self.self_attention_res_proj = build_linear(hs, 1, bias=False)
            self.mlp_res_proj = build_linear(hs, 1, bias=False)

    @override
    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
        block_residual: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> KimiDenseDecoderLayerOutput | KimiDenseDecoderLayerMicroBatchOutput:
        if isinstance(hidden_states, list):
            n = len(hidden_states)
            assert isinstance(position_embeddings, list) and isinstance(seq_ctx, list)
            br_list: list[torch.Tensor | None]
            if block_residual is None:
                br_list = [None] * n
            else:
                assert isinstance(block_residual, list) and len(block_residual) == n
                br_list = list(block_residual)
            results = [
                self._kimi_forward(hs, pe, sc, br)
                for hs, pe, sc, br in zip(hidden_states, position_embeddings, seq_ctx, br_list)
            ]
            return {
                "hidden_states": [r["hidden_states"] for r in results],
                "block_residual": [r["block_residual"] for r in results],
            }

        assert isinstance(position_embeddings, tuple) and isinstance(seq_ctx, SequenceContext)
        assert block_residual is None or isinstance(block_residual, torch.Tensor)
        return self._kimi_forward(hidden_states, position_embeddings, seq_ctx, block_residual)

    def _kimi_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        block_residual: torch.Tensor | None,
    ) -> KimiDenseDecoderLayerOutput:
        if self.attn_res_block_size is None:
            out = DenseDecoderLayer._forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )
            return {"hidden_states": out, "block_residual": block_residual}

        return self._forward_attn_residual(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            block_residual=block_residual,
            mlp_fn=self.mlp,
        )

    def _forward_attn_residual(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        block_residual: torch.Tensor | None,
        mlp_fn: nn.Module,
    ) -> KimiDenseDecoderLayerOutput:
        assert self.attn_res_block_size is not None
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum: torch.Tensor | None = hidden_states

        if block_residual is None:
            block_residual = hidden_states.new_zeros(batch_size * seq_len, 0, hidden_size)

        if block_residual.shape[1] > 0:
            assert prefix_sum is not None
            hidden_states = apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            assert prefix_sum is not None
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)],
                dim=1,
            )
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs: AttnOutputs = _run_self_attn(
            self.self_attn, hidden_states, seq_ctx, position_embeddings
        )
        attn_out = attn_outputs["projected_output"]
        if prefix_sum is not None:
            prefix_sum = prefix_sum + attn_out
        else:
            prefix_sum = attn_out

        hidden_states = apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = mlp_fn(hidden_states)

        prefix_sum = hidden_states if prefix_sum is None else prefix_sum + hidden_states
        return {"hidden_states": prefix_sum, "block_residual": block_residual}


class KimiMoEDecoderLayer(MoEDecoderLayer):
    """MoE decoder with LatentMoE, SiTU shared experts, and AttnRes."""

    def __init__(
        self,
        *args,
        routed_expert_hidden_size: int | None = None,
        latent_moe_use_norm: bool = False,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
        attn_res_block_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.attn_res_block_size = attn_res_block_size

        float8_cfg: Float8Config | None = kwargs.get("float8_cfg")
        moe_act_fn_cfg: MoEActFnConfig = kwargs["moe_act_fn_cfg"]
        moe_intermediate_size = kwargs["moe_intermediate_size"]
        n_routed_experts = kwargs["n_routed_experts"]
        n_shared_experts = kwargs["n_shared_experts"]
        hidden_act = kwargs["hidden_act"]
        mlp_bias = kwargs.get("mlp_bias", False)
        moe_bias = kwargs.get("moe_bias", False)
        rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)

        if n_shared_experts > 0 and hidden_act == "situ":
            self.shared_experts = MoEMLP(
                hidden_size=self.hidden_size,
                n_shared_experts=n_shared_experts,
                moe_intermediate_size=moe_intermediate_size,
                hidden_act=hidden_act,
                mlp_bias=mlp_bias,
                float8_cfg=float8_cfg,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )

        if routed_expert_hidden_size is not None:
            self.routed_expert_down_proj = build_linear(
                self.hidden_size, routed_expert_hidden_size, bias=False, float8_cfg=float8_cfg
            )
            self.routed_expert_up_proj = build_linear(
                routed_expert_hidden_size, self.hidden_size, bias=False, float8_cfg=float8_cfg
            )
            if latent_moe_use_norm:
                self.routed_expert_norm = RMSNorm(routed_expert_hidden_size, eps=rms_norm_eps)
            self.experts = MoEBlock(
                hidden_size=routed_expert_hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                n_routed_experts=n_routed_experts,
                moe_bias=moe_bias,
                ep_mesh=self.ep_mesh,
                expert_tp_mesh=kwargs.get("expert_tp_mesh"),
                float8_cfg=float8_cfg,
                moe_act_fn_cfg=moe_act_fn_cfg,
                ep_tp_mesh=kwargs.get("ep_tp_mesh"),
                expert_backend=kwargs.get("expert_backend", "grouped_gemm"),
                sonicmoe_cfg=kwargs.get("sonicmoe_cfg"),
            )

        if attn_res_block_size is not None:
            self.self_attention_res_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
            self.mlp_res_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
            self.self_attention_res_proj = build_linear(self.hidden_size, 1, bias=False)
            self.mlp_res_proj = build_linear(self.hidden_size, 1, bias=False)

    @override
    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        *,
        seq_ctx: SequenceContext | list[SequenceContext],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        block_residual: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> KimiMoEDecoderLayerOutput | KimiMoEDecoderLayerMicroBatchOutput:
        if isinstance(hidden_states, list):
            n = len(hidden_states)
            assert isinstance(seq_ctx, list) and isinstance(position_embeddings, list)
            br_list: list[torch.Tensor | None]
            if block_residual is None:
                br_list = [None] * n
            else:
                assert isinstance(block_residual, list) and len(block_residual) == n
                br_list = list(block_residual)
            results = [
                self._kimi_forward(hs, sc, pe, br)
                for hs, sc, pe, br in zip(hidden_states, seq_ctx, position_embeddings, br_list)
            ]
            return {
                "hidden_states": [r["hidden_states"] for r in results],
                "router_logits": [r["router_logits"] for r in results],
                "router_weights": [r["router_weights"] for r in results],
                "router_topk_ids": [r["router_topk_ids"] for r in results],
                "block_residual": [r["block_residual"] for r in results],
            }

        assert isinstance(seq_ctx, SequenceContext)
        assert isinstance(position_embeddings, tuple)
        assert block_residual is None or isinstance(block_residual, torch.Tensor)
        return self._kimi_forward(hidden_states, seq_ctx, position_embeddings, block_residual)

    def _kimi_forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_residual: torch.Tensor | None,
    ) -> KimiMoEDecoderLayerOutput:
        if self.attn_res_block_size is None:
            out = self._forward_moe_body(
                hidden_states=hidden_states,
                seq_ctx=seq_ctx,
                position_embeddings=position_embeddings,
                residual_after_attn=True,
            )
            out["block_residual"] = block_residual
            return out

        return self._forward_attn_residual(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            block_residual=block_residual,
        )

    def _forward_attn_residual(
        self,
        *,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_residual: torch.Tensor | None,
    ) -> KimiMoEDecoderLayerOutput:
        assert self.attn_res_block_size is not None
        batch_size, seq_len, hidden_size = hidden_states.shape
        prefix_sum: torch.Tensor | None = hidden_states

        if block_residual is None:
            block_residual = hidden_states.new_zeros(batch_size * seq_len, 0, hidden_size)

        if block_residual.shape[1] > 0:
            assert prefix_sum is not None
            hidden_states = apply_attn_res(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
            ).view(batch_size, seq_len, hidden_size)

        if self.layer_idx % self.attn_res_block_size == 0:
            assert prefix_sum is not None
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)],
                dim=1,
            )
            prefix_sum = None

        # Attention (no classic residual — AttnRes owns the mix)
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs: AttnOutputs = _run_self_attn(
            self.self_attn, hidden_states, seq_ctx, position_embeddings
        )
        attn_out = attn_outputs["projected_output"]
        if prefix_sum is not None:
            prefix_sum = prefix_sum + attn_out
        else:
            prefix_sum = attn_out

        hidden_states = apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
        ).view(batch_size, seq_len, hidden_size)

        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_out = self._moe_experts_forward(hidden_states=hidden_states)
        router_results = moe_out["router_results"]
        combined = moe_out["combined_hidden_states"]

        # Shared experts see post-norm MoE input (HF identity), then accumulate into prefix.
        if self.n_shared_experts > 0:
            combined = combined + self._shared_experts_forward(hidden_states=hidden_states)

        # AttnRes: no classic residual add; accumulate MLP/MoE into prefix_sum.
        # HF uses hidden_factor only in classic path; keep identity scale here.
        prefix_sum = combined if prefix_sum is None else prefix_sum + combined

        return {
            "hidden_states": prefix_sum,
            "router_logits": router_results["logits"],
            "router_weights": router_results["router_weights"],
            "router_topk_ids": router_results["topk_ids"],
            "block_residual": block_residual,
        }

    def _forward_moe_body(
        self,
        *,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        residual_after_attn: bool,
    ) -> KimiMoEDecoderLayerOutput:
        """Classic residual MoE path (AttnRes disabled)."""
        del residual_after_attn
        residual, hidden_states, router_results, attn_outputs = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.TRAINING,
        )
        moe = self._moe_experts_forward(hidden_states=hidden_states)
        combined = moe["combined_hidden_states"]
        if self.n_shared_experts > 0:
            shared = self._shared_experts_forward(hidden_states=hidden_states)
        else:
            shared = None
        hidden_states = self._post_moe_forward(
            combined_hidden_states=combined,
            residual=residual,
            shared_experts_out=shared,
        )
        return {
            "hidden_states": hidden_states,
            "router_logits": router_results["logits"],
            "router_weights": router_results["router_weights"],
            "router_topk_ids": router_results["topk_ids"],
            "block_residual": None,
        }

    def _moe_experts_forward(self, *, hidden_states: torch.Tensor) -> dict:
        """Gate + LatentMoE down/experts/up (+ optional shared is caller-side)."""
        origin_shape = hidden_states.shape
        router_results: RouterResults = self.gate(hidden_states)
        expert_in = hidden_states
        if self.routed_expert_hidden_size is not None:
            expert_in = self.routed_expert_down_proj(expert_in)

        if self.experts.uses_sonicmoe:
            raise NotImplementedError("Kimi LatentMoE does not support sonicmoe yet")

        assert self.dispatcher is not None
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=expert_in.view(-1, expert_in.shape[-1]),
            topk_ids=router_results["topk_ids"],
            topk_weights=router_results["topk_weights"],
        )
        dispatched = self.dispatcher.dispatch(
            pre_dispatched=pre_dispatched,
            topk_weights=router_results["topk_weights"],
            decoding=False,
        )
        post_dispatched = self.dispatcher.dispatch_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
        )
        experts_out = self.experts(
            post_dispatched["hidden_states"],
            post_dispatched["tokens_per_expert"],
            decoding=False,
        )
        pre_combined = self.dispatcher.combine_preprocess(
            hidden_states=experts_out,
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            decoding=False,
        )
        combined = self.dispatcher.combine(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            decoding=False,
        )
        post_combined = self.dispatcher.combine_postprocess(
            pre_dispatched=pre_dispatched,
            dispatched=dispatched,
            post_dispatched=post_dispatched,
            pre_combined=pre_combined,
            combined=combined,
        )
        combined_hidden_states = post_combined["hidden_states"].view(origin_shape[0], origin_shape[1], -1)

        if self.routed_expert_hidden_size is not None:
            if self.latent_moe_use_norm:
                combined_hidden_states = self.routed_expert_norm(combined_hidden_states)
            combined_hidden_states = self.routed_expert_up_proj(combined_hidden_states)

        return {
            "combined_hidden_states": combined_hidden_states,
            "router_results": router_results,
        }
