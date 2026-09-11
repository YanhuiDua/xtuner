# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Linear MoE decoder: LatentMoE + SiTU; AttnRes params registered for load."""

from __future__ import annotations

import torch
from typing_extensions import override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.decoder_layer.dense_decoder_layer import DenseDecoderLayer, DenseMLP
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEActFnConfig,
    MoEBlock,
    MoEDecoderLayer,
    MoEDecoderLayerOutput,
    MoEMLP,
)
from xtuner.v1.module.linear import build_linear
from xtuner.v1.utils import ForwardState


class KimiMoEDecoderLayerOutput(MoEDecoderLayerOutput):
    block_residual: torch.Tensor | None


class KimiDenseDecoderLayer(DenseDecoderLayer):
    """Dense first-k layer with SiTU MLP and optional AttnRes parameters."""

    def __init__(
        self,
        *args,
        situ_beta: float = 1.0,
        situ_linear_beta: float | None = None,
        attn_res_block_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
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


class KimiMoEDecoderLayer(MoEDecoderLayer):
    """MoE decoder with LatentMoE, SiTU shared experts, AttnRes params for load."""

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
    def _forward(
        self,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_kwargs: dict[str, object] | None = None,
    ) -> MoEDecoderLayerOutput:
        residual, hidden_states, router_results, attn_outputs = self._pre_moe_forward(
            hidden_states=hidden_states,
            seq_ctx=seq_ctx,
            position_embeddings=position_embeddings,
            state=ForwardState.TRAINING,
            attention_kwargs=attention_kwargs,
        )
        origin_shape = hidden_states.shape
        identity_for_shared = hidden_states

        if self.routed_expert_hidden_size is not None:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        if self.experts.uses_sonicmoe:
            raise NotImplementedError("Kimi LatentMoE does not support sonicmoe in phase-1")

        assert self.dispatcher is not None
        pre_dispatched = self.dispatcher.dispatch_preprocess(
            hidden_states=hidden_states.view(-1, hidden_states.shape[-1]),
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

        if self.n_shared_experts > 0:
            shared_experts_out = self._shared_experts_forward(hidden_states=identity_for_shared)
        else:
            shared_experts_out = None

        hidden_states = self._post_moe_forward(
            combined_hidden_states=combined_hidden_states,
            residual=residual,
            shared_experts_out=shared_experts_out,
        )
        return self._build_output(
            hidden_states=hidden_states,
            router_results=router_results,
            attn_outputs=attn_outputs,
        )
