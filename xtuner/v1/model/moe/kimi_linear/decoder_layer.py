# Copyright (c) OpenMMLab. All rights reserved.
"""Kimi Linear MoE decoder: LatentMoE + SiTU + AttnRes cross-layer residual."""

from __future__ import annotations

from typing import Callable, Protocol

import torch
import torch.nn as nn
from typing_extensions import override

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import AttnOutputs, RMSNorm, RouterResults
from xtuner.v1.module.decoder_layer.dense_decoder_layer import (
    DenseDecoderLayer,
    DenseDecoderLayerMicroBatchOutput,
    DenseDecoderLayerOutput,
)
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayer,
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)
from xtuner.v1.module.linear import build_linear
from xtuner.v1.utils import ForwardState


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


class _AttnResLayer(Protocol):
    attn_res_block_size: int | None
    layer_idx: int
    self_attention_res_proj: nn.Linear
    self_attention_res_norm: RMSNorm
    mlp_res_proj: nn.Linear
    mlp_res_norm: RMSNorm
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    self_attn: nn.Module


def run_attn_res_forward(
    layer: _AttnResLayer,
    *,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    seq_ctx: SequenceContext,
    block_residual: torch.Tensor | None,
    ffn_fn: Callable[[torch.Tensor], tuple[torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Single AttnRes orchestration shared by dense and MoE decoder layers.

    ``ffn_fn`` runs after ``post_attention_layernorm`` and returns
    ``(ffn_hidden_states, extras)`` — extras carry MoE router fields (empty for dense).
    """
    assert layer.attn_res_block_size is not None
    batch_size, seq_len, hidden_size = hidden_states.shape
    prefix_sum: torch.Tensor | None = hidden_states

    if block_residual is None:
        block_residual = hidden_states.new_zeros(batch_size * seq_len, 0, hidden_size)

    if block_residual.shape[1] > 0:
        assert prefix_sum is not None
        hidden_states = apply_attn_res(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        ).view(batch_size, seq_len, hidden_size)

    if layer.layer_idx % layer.attn_res_block_size == 0:
        assert prefix_sum is not None
        block_residual = torch.cat(
            [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)],
            dim=1,
        )
        prefix_sum = None

    # Attention (no classic residual — AttnRes owns the mix)
    hidden_states = layer.input_layernorm(hidden_states)
    attn_outputs: AttnOutputs = layer.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        seq_ctx=seq_ctx,
    )
    attn_out = attn_outputs["projected_output"]
    prefix_sum = attn_out if prefix_sum is None else prefix_sum + attn_out

    hidden_states = apply_attn_res(
        prefix_sum.view(-1, hidden_size),
        block_residual,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).view(batch_size, seq_len, hidden_size)

    hidden_states = layer.post_attention_layernorm(hidden_states)
    ffn_out, extras = ffn_fn(hidden_states)
    # AttnRes: accumulate FFN into prefix_sum (no classic residual add).
    prefix_sum = ffn_out if prefix_sum is None else prefix_sum + ffn_out
    return prefix_sum, block_residual, extras


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
        attn_res_block_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attn_res_block_size = attn_res_block_size
        if attn_res_block_size is not None:
            eps = kwargs.get("rms_norm_eps", 1e-6)
            hs = self.hidden_size
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

        def _dense_ffn(hs: torch.Tensor) -> tuple[torch.Tensor, dict]:
            return self.mlp(hs), {}

        prefix_sum, block_residual, _ = run_attn_res_forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            block_residual=block_residual,
            ffn_fn=_dense_ffn,
        )
        return {"hidden_states": prefix_sum, "block_residual": block_residual}


class KimiMoEDecoderLayer(MoEDecoderLayer):
    """MoE decoder with LatentMoE, SiTU shared experts, and AttnRes."""

    def __init__(
        self,
        *args,
        attn_res_block_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attn_res_block_size = attn_res_block_size
        if attn_res_block_size is not None:
            eps = kwargs.get("rms_norm_eps", 1e-6)
            hs = self.hidden_size
            self.self_attention_res_norm = RMSNorm(hs, eps=eps)
            self.mlp_res_norm = RMSNorm(hs, eps=eps)
            self.self_attention_res_proj = build_linear(hs, 1, bias=False)
            self.mlp_res_proj = build_linear(hs, 1, bias=False)

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
            )
            out["block_residual"] = block_residual
            return out

        def _moe_ffn(hs: torch.Tensor) -> tuple[torch.Tensor, dict]:
            moe_out = self._moe_experts_forward(hidden_states=hs)
            combined = moe_out["combined_hidden_states"]
            # Shared experts see post-norm MoE input (HF identity).
            if self.n_shared_experts > 0:
                combined = combined + self._shared_experts_forward(hidden_states=hs)
            router_results = moe_out["router_results"]
            return combined, {
                "router_logits": router_results["logits"],
                "router_weights": router_results["router_weights"],
                "router_topk_ids": router_results["topk_ids"],
            }

        prefix_sum, block_residual, extras = run_attn_res_forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
            block_residual=block_residual,
            ffn_fn=_moe_ffn,
        )
        return {
            "hidden_states": prefix_sum,
            "block_residual": block_residual,
            **extras,
        }

    def _forward_moe_body(
        self,
        *,
        hidden_states: torch.Tensor,
        seq_ctx: SequenceContext,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> KimiMoEDecoderLayerOutput:
        """Classic residual MoE path (AttnRes disabled)."""
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
        """Gate + LatentMoE routed path (shared experts stay caller-side)."""
        router_results: RouterResults = self.gate(hidden_states)
        combined_hidden_states = self._dispatch_experts_combine(
            hidden_states=hidden_states,
            router_results=router_results,
        )
        return {
            "combined_hidden_states": combined_hidden_states,
            "router_results": router_results,
        }
