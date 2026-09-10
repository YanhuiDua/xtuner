"""Multi-Token Prediction (MTP) Layer implementation."""

from typing import Literal

import torch
import torch.nn as nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module import RMSNorm
from xtuner.v1.module.decoder_layer.moe_decoder_layer import (
    MoEDecoderLayerMicroBatchOutput,
    MoEDecoderLayerOutput,
)
from xtuner.v1.module.linear import build_linear


class MTPLayer(nn.Module):
    """Single Multi-Token Prediction (MTP) layer.

    MTP Layer wraps a standard decoder layer with MTP-specific preprocessing
    and postprocessing. The structure is:

        [enorm + hnorm + projection] → [DecoderLayer] → [final_layernorm]

    The k-th MTP layer predicts the (i+k)-th token by combining:
    1. Hidden states from the previous MTP layer (or main model)
    2. Embedding of the future token at position (i+k)

    Note: The decoder layer's internal normalization (input_layernorm) is preserved
    for simplicity and modularity. While this adds a small computational overhead,
    it allows MTP to work with any decoder layer implementation (Dense, MoE, etc.)
    without modification.

    Args:
        hidden_size (int): Hidden dimension size.
        rms_norm_eps (float): Epsilon for RMSNorm.
        rms_norm_type (str): Type of RMSNorm ("default" or "zero_centered").
        decoder_layer (nn.Module): A fully constructed decoder layer instance.
            This can be DenseDecoderLayer, MoEDecoderLayer, or any custom decoder layer
            that implements the standard forward signature.
        float8_cfg: Float8 configuration for the projection layer.

    Example:
        >>> from xtuner.v1.module.decoder_layer import DenseDecoderLayer
        >>> decoder_layer = DenseDecoderLayer(
        ...     hidden_size=512,
        ...     intermediate_size=2048,
        ...     ...
        ... )
        >>> mtp_layer = MTPLayer(
        ...     hidden_size=512,
        ...     rms_norm_eps=1e-6,
        ...     rms_norm_type="default",
        ...     decoder_layer=decoder_layer,
        ... )
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        rms_norm_eps: float,
        rms_norm_type: Literal["default", "zero_centered"],
        decoder_layer: nn.Module,
        float8_cfg=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # MTP-specific preprocessing components
        self.enorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.hnorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)
        self.eh_proj = build_linear(
            hidden_size * 2,
            hidden_size,
            bias=False,
            float8_cfg=float8_cfg,
        )

        # Core decoder layer (Dense, MoE, or any custom implementation)
        self.decoder_layer = decoder_layer

        # MTP-specific postprocessing component
        self.final_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps, type=rms_norm_type)

    def forward(
        self,
        *hidden_states: torch.Tensor | list[torch.Tensor],
        future_embeddings: torch.Tensor | list[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx: SequenceContext | list[SequenceContext],
    ) -> MoEDecoderLayerOutput | MoEDecoderLayerMicroBatchOutput:
        """Forward pass through the MTP layer.

        Mirrors :meth:`MoEDecoderLayer.forward`: passing lists runs ``N`` micro-batches together
        (intra-layer micro-batching / domino EP). The per-microbatch preprocessing
        (enorm/hnorm/eh_proj) is run independently and a single underlying decoder forward is
        issued, so the inner MoE EP communication can be overlapped across micro-batches.

        Args:
            hidden_states (torch.Tensor | list[torch.Tensor]): Hidden states, one tensor per
                micro-batch.
            future_embeddings (torch.Tensor | list[torch.Tensor]): Embeddings of the future tokens,
                aligned with ``hidden_states``.
            position_embeddings (tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]):
                Rotary position embeddings ``(cos, sin)``, aligned with ``hidden_states``.
            seq_ctx (SequenceContext | list[SequenceContext]): Sequence context, aligned with
                ``hidden_states``.
        Returns:
            MoEDecoderLayerOutput | MoEDecoderLayerMicroBatchOutput: The wrapped decoder layer's
            outputs with the MTP final layernorm applied to the hidden states.
        """
        if len(hidden_states) == 1 and isinstance(hidden_states[0], list):
            hidden_states = tuple(hidden_states[0])
        if not hidden_states:
            raise ValueError("MTPLayer requires at least one hidden-state tensor")

        if len(hidden_states) == 1:
            assert isinstance(future_embeddings, torch.Tensor), (
                "future_embeddings should be a Tensor in single-microbatch mode"
            )
            assert isinstance(seq_ctx, SequenceContext), (
                "seq_ctx should be a SequenceContext instance in single-microbatch mode"
            )
            assert isinstance(position_embeddings, tuple) and len(position_embeddings) == 2, (
                "position_embeddings should be a (cos, sin) tuple in single-microbatch mode"
            )
            return self._forward(
                hidden_states=hidden_states[0],
                future_embeddings=future_embeddings,
                position_embeddings=position_embeddings,
                seq_ctx=seq_ctx,
            )

        assert isinstance(future_embeddings, list), (
            "future_embeddings should be a list aligned with hidden_states in multi-microbatch mode"
        )
        assert isinstance(seq_ctx, list), (
            "seq_ctx should be a list aligned with hidden_states in multi-microbatch mode"
        )
        assert isinstance(position_embeddings, list), (
            "position_embeddings should be a list aligned with hidden_states in multi-microbatch mode"
        )
        return self._micro_batch_forward(
            hidden_states_list=list(hidden_states),
            future_embeddings_list=future_embeddings,
            position_embeddings_list=position_embeddings,
            seq_ctx_list=seq_ctx,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
    ) -> MoEDecoderLayerOutput:
        projected = self._preprocess(hidden_states=hidden_states, future_embeddings=future_embeddings)

        layer_outputs = self.decoder_layer(
            projected,
            position_embeddings=position_embeddings,
            seq_ctx=seq_ctx,
        )

        if isinstance(layer_outputs, torch.Tensor):
            hidden_states = layer_outputs
            empty = hidden_states.new_empty(0)
            router_results, router_weights, router_topk_ids = empty, empty, empty
        elif isinstance(layer_outputs, dict):
            hidden_states = layer_outputs["hidden_states"]
            router_results = layer_outputs["router_logits"]
            router_weights = layer_outputs["router_weights"]
            router_topk_ids = layer_outputs["router_topk_ids"]
        else:
            hidden_states, router_results, router_weights, router_topk_ids = layer_outputs

        hidden_states = self.final_layernorm(hidden_states)
        return {
            "hidden_states": hidden_states,
            "router_logits": router_results,
            "router_weights": router_weights,
            "router_topk_ids": router_topk_ids,
        }

    def _micro_batch_forward(
        self,
        *,
        hidden_states_list: list[torch.Tensor],
        future_embeddings_list: list[torch.Tensor],
        position_embeddings_list: list[tuple[torch.Tensor, torch.Tensor]],
        seq_ctx_list: list[SequenceContext],
    ) -> MoEDecoderLayerMicroBatchOutput:
        n = len(hidden_states_list)
        assert n > 0, "MTP micro-batch input cannot be empty"
        assert len(future_embeddings_list) == n and len(position_embeddings_list) == n and len(seq_ctx_list) == n, (
            "All per-microbatch inputs must share the same length"
        )

        # Run MTP preprocessing eagerly across all micro-batches so the underlying decoder
        # layer can overlap its EP communication in a single fused forward.
        projected_list = [
            self._preprocess(hidden_states=h, future_embeddings=e)
            for h, e in zip(hidden_states_list, future_embeddings_list)
        ]
        layer_results = self.decoder_layer(
            projected_list,
            position_embeddings=position_embeddings_list,
            seq_ctx=seq_ctx_list,
        )
        if isinstance(layer_results, dict):
            hidden_out = [self.final_layernorm(h) for h in layer_results["hidden_states"]]
            router_logits = layer_results["router_logits"]
            router_weights = layer_results["router_weights"]
            router_topk_ids = layer_results["router_topk_ids"]
        else:
            assert isinstance(layer_results, tuple)
            if len(layer_results) == n:
                hidden_out = [self.final_layernorm(h) for h in layer_results]
                empty = hidden_out[0].new_empty(0)
                router_logits = [empty] * n
                router_weights = [empty] * n
                router_topk_ids = [empty] * n
            else:
                assert len(layer_results) == 4 * n, (
                    "Multi-microbatch MTP requires the wrapped decoder layer to return either "
                    f"{n} dense outputs or {4 * n} MoE outputs, got {len(layer_results)}"
                )
                hidden_out = [self.final_layernorm(h) for h in layer_results[:n]]
                router_logits = list(layer_results[n : 2 * n])
                router_weights = list(layer_results[2 * n : 3 * n])
                router_topk_ids = list(layer_results[3 * n :])

        return {
            "hidden_states": hidden_out,
            "router_logits": router_logits,
            "router_weights": router_weights,
            "router_topk_ids": router_topk_ids,
        }

    def _preprocess(
        self,
        *,
        hidden_states: torch.Tensor,
        future_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize embeddings and hidden states separately so both inputs share a numerical
        # range, then concatenate along the last dim and project back to ``hidden_size``.
        normalized_embedding = self.enorm(future_embeddings)
        normalized_hidden = self.hnorm(hidden_states)
        combined = torch.cat([normalized_embedding, normalized_hidden], dim=-1)
        return self.eh_proj(combined)
