# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.float8.config import Float8Config
from xtuner.v1.module.attention.concept_olmo import ConceptOlmoRMSNorm
from xtuner.v1.module.decoder_layer.concept_olmo_decoder_layer import ConceptOlmoDecoderLayer
from xtuner.v1.module.linear import build_linear


class CheckpointRMSNorm(ConceptOlmoRMSNorm):
    """RMSNorm carrying the empty extra-state tensor found in the
    checkpoint."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps)
        self.register_buffer("_extra_state", torch.empty(0, dtype=torch.uint8), persistent=True)

    def get_extra_state(self) -> torch.Tensor:
        return self._extra_state

    def set_extra_state(self, state: torch.Tensor) -> None:
        self._extra_state = state


class DepthDD(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        use_softmax: bool,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        num_prev = layer_idx + 2
        self.w1 = build_linear(hidden_size, num_prev, bias=False, float8_cfg=float8_cfg)
        self.w2 = build_linear(num_prev, num_prev, bias=False, float8_cfg=float8_cfg)
        self.static_a = nn.Parameter(torch.zeros(num_prev))
        self.use_softmax = use_softmax

    def forward(self, hidden_states: torch.Tensor, history_states: torch.Tensor) -> torch.Tensor:
        history = torch.cat((history_states, hidden_states.unsqueeze(2)), dim=2)
        weights = self.w2(F.gelu(self.w1(hidden_states)))
        weights = weights + self.static_a.view(1, 1, -1)
        if self.use_softmax:
            weights = weights.softmax(dim=-1)
        return (weights.unsqueeze(-1) * history).sum(dim=2)


class ResidualRoute(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_source_states: int,
        use_softmax: bool = True,
        float8_cfg: Float8Config | None = None,
    ):
        super().__init__()
        route_hidden_size = max(1, num_source_states)
        self.w1 = build_linear(hidden_size, route_hidden_size, bias=False, float8_cfg=float8_cfg)
        self.w2 = build_linear(route_hidden_size, num_source_states, bias=False, float8_cfg=float8_cfg)
        self.residual_diag = nn.Parameter(torch.zeros(hidden_size))
        self.use_softmax = use_softmax

    def forward(
        self,
        target_hidden: torch.Tensor,
        source_states: torch.Tensor | None,
        residual_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_states is None or source_states.shape[2] == 0:
            return target_hidden
        active_sources = source_states.shape[2]
        weights = self.w2(F.gelu(self.w1(target_hidden)))[..., -active_sources:]
        if self.use_softmax:
            weights = weights.softmax(dim=-1)
        source_mix = torch.einsum("bsm,bsmh->bsh", weights, source_states)
        update = source_mix * self.residual_diag.to(source_mix.dtype)
        if residual_scale is not None:
            update = update * residual_scale.to(update.dtype)
        return target_hidden + update.to(target_hidden.dtype)


class ConceptRoute(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.concept_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.final_diag = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        final_concept_state: torch.Tensor,
        final_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        concept = self.concept_norm(final_concept_state)
        update = concept * self.final_diag.to(concept.dtype)
        if final_scale is not None:
            update = update * final_scale.to(update.dtype)
        return hidden_states + update.to(hidden_states.dtype)


class ConceptQuantizer(nn.Module):
    def __init__(self, hidden_size: int, num_codebooks: int, codebook_size: int):
        super().__init__()
        if hidden_size % num_codebooks != 0:
            raise ValueError("hidden_size must be divisible by num_codebooks")
        code_dim = hidden_size // num_codebooks
        self.codebook = nn.ParameterList(
            [nn.Parameter(torch.empty(codebook_size, code_dim)) for _ in range(num_codebooks)]
        )

    def transformed_codebook(self) -> torch.Tensor:
        return torch.stack(tuple(self.codebook), dim=0)


class EncoderStage(nn.Module):
    def __init__(self, transformer: ConceptOlmoDecoderLayer, depth_dd: DepthDD):
        super().__init__()
        self.transformer = transformer
        self.depth_dd = depth_dd

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        history_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.transformer(hidden_states, position_embeddings, seq_ctx)
        return self.depth_dd(raw, history_states), raw


class ConceptStage(nn.Module):
    def __init__(
        self,
        transformer: ConceptOlmoDecoderLayer,
        depth_dd: DepthDD,
        read_encoder_route: ResidualRoute,
    ):
        super().__init__()
        self.transformer = transformer
        self.depth_dd = depth_dd
        self.read_encoder_route = read_encoder_route

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        history_states: torch.Tensor,
        encoder_concept_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.transformer(hidden_states, position_embeddings, seq_ctx)
        hidden_states = self.depth_dd(raw, history_states)
        hidden_states = self.read_encoder_route(hidden_states, encoder_concept_states)
        return hidden_states, raw


class DecoderStage(nn.Module):
    def __init__(
        self,
        transformer: ConceptOlmoDecoderLayer,
        decoder_dd: DepthDD,
        concept_route: ConceptRoute,
        read_encoder_route: ResidualRoute,
        read_concept_route: ResidualRoute,
    ):
        super().__init__()
        self.transformer = transformer
        self.decoder_dd = decoder_dd
        self.concept_route = concept_route
        self.read_encoder_route = read_encoder_route
        self.read_concept_route = read_concept_route

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        history_states: torch.Tensor,
        repeated_concepts: torch.Tensor,
        encoder_states: torch.Tensor,
        concept_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.transformer(hidden_states, position_embeddings, seq_ctx)
        hidden_states = self.decoder_dd(raw, history_states)
        hidden_states = self.concept_route(hidden_states, repeated_concepts, gate[0])
        hidden_states = self.read_encoder_route(hidden_states, encoder_states)
        hidden_states = self.read_concept_route(hidden_states, concept_states, gate[1])
        return hidden_states, raw
