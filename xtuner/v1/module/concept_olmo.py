# Copyright (c) OpenMMLab. All rights reserved.

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.attention.concept_olmo import ConceptOlmoRMSNorm
from xtuner.v1.module.decoder_layer.concept_olmo_decoder_layer import ConceptOlmoDecoderLayer
from xtuner.v1.module.linear import build_linear


class CheckpointRMSNorm(ConceptOlmoRMSNorm):
    """RMSNorm carrying the empty extra-state tensor found in some checkpoints.

    Megatron ``DUMP_INIT_HF`` exports parameters only, so this buffer is not
    persistent and is not required by ``from_hf``.
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps)
        self.register_buffer("_extra_state", torch.empty(0, dtype=torch.uint8), persistent=False)

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
    ):
        super().__init__()
        num_prev = layer_idx + 2
        self.w1 = build_linear(hidden_size, num_prev, bias=False)
        self.w2 = build_linear(num_prev, num_prev, bias=False)
        self.static_a = nn.Parameter(torch.zeros(num_prev))
        self.use_softmax = use_softmax

    def forward(
        self,
        hidden_states: torch.Tensor,
        history: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        weights = self.w2(F.gelu(self.w1(hidden_states)))
        weights = weights + self.static_a.view(1, 1, -1)
        if self.use_softmax:
            weights = weights.softmax(dim=-1)

        # Preserve the eager/HuggingFace reduction order. Compiled stages can
        # instead keep history as separate inputs and fuse this fixed-length
        # multiply-add chain without materializing a growing [B, S, D, H]
        # history tensor at every layer.
        if not torch.compiler.is_compiling():
            history_tensor = torch.stack(history, dim=2)
            return (weights.unsqueeze(-1) * history_tensor).sum(dim=2)

        mixed = history[0] * weights[..., 0:1]
        for index, state in enumerate(history[1:], start=1):
            mixed = torch.addcmul(mixed, state, weights[..., index : index + 1])
        return mixed


class _CompiledRouteReduce(torch.autograd.Function):
    """Use complementary kernels for the two directions of route reduction.

    The fixed-length addcmul chain is substantially faster than a Triton reduction in forward, while spelling out the
    analytical backward avoids the slice/select-heavy graph produced by autograd for that chain.
    """

    @staticmethod
    def forward(ctx, weights: torch.Tensor, source_states: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(weights, source_states)
        source_mix = source_states[:, :, 0] * weights[..., 0:1]
        for index in range(1, source_states.shape[2]):
            source_mix = torch.addcmul(
                source_mix,
                source_states[:, :, index],
                weights[..., index : index + 1],
            )
        return source_mix

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, source_states = ctx.saved_tensors
        grad_weights = (grad_output.unsqueeze(2) * source_states).sum(dim=-1)
        grad_source_states = grad_output.unsqueeze(2) * weights.unsqueeze(-1)
        return grad_weights, grad_source_states


class ResidualRoute(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_source_states: int,
        use_softmax: bool = True,
    ):
        super().__init__()
        route_hidden_size = max(1, num_source_states)
        self.w1 = build_linear(hidden_size, route_hidden_size, bias=False)
        self.w2 = build_linear(route_hidden_size, num_source_states, bias=False)
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

        # Keep the eager/HuggingFace GEMV path for exact checkpoint parity.
        # Compiled stages fuse the elementwise multiply and source reduction,
        # avoiding a batched GEMV for every route in every decoder layer.
        if not torch.compiler.is_compiling():
            source_mix = torch.einsum("bsm,bsmh->bsh", weights, source_states)
        else:
            source_mix = _CompiledRouteReduce.apply(weights, source_states)
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
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.code_dim = hidden_size // num_codebooks
        # Keep the codebooks in one FSDP-friendly parameter. HuggingFace stores
        # one [codebook_size, code_dim] tensor per codebook, so flattening the
        # first two dimensions also makes checkpoint loading/saving a direct
        # concatenate/split along dim 0.
        self.codebook = nn.Parameter(torch.empty(num_codebooks * codebook_size, self.code_dim))

    def transformed_codebook(self) -> torch.Tensor:
        return self.codebook.view(self.num_codebooks, self.codebook_size, self.code_dim)


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
        *history_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.transformer(hidden_states, position_embeddings, seq_ctx)
        history = (*history_states, raw)
        hidden_states = self.depth_dd(raw, history)
        from xtuner.v1.model.dense.concept_layer_dump import record

        record(getattr(self, "dump_name", None), "hidden", hidden_states)
        return hidden_states, raw


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
        encoder_concept_states: torch.Tensor,
        *history_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.transformer(hidden_states, position_embeddings, seq_ctx)
        history = (*history_states, raw)
        hidden_states = self.depth_dd(raw, history)
        hidden_states = self.read_encoder_route(hidden_states, encoder_concept_states)
        from xtuner.v1.model.dense.concept_layer_dump import record

        record(getattr(self, "dump_name", None), "hidden", hidden_states)
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
        repeated_concepts: torch.Tensor,
        encoder_states: torch.Tensor,
        concept_states: torch.Tensor,
        gate: torch.Tensor,
        *history_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.transformer(hidden_states, position_embeddings, seq_ctx)
        history = (*history_states, raw)
        hidden_states = self.decoder_dd(raw, history)
        hidden_states = self.concept_route(hidden_states, repeated_concepts, gate[0])
        hidden_states = self.read_encoder_route(hidden_states, encoder_states)
        hidden_states = self.read_concept_route(hidden_states, concept_states, gate[1])
        from xtuner.v1.model.dense.concept_layer_dump import record

        record(getattr(self, "dump_name", None), "hidden", hidden_states)
        return hidden_states, raw


class EncoderStageGroup(nn.Sequential):
    """A compiled/FSDP unit containing consecutive encoder stages."""

    def __init__(self, stages: list[tuple[int, EncoderStage]]):
        super().__init__(OrderedDict((str(index), stage) for index, stage in stages))

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        *history_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        raw_states: list[torch.Tensor] = []
        for stage in self:
            hidden_states, raw = stage(
                hidden_states,
                position_embeddings,
                seq_ctx,
                *history_states,
                *raw_states,
            )
            raw_states.append(raw)
        return hidden_states, *raw_states


class ConceptStageGroup(nn.Sequential):
    """A compiled/FSDP unit containing consecutive concept stages."""

    def __init__(self, stages: list[tuple[int, ConceptStage]]):
        super().__init__(OrderedDict((str(index), stage) for index, stage in stages))

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        encoder_concept_states: torch.Tensor,
        *history_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        raw_states: list[torch.Tensor] = []
        for stage in self:
            hidden_states, raw = stage(
                hidden_states,
                position_embeddings,
                seq_ctx,
                encoder_concept_states,
                *history_states,
                *raw_states,
            )
            raw_states.append(raw)
        return hidden_states, *raw_states


class DecoderStageGroup(nn.Sequential):
    """A compiled/FSDP unit containing consecutive decoder stages."""

    def __init__(self, stages: list[tuple[int, DecoderStage]]):
        super().__init__(OrderedDict((str(index), stage) for index, stage in stages))
        self.stage_indices = tuple(index for index, _ in stages)

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seq_ctx: SequenceContext,
        repeated_concepts: torch.Tensor,
        encoder_states: torch.Tensor,
        concept_states: torch.Tensor,
        gate_logits: torch.Tensor,
        *history_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        raw_states: list[torch.Tensor] = []
        for stage_idx, stage in zip(self.stage_indices, self, strict=True):
            gate = gate_logits[stage_idx].float().softmax(dim=-1)
            hidden_states, raw = stage(
                hidden_states,
                position_embeddings,
                seq_ctx,
                repeated_concepts,
                encoder_states,
                concept_states,
                gate,
                *history_states,
                *raw_states,
            )
            raw_states.append(raw)
        return hidden_states, *raw_states
