import torch
from torch import nn

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.module.decoder_layer.moe_decoder_layer import MoEDecoderLayerOutput
from xtuner.v1.module.mtp import MTPLayer


def _build_seq_ctx() -> SequenceContext:
    input_ids = torch.tensor([[1, 2, 3]])
    return SequenceContext.from_input_ids((input_ids,), device="cpu")


def _build_position_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 3, 2), torch.zeros(1, 3, 2)


class _DenseDecoder(nn.Module):
    def forward(self, hidden_states, *, position_embeddings, seq_ctx):
        if isinstance(hidden_states, list):
            return tuple(hidden_state + 1 for hidden_state in hidden_states)
        return hidden_states + 1


class _MoEDecoder(nn.Module):
    def forward(self, hidden_states, *, position_embeddings, seq_ctx) -> MoEDecoderLayerOutput:
        return {
            "hidden_states": hidden_states + 1,
            "router_logits": torch.ones(1),
            "router_weights": torch.full((1,), 2.0),
            "router_topk_ids": torch.zeros(1, dtype=torch.long),
        }


def _build_mtp_layer(decoder_layer: nn.Module) -> MTPLayer:
    return MTPLayer(
        hidden_size=4,
        rms_norm_eps=1e-6,
        rms_norm_type="default",
        decoder_layer=decoder_layer,
    )


def test_mtp_layer_normalizes_dense_tensor_and_micro_batch_outputs() -> None:
    layer = _build_mtp_layer(_DenseDecoder())
    hidden_states = torch.randn(1, 3, 4)
    future_embeddings = torch.randn_like(hidden_states)
    position_embeddings = _build_position_embeddings()
    seq_ctx = _build_seq_ctx()

    single_output = layer(
        hidden_states,
        future_embeddings=future_embeddings,
        position_embeddings=position_embeddings,
        seq_ctx=seq_ctx,
    )
    assert single_output["hidden_states"].shape == hidden_states.shape
    assert single_output["router_logits"].numel() == 0

    multi_output = layer(
        [hidden_states, hidden_states],
        future_embeddings=[future_embeddings, future_embeddings],
        position_embeddings=[position_embeddings, position_embeddings],
        seq_ctx=[seq_ctx, seq_ctx],
    )
    assert len(multi_output["hidden_states"]) == 2
    assert len(multi_output["router_logits"]) == 2
    assert all(router_logits.numel() == 0 for router_logits in multi_output["router_logits"])


def test_mtp_layer_preserves_moe_router_outputs() -> None:
    layer = _build_mtp_layer(_MoEDecoder())
    hidden_states = torch.randn(1, 3, 4)
    output = layer(
        hidden_states,
        future_embeddings=torch.randn_like(hidden_states),
        position_embeddings=_build_position_embeddings(),
        seq_ctx=_build_seq_ctx(),
    )

    torch.testing.assert_close(output["router_logits"], torch.ones(1))
    torch.testing.assert_close(output["router_weights"], torch.full((1,), 2.0))
    assert output["router_topk_ids"].dtype == torch.long
