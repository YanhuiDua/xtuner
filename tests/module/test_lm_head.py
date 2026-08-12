import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.loss.ce_loss import CELossConfig, LMHeadLossContext
from xtuner.v1.module import LMHead


@pytest.mark.parametrize("mode", ["eager", "chunk"])
def test_padded_lm_head_preserves_logical_loss_and_gradients(mode: str):
    torch.manual_seed(7)
    hidden_size = 4
    logical_vocab_size = 6
    physical_vocab_size = 8
    hidden = torch.randn(1, 5, hidden_size, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    head = LMHead(
        hidden_size,
        physical_vocab_size,
        bias=False,
        logical_out_features=logical_vocab_size,
    )
    reference_weight = head.weight[:logical_vocab_size].detach().clone().requires_grad_()
    with torch.no_grad():
        head.weight[logical_vocab_size:].fill_(1000.0)

    labels = torch.tensor([[0, 5, -100, 2, 3]])
    loss_cfg = CELossConfig(mode=mode, chunk_size=2)
    loss_ctx = loss_cfg.build({"shifted_labels": labels})
    assert loss_ctx is not None
    [loss_ctx] = LMHeadLossContext.build_batches([loss_ctx])

    loss, (logits, _) = head(hidden, loss_ctx)
    reference_logits = F.linear(reference_hidden, reference_weight).float()
    reference_loss = F.cross_entropy(
        reference_logits.flatten(0, 1),
        labels.flatten(),
        reduction="none",
        ignore_index=-100,
    )
    assert loss_ctx.loss_kwargs.loss_weight is not None
    reference_loss = (reference_loss * loss_ctx.loss_kwargs.loss_weight.flatten()).sum()

    torch.testing.assert_close(loss, reference_loss)
    if mode == "eager":
        assert logits is not None
        torch.testing.assert_close(logits, reference_logits.flatten(0, 1))
    else:
        assert logits is None

    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(head.weight.grad[:logical_vocab_size], reference_weight.grad)
    torch.testing.assert_close(
        head.weight.grad[logical_vocab_size:],
        torch.zeros_like(head.weight.grad[logical_vocab_size:]),
    )


def test_padded_lm_head_returns_logical_logits_without_loss():
    head = LMHead(4, 8, bias=False, logical_out_features=6)
    hidden = torch.randn(1, 3, 4)
    _, (logits, _) = head(hidden)
    assert logits is not None
    assert logits.shape == (1, 3, 6)
