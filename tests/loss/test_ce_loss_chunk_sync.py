import torch
import torch.nn.functional as F

from xtuner.v1.loss.ce_loss import CELossConfig, CELossContext


def test_chunk_loss_handles_an_all_ignored_chunk_without_host_branch() -> None:
    config = CELossConfig(mode="chunk", chunk_size=2, loss_reduction="token")
    labels = torch.tensor([[-100, -100, 1, 2]])
    cu_seq_lens = torch.tensor([0, 4], dtype=torch.int32)
    loss_context = config.build({"shifted_labels": labels})
    assert loss_context is not None
    loss_context = CELossContext.build_batches([loss_context], cu_seq_lens_list=[cu_seq_lens])[0]

    hidden_states = torch.randn(1, 4, 3, requires_grad=True)
    head_weight = torch.randn(5, 3, requires_grad=True)
    loss, _ = loss_context(hidden_states, head_weight)

    logits = F.linear(hidden_states, head_weight).float().flatten(0, 1)
    reference = F.cross_entropy(logits, labels.flatten(), reduction="none", ignore_index=-100).sum() / 2
    torch.testing.assert_close(loss, reference)

    loss.backward()
    assert torch.isfinite(hidden_states.grad).all()
    assert torch.isfinite(head_weight.grad).all()
