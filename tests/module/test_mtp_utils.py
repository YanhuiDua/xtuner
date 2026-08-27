import pytest
import torch

from xtuner.v1.module.mtp.utils import roll_packed_tensor


@pytest.mark.parametrize(
    ("shifts", "expected"),
    [
        (-1, [[2, 3, -100, 5, 6, -100]]),
        (-2, [[3, -100, -100, 6, -100, -100]]),
    ],
)
def test_roll_packed_tensor_respects_sequence_boundaries(shifts, expected) -> None:
    tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
    cu_seq_lens = torch.tensor([0, 3, 6], dtype=torch.int32)

    rolled = roll_packed_tensor(tensor, cu_seq_lens, shifts=shifts, fill_value=-100)

    torch.testing.assert_close(rolled, torch.tensor(expected))


def test_roll_packed_tensor_handles_short_and_empty_sequences() -> None:
    tensor = torch.tensor([[1, 2, 3]])
    cu_seq_lens = torch.tensor([0, 1, 1, 3], dtype=torch.int32)

    rolled = roll_packed_tensor(tensor, cu_seq_lens, shifts=-2, fill_value=0)

    torch.testing.assert_close(rolled, torch.zeros_like(tensor))


def test_roll_packed_tensor_broadcasts_boundary_mask() -> None:
    tensor = torch.arange(12).reshape(1, 6, 2)
    cu_seq_lens = torch.tensor([0, 3, 6], dtype=torch.int32)

    rolled = roll_packed_tensor(tensor, cu_seq_lens, shifts=-1, dim=-2)

    expected = torch.tensor([[[2, 3], [4, 5], [0, 0], [8, 9], [10, 11], [0, 0]]])
    torch.testing.assert_close(rolled, expected)
