import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.utils.dtensor import cal_total_norm
from xtuner.v1.utils.grad_norm import cal_grad_norm_from_groups


def test_cal_grad_norm_requires_a_gradient():
    with pytest.raises(ValueError, match="at least one gradient"):
        cal_grad_norm_from_groups({})


class TestDTensorUtils(DeterministicDDPTestCase):
    def test_cal_total_norm_matches_global_sharded_norm(self):
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("fsdp",))

        values = torch.arange(1, self.world_size * 8 + 1, device="cuda", dtype=torch.bfloat16)
        other_values = values.reshape(-1, 2) * 0.25
        tensors = [
            distribute_tensor(values, mesh, (Shard(0),)),
            distribute_tensor(other_values, mesh, (Shard(0),)),
        ]

        foreach_inputs: list[list[torch.Tensor]] = []
        original_foreach_norm = torch._foreach_norm

        def record_foreach_norm(inputs, *args, **kwargs):
            foreach_inputs.append(list(inputs))
            return original_foreach_norm(inputs, *args, **kwargs)

        torch._foreach_norm = record_foreach_norm
        try:
            actual = cal_total_norm(tensors, foreach=True, dtype=torch.float32)
        finally:
            torch._foreach_norm = original_foreach_norm

        expected = torch.linalg.vector_norm(
            torch.cat((values.flatten(), other_values.flatten())),
            dtype=torch.float32,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(foreach_inputs), 1)
        self.assertTrue(all(not isinstance(tensor, DTensor) for tensor in foreach_inputs[0]))

        local_before = [tensor.to_local().clone() for tensor in tensors]
        torch._foreach_mul_([tensor.to_local() for tensor in tensors], 0.5)
        for tensor, before in zip(tensors, local_before):
            torch.testing.assert_close(tensor.to_local(), before * 0.5)
