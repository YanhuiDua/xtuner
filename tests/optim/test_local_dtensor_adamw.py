import copy
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.config import AdamWConfig
from xtuner.v1.engine import TrainEngine
from xtuner.v1.optim import LocalDTensorAdamW


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class _ParameterModel(nn.Module):
    def __init__(self, param: nn.Parameter) -> None:
        super().__init__()
        self.weight = param

    def trainable_parameters(self):
        return self.named_parameters()

    def scale_and_reduce_grad(self) -> None:
        pass


def _deferred_fused_step(
    optimizer: torch.optim.Optimizer,
    *,
    grad_norm: torch.Tensor,
    grad_scale: torch.Tensor,
    skip_threshold: float | None,
) -> None:
    """Exercise TrainEngine's deferred clipping/skip handoff in isolation."""
    engine = object.__new__(TrainEngine)
    engine.optimizer = optimizer
    engine.optim_cfg = SimpleNamespace(skip_grad_norm_threshold=skip_threshold)
    engine._defer_grad_clip_to_optimizer = True
    engine._optimizer_grad_scale = grad_scale
    engine.step_optimizer(grad_norm)


def test_fused_grad_clip_rejects_resumed_nonfused_groups():
    param = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([param], fused=True)
    engine = object.__new__(TrainEngine)
    engine.optimizer = optimizer
    engine.optim_cfg = SimpleNamespace(fused_grad_clip=True)

    engine._validate_fused_grad_clip_optimizer()
    optimizer.param_groups[0]["fused"] = False
    with pytest.raises(ValueError, match="fused=True for every optimizer parameter group"):
        engine._validate_fused_grad_clip_optimizer()


def test_local_dtensor_adamw_rejects_nonfused_state_dict():
    param = nn.Parameter(torch.ones(2))
    optimizer = LocalDTensorAdamW([param], fused=True)
    state_dict = optimizer.state_dict()
    state_dict["param_groups"][0]["fused"] = False

    with pytest.raises(ValueError, match="requires fused=True"):
        optimizer.load_state_dict(state_dict)
    assert optimizer.param_groups[0]["fused"] is True


def test_adamw_config_rejects_swap_with_local_dtensor_step():
    model = _ParameterModel(nn.Parameter(torch.ones(2)))
    config = AdamWConfig(fused=True, swap_optimizer=True, local_dtensor_step=True)

    with pytest.raises(ValueError, match="swap_optimizer and local_dtensor_step are mutually exclusive"):
        config.build(model)


class TestFusedGradClipCPU(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 1

    def _assert_state_equal(
        self,
        actual: LocalDTensorAdamW,
        actual_param: nn.Parameter,
        expected: torch.optim.Optimizer,
        expected_param: nn.Parameter,
    ) -> None:
        self.assertEqual(set(actual.state[actual_param]), set(expected.state[expected_param]))
        for key, expected_value in expected.state[expected_param].items():
            torch.testing.assert_close(_local(actual.state[actual_param][key]), expected_value, rtol=0, atol=0)

    def test_train_engine_fused_grad_clip_and_skip(self):
        self.create_pg("cpu")
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        initial = torch.tensor([1.0, 2.0])
        param = nn.Parameter(distribute_tensor(initial.clone(), mesh, (Shard(0),)))
        ref_param = nn.Parameter(initial.clone())
        model = _ParameterModel(param)
        optimizer = LocalDTensorAdamW(
            [param],
            lr=1e-3,
            betas=(0.9, 0.95),
            fused=True,
            amsgrad=True,
        )
        ref_optimizer = torch.optim.AdamW(
            [ref_param],
            lr=1e-3,
            betas=(0.9, 0.95),
            fused=True,
            amsgrad=True,
        )

        engine = object.__new__(TrainEngine)
        engine.model = model
        engine.optimizer = optimizer
        engine.optim_cfg = SimpleNamespace(
            fused_grad_clip=True,
            max_grad_norm=1.0,
            skip_grad_norm_threshold=None,
        )
        engine._defer_grad_clip_to_optimizer = False
        engine._optimizer_grad_scale = None
        engine._grad_norm_param_groups = {(param.device_mesh, param.placements): [param]}

        # norm=5 => coefficient < 1. The deferred path keeps the DTensor grad
        # untouched and passes the reciprocal coefficient to fused AdamW.
        grad = torch.tensor([3.0, 4.0])
        param.grad = distribute_tensor(grad.clone(), mesh, (Shard(0),))
        ref_param.grad = grad.clone()
        grad_norm = engine.clip_grad_norm()
        clip_coef = torch.clamp(engine.optim_cfg.max_grad_norm / (grad_norm + 1e-6), max=1.0)
        self.assertAlmostEqual(grad_norm.item(), 5.0)
        self.assertLess(clip_coef.item(), 1.0)
        torch.testing.assert_close(_local(param.grad), grad, rtol=0, atol=0)
        torch.testing.assert_close(engine._optimizer_grad_scale, clip_coef.reciprocal(), rtol=0, atol=0)
        ref_param.grad.mul_(clip_coef)
        engine.step_optimizer(grad_norm)
        ref_optimizer.step()
        torch.testing.assert_close(_local(param), ref_param, rtol=0, atol=0)
        self._assert_state_equal(optimizer, param, ref_optimizer, ref_param)
        self.assertEqual(_local(optimizer.state[param]["step"]).item(), 1)

        # norm=0.5 => coefficient == 1. This must behave exactly like an
        # unclipped fused optimizer step.
        grad = torch.tensor([-0.3, 0.4])
        param.grad = distribute_tensor(grad.clone(), mesh, (Shard(0),))
        ref_param.grad = grad.clone()
        grad_norm = engine.clip_grad_norm()
        self.assertAlmostEqual(grad_norm.item(), 0.5)
        torch.testing.assert_close(engine._optimizer_grad_scale, torch.tensor(1.0), rtol=0, atol=0)
        engine.step_optimizer(grad_norm)
        ref_optimizer.step()
        torch.testing.assert_close(_local(param), ref_param, rtol=0, atol=0)
        self._assert_state_equal(optimizer, param, ref_optimizer, ref_param)
        self.assertEqual(_local(optimizer.state[param]["step"]).item(), 2)

        # A finite norm above the configured skip threshold is communicated as
        # found_inf to the fused kernel. It must not mutate any optimizer state
        # (including the AMSGrad maximum and step counter) or the parameter.
        engine.optim_cfg.skip_grad_norm_threshold = 0.25
        param_before = _local(param).clone()
        state_before = {key: _local(value).clone() for key, value in optimizer.state[param].items()}
        param.grad = distribute_tensor(grad.clone(), mesh, (Shard(0),))
        grad_norm = engine.clip_grad_norm()
        engine.step_optimizer(grad_norm)
        torch.testing.assert_close(_local(param), param_before, rtol=0, atol=0)
        for key, expected in state_before.items():
            torch.testing.assert_close(_local(optimizer.state[param][key]), expected, rtol=0, atol=0)
        self.assertEqual(_local(optimizer.state[param]["step"]).item(), 2)
        self.assertIsNone(param.grad)
        self.assertFalse(hasattr(optimizer, "grad_scale"))
        self.assertFalse(hasattr(optimizer, "found_inf"))

        # Non-finite gradients take the same device-side found_inf path even
        # when no explicit skip threshold is configured.
        engine.optim_cfg.skip_grad_norm_threshold = None
        param_before = _local(param).clone()
        state_before = {key: _local(value).clone() for key, value in optimizer.state[param].items()}
        param.grad = distribute_tensor(torch.tensor([float("nan"), 1.0]), mesh, (Shard(0),))
        grad_norm = engine.clip_grad_norm()
        self.assertTrue(torch.isnan(grad_norm))
        engine.step_optimizer(grad_norm)
        torch.testing.assert_close(_local(param), param_before, rtol=0, atol=0)
        for key, expected in state_before.items():
            torch.testing.assert_close(_local(optimizer.state[param][key]), expected, rtol=0, atol=0)
        self.assertEqual(_local(optimizer.state[param]["step"]).item(), 2)

    def test_train_engine_fused_grad_clip_with_torch_adamw(self):
        self.create_pg("cpu")
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        param = nn.Parameter(distribute_tensor(torch.tensor([1.0, 2.0]), mesh, (Shard(0),)))
        model = _ParameterModel(param)
        optimizer = torch.optim.AdamW([param], lr=1e-3, fused=True)

        engine = object.__new__(TrainEngine)
        engine.model = model
        engine.optimizer = optimizer
        engine.optim_cfg = SimpleNamespace(
            fused_grad_clip=True,
            max_grad_norm=1.0,
            skip_grad_norm_threshold=None,
        )
        engine._defer_grad_clip_to_optimizer = False
        engine._optimizer_grad_scale = None
        engine._grad_norm_param_groups = {(param.device_mesh, param.placements): [param]}

        param.grad = distribute_tensor(torch.tensor([3.0, 4.0]), mesh, (Shard(0),))
        grad_norm = engine.clip_grad_norm()
        self.assertAlmostEqual(grad_norm.item(), 5.0)
        self.assertTrue(engine._defer_grad_clip_to_optimizer)
        engine.step_optimizer(grad_norm)

        self.assertEqual(_local(optimizer.state[param]["step"]).item(), 1)
        self.assertIsNone(param.grad)
        self.assertFalse(hasattr(optimizer, "grad_scale"))
        self.assertFalse(hasattr(optimizer, "found_inf"))


class TestLocalDTensorAdamW(DeterministicDDPTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def _assert_optimizer_state_equal(
        self,
        actual: LocalDTensorAdamW,
        actual_param: nn.Parameter,
        expected: torch.optim.Optimizer,
        expected_param: nn.Parameter,
    ) -> None:
        actual_state = actual.state[actual_param]
        expected_state = expected.state[expected_param]
        self.assertEqual(set(actual_state), set(expected_state))
        for key in expected_state:
            torch.testing.assert_close(_local(actual_state[key]), _local(expected_state[key]), rtol=0, atol=0)

    def test_matches_torch_and_roundtrips_optimizer_state(self):
        self.create_pg("cuda")
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("fsdp",))
        values = torch.linspace(-1, 1, self.world_size * 16, device="cuda", dtype=torch.bfloat16)

        ref_param = nn.Parameter(distribute_tensor(values.clone(), mesh, (Shard(0),)))
        local_param = nn.Parameter(distribute_tensor(values.clone(), mesh, (Shard(0),)))
        ref_optim = torch.optim.AdamW([ref_param], lr=1e-3, fused=True, amsgrad=True)
        local_optim = LocalDTensorAdamW([local_param], lr=1e-3, fused=True, amsgrad=True)

        grad = torch.linspace(0.5, -0.5, values.numel(), device="cuda", dtype=torch.bfloat16)
        for param in (ref_param, local_param):
            param.grad = distribute_tensor(grad.clone(), mesh, (Shard(0),))
        for optimizer in (ref_optim, local_optim):
            optimizer.grad_scale = torch.tensor(2.0, device="cuda")  # type: ignore[attr-defined]
            optimizer.found_inf = torch.tensor(0.0, device="cuda")  # type: ignore[attr-defined]
            optimizer.step()

        torch.testing.assert_close(_local(local_param), _local(ref_param), rtol=0, atol=0)
        self._assert_optimizer_state_equal(local_optim, local_param, ref_optim, ref_param)
        self.assertIn("max_exp_avg_sq", local_optim.state[local_param])
        self.assertEqual(_local(local_optim.state[local_param]["step"]).item(), 1)

        # A regular optimizer state_dict roundtrip must preserve DTensor state
        # layout and AMSGrad buffers, then produce the same next update.
        roundtrip_param = nn.Parameter(local_param.detach().clone())
        roundtrip_optim = LocalDTensorAdamW([roundtrip_param], lr=1e-3, fused=True, amsgrad=True)
        roundtrip_optim.load_state_dict(copy.deepcopy(local_optim.state_dict()))
        self._assert_optimizer_state_equal(roundtrip_optim, roundtrip_param, local_optim, local_param)
        self.assertIsInstance(roundtrip_optim.state[roundtrip_param]["exp_avg"], DTensor)

        next_grad = distribute_tensor(grad.flip(0).clone(), mesh, (Shard(0),))
        for param in (local_param, roundtrip_param):
            param.grad = next_grad.clone()
        for optimizer in (local_optim, roundtrip_optim):
            optimizer.grad_scale = torch.tensor(4.0, device="cuda")  # type: ignore[attr-defined]
            optimizer.found_inf = torch.tensor(0.0, device="cuda")  # type: ignore[attr-defined]
            optimizer.step()
        torch.testing.assert_close(_local(roundtrip_param), _local(local_param), rtol=0, atol=0)
        self._assert_optimizer_state_equal(roundtrip_optim, roundtrip_param, local_optim, local_param)
        self.assertEqual(_local(local_optim.state[local_param]["step"]).item(), 2)

        # Both the skip-threshold path and the fused found_inf handoff must
        # leave parameters, AMSGrad buffers, and the step counter untouched.
        param_before = _local(local_param).clone()
        state_before = {key: _local(value).clone() for key, value in local_optim.state[local_param].items()}
        local_param.grad = distribute_tensor(grad.clone(), mesh, (Shard(0),))
        _deferred_fused_step(
            local_optim,
            grad_norm=torch.tensor(2.0, device="cuda"),
            grad_scale=torch.tensor(1.0, device="cuda"),
            skip_threshold=1.0,
        )
        torch.testing.assert_close(_local(local_param), param_before, rtol=0, atol=0)
        for key, before in state_before.items():
            torch.testing.assert_close(_local(local_optim.state[local_param][key]), before, rtol=0, atol=0)
        self.assertFalse(hasattr(local_optim, "grad_scale"))
        self.assertFalse(hasattr(local_optim, "found_inf"))

        # Exercise the same optimizer-state format through the DCP APIs used by
        # TrainEngine. Corrupting the loaded group proves resume re-validates
        # fused=True before a deferred-clipping step can run.
        source_model = _ParameterModel(local_param)
        options = StateDictOptions(cpu_offload=True)
        checkpoint_path: list[str | None] = [
            tempfile.mkdtemp(prefix="xtuner-local-dtensor-adamw-") if dist.get_rank() == 0 else None
        ]
        dist.broadcast_object_list(checkpoint_path, src=0)
        dcp_path = Path(checkpoint_path[0])  # type: ignore[arg-type]

        save_state = {"optimizer": get_optimizer_state_dict(source_model, local_optim, options=options)}
        dcp.save(save_state, checkpoint_id=dcp_path)

        resumed_param = nn.Parameter(local_param.detach().clone())
        resumed_model = _ParameterModel(resumed_param)
        resumed_optim = LocalDTensorAdamW([resumed_param], lr=1e-3, fused=True, amsgrad=True)
        load_state = {"optimizer": get_optimizer_state_dict(resumed_model, resumed_optim, options=options)}
        dcp.load(load_state, checkpoint_id=dcp_path)

        loaded_group = load_state["optimizer"]["param_groups"][0]
        self.assertIs(loaded_group["fused"], True)
        loaded_group["fused"] = False
        with self.assertRaisesRegex(ValueError, "requires fused=True"):
            set_optimizer_state_dict(resumed_model, resumed_optim, load_state["optimizer"], options=options)
        loaded_group["fused"] = True
        set_optimizer_state_dict(resumed_model, resumed_optim, load_state["optimizer"], options=options)

        self._assert_optimizer_state_equal(resumed_optim, resumed_param, local_optim, local_param)
        self.assertIsInstance(resumed_optim.state[resumed_param]["exp_avg"], DTensor)
        self.assertIn("max_exp_avg_sq", resumed_optim.state[resumed_param])

        resume_grad = distribute_tensor(grad.square(), mesh, (Shard(0),))
        for param in (local_param, resumed_param):
            param.grad = resume_grad.clone()
        for optimizer in (local_optim, resumed_optim):
            optimizer.grad_scale = torch.tensor(2.0, device="cuda")  # type: ignore[attr-defined]
            optimizer.found_inf = torch.tensor(0.0, device="cuda")  # type: ignore[attr-defined]
            optimizer.step()
        torch.testing.assert_close(_local(resumed_param), _local(local_param), rtol=0, atol=0)
        self._assert_optimizer_state_equal(resumed_optim, resumed_param, local_optim, local_param)
        self.assertEqual(_local(resumed_optim.state[resumed_param]["step"]).item(), 3)

        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(dcp_path)
