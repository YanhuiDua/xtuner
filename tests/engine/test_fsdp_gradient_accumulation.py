from unittest.mock import patch

import torch

from xtuner.v1.config import FSDPConfig
from xtuner.v1.engine.train_engine import TrainEngine


class _Output:
    def free_nongrad_feature(self) -> None:
        pass


class _Model:
    def __init__(self) -> None:
        self.events: list[tuple[str, bool | int]] = []
        self.forward_count = 0

    def pre_micro_batch_forward(self, _data_batches):
        return {}

    def post_micro_batch_forward(self, _micro_batch_results):
        return {}

    def set_requires_gradient_sync(self, enabled: bool) -> None:
        self.events.append(("gradient_sync", enabled))

    def set_reshard_after_backward(self, enabled: bool) -> None:
        self.events.append(("reshard_after_backward", enabled))

    def __call__(self, *, seq_ctx, loss_ctx):
        self.events.append(("forward", self.forward_count))
        self.forward_count += 1
        return _Output()


def _build_engine(no_sync: bool) -> TrainEngine:
    engine = object.__new__(TrainEngine)
    engine.fsdp_cfg = FSDPConfig(gradient_accumulation_no_sync=no_sync)
    engine.intra_layer_micro_batch = 1
    engine._count = 0
    engine.model = _Model()
    engine._maybe_precompute_float8_dynamic_scale_for_fsdp = lambda: None
    engine._get_total_loss = lambda _output: torch.tensor(1.0, requires_grad=True)
    return engine


def test_fsdp2_no_sync_only_synchronizes_final_micro_batch() -> None:
    engine = _build_engine(no_sync=True)
    batches = [{"seq_ctx": i, "loss_ctx": i} for i in range(3)]

    with patch("xtuner.v1.engine.train_engine.DEVICE", torch.device("cpu")):
        engine.train_step(batches)

    assert engine.model.events == [
        ("gradient_sync", False),
        ("reshard_after_backward", False),
        ("forward", 0),
        ("forward", 1),
        ("gradient_sync", True),
        ("reshard_after_backward", True),
        ("forward", 2),
        ("gradient_sync", True),
        ("reshard_after_backward", True),
    ]


def test_fsdp2_no_sync_is_inactive_for_single_micro_batch() -> None:
    engine = _build_engine(no_sync=True)

    with patch("xtuner.v1.engine.train_engine.DEVICE", torch.device("cpu")):
        engine.train_step([{"seq_ctx": 0, "loss_ctx": 0}])

    assert engine.model.events == [("forward", 0)]
