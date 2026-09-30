# Copyright (c) OpenMMLab. All rights reserved.
"""Shared DTensor materialization helpers for FLA modules under FSDP."""

from __future__ import annotations

import torch
from einops import rearrange
from torch.distributed.tensor import DTensor


def dtensor_to_local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def dtensor_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def materialize_depthwise_conv1d_weight_bias(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gather FSDP weights and rearrange Conv1d ``(D, 1, W)`` → ``(D, W)`` for
    FLA kernels."""
    w = dtensor_full_tensor(weight)
    b = None if bias is None else dtensor_full_tensor(bias)
    return rearrange(w, "d 1 w -> d w"), b
