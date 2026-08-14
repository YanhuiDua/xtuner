from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _has_foreach_support,
)


def group_tensors_by_device_mesh_and_placements(
    tensors: list[DTensor],
) -> dict[tuple[DeviceMesh, tuple[Placement, ...]], list[DTensor]]:
    """Group DTensors by their device_mesh and placements.

    Args:
        tensors (list[DTensor]): List of DTensors to group.

    Returns:
        dict[tuple[DeviceMesh, tuple[Placement, ...]], list[DTensor]]:
            A dictionary mapping (device_mesh, placements) to a list of DTensors.
    """
    grouped_tensors: dict[tuple[DeviceMesh, tuple[Placement, ...]], list[DTensor]] = {}
    for tensor in tensors:
        assert isinstance(tensor, DTensor)
        key = (tensor.device_mesh, tensor.placements)
        if key in grouped_tensors:
            grouped_tensors[key].append(tensor)
        else:
            grouped_tensors[key] = [tensor]
    return grouped_tensors


def cal_total_norm(
    tensors: list[DTensor], norm_type: float = 2.0, foreach: bool | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Compute the total norm of a list of DTensors.

    All tensors must share the same device_mesh and placements. Supports L2 norm with
    distributed all-reduce across sharded mesh dimensions.

    Args:
        tensors (list[DTensor]): List of DTensors to compute the norm of.
        norm_type (float): Type of the norm. Only 2.0 is supported.
        foreach (bool | None): Whether to use the foreach API. None for auto-detection.
        dtype (torch.dtype): Dtype for norm computation.

    Returns:
        torch.Tensor: The total norm as a scalar tensor.
    """
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)

    device_mesh: DeviceMesh = tensors[
        0
    ].device_mesh  # For eg: DeviceMesh('cuda', [0, 1], mesh_dim_names=('default.fsdp',))
    placements = tensors[0].placements  # For eg: (Shard(dim=0),)
    device = tensors[0].device  # For eg: device(type='cuda', index=0)
    local_tensors = [tensor.to_local() for tensor in tensors]
    norms: tuple[torch.Tensor, ...]
    if (foreach is None and _has_foreach_support(local_tensors, device)) or (
        foreach and _device_has_foreach_support(device)
    ):
        # Run foreach directly on the local shards. DTensor dispatch eventually
        # executes the same local foreach kernel, but adds operator propagation
        # and wraps every scalar result back into a DTensor first.
        norms = torch._foreach_norm(local_tensors, norm_type, dtype=dtype)  # type: ignore[assignment]
    elif foreach:
        raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
    else:
        norms = tuple(torch.linalg.vector_norm(g, norm_type, dtype=dtype) for g in local_tensors)

    local_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type, dtype=dtype)
    if norm_type == 2:
        local_norm_squared = local_norm**2
        for i, placement in enumerate(placements):
            if isinstance(placement, Shard):
                # FSDP's strided bookkeeping placement is a Shard subclass, so
                # RuntimeLayout owns the only concrete private-type dependency.
                dist.all_reduce(local_norm_squared, group=device_mesh.get_group(i))
            elif isinstance(placement, Replicate):
                pass
            else:
                raise ValueError(f"Unsupported placement type {placement} in clip_grad_norm")
        global_norm = local_norm_squared**0.5
    else:
        raise NotImplementedError
    return global_norm


def is_evenly_distributed(dtensor: DTensor) -> bool:
    """Check if a DTensor is evenly distributed across the device mesh."""
    global_shape = dtensor.shape

    mesh = dtensor.device_mesh
    placements = dtensor.placements

    tensor_dim_to_mesh_dims: dict[int, list[int]] = {}

    for dim_idx, placement in enumerate(placements):
        if hasattr(placement, "is_shard") and placement.is_shard():
            mesh_dim = cast(Shard, placement).dim
            if dim_idx not in tensor_dim_to_mesh_dims:
                tensor_dim_to_mesh_dims[dim_idx] = []
            tensor_dim_to_mesh_dims[dim_idx].append(mesh_dim)

    for tensor_dim, mesh_dims in tensor_dim_to_mesh_dims.items():
        total_devices = 1
        for mesh_dim in mesh_dims:
            total_devices *= mesh.size(mesh_dim)

        if global_shape[tensor_dim] % total_devices != 0:
            return False

    return True
