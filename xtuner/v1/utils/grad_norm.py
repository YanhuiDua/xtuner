from typing import List

import torch
from torch.distributed.tensor import DTensor

from .dtensor import cal_total_norm, group_tensors_by_device_mesh_and_placements


def cal_grad_norm_from_groups(grouped_grads, dtype=torch.float32):
    total_norms = []
    output_dtype = None
    for grads in grouped_grads.values():
        if not grads:
            continue
        total_norm = cal_total_norm(grads, norm_type=2.0, foreach=True, dtype=dtype)
        total_norms.append(total_norm)
        output_dtype = grads[0].dtype
    if not total_norms:
        raise ValueError("cal_grad_norm requires at least one gradient")
    assert output_dtype is not None
    grad_norm = torch.linalg.vector_norm(torch.stack(total_norms), ord=2.0, dtype=dtype)
    grad_norm = grad_norm.to(output_dtype)
    return grad_norm, grouped_grads


def cal_grad_norm(grads: List[DTensor], dtype=torch.float32):
    grouped_grads = group_tensors_by_device_mesh_and_placements(grads)
    return cal_grad_norm_from_groups(grouped_grads, dtype=dtype)
