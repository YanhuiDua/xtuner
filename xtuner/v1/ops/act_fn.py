from functools import partial

import torch
from torch.nn import functional as F


def native_swiglu(fused_x: torch.Tensor, split_dim=-1) -> torch.Tensor:
    x1, x2 = torch.chunk(fused_x, 2, dim=split_dim)
    return F.silu(x1) * x2


def npu_swiglu(fused_x: torch.Tensor, split_dim: int = -1) -> torch.Tensor:
    import torch_npu

    return torch_npu.npu_swiglu(fused_x, dim=split_dim)


def native_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    gated_output = (up + 1) * glu
    return gated_output


def native_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    if approximate is not None:
        return F.gelu(x, approximate=approximate)
    return F.gelu(x)


def npu_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    import torch_npu

    if approximate is not None:
        return torch_npu.npu_gelu(x, approximate=approximate)
    return torch_npu.npu_gelu(x)


def npu_clipped_swiglu(fused_x: torch.Tensor, split_dim=-1, alpha=1.702, limit=7) -> torch.Tensor:
    raise NotImplementedError


def native_silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


def situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """SiTU-GLU on separate gate/up tensors (no cat/chunk)."""
    dtype = gate.dtype
    gate_f = gate.float()
    up_f = up.float()
    situ_a = beta * torch.tanh(gate_f / beta) * torch.sigmoid(gate_f)
    if linear_beta is not None:
        up_f = linear_beta * torch.tanh(up_f / linear_beta)
    return (situ_a * up_f).to(dtype)


def silu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU-style combine with unary SiLU on gate."""
    return F.silu(gate) * up


def native_situ(
    fused_x: torch.Tensor,
    split_dim: int = -1,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> torch.Tensor:
    """SiTU-GLU for fused ``[gate, up]`` (MoE grouped-gemm path)."""
    gate, up = torch.chunk(fused_x, 2, dim=split_dim)
    return situ_glu(gate, up, beta=beta, linear_beta=linear_beta)


def get_mlp_act_fn(
    hidden_act: str,
    *,
    situ_beta: float = 1.0,
    situ_linear_beta: float | None = None,
):
    """Return ``(gate, up) -> Tensor`` for dense / shared-expert MLPs."""
    if hidden_act == "situ":
        return partial(situ_glu, beta=situ_beta, linear_beta=situ_linear_beta)
    if hidden_act == "silu":
        return silu_glu
    unary = get_act_fn(hidden_act)
    return lambda gate, up, _unary=unary: _unary(gate) * up


act_fn_type_map_cuda = {
    "swiglu": native_swiglu,
    "clipped_swiglu": native_clipped_swiglu,
    "gelu": native_gelu,
    "gelu_pytorch_tanh": partial(native_gelu, approximate="tanh"),
    "silu": native_silu,
    "situ": native_situ,
}
act_fn_type_map_npu = {
    "swiglu": npu_swiglu,
    "clipped_swiglu": npu_clipped_swiglu,
    "gelu": npu_gelu,
    "gelu_pytorch_tanh": partial(npu_gelu, approximate="tanh"),
    "silu": native_silu,
    "situ": native_situ,
}


def get_act_fn(act_type):
    from xtuner.v1.utils.device import get_device

    device = get_device()
    if device == "npu":
        return act_fn_type_map_npu[act_type]
    else:
        return act_fn_type_map_cuda[act_type]
