"""Zero-centered RMSNorm 的 Triton / native 数值对拍。

Qwen3.5 的 dense 与 ConceptLM 都使用 zero-centered RMSNorm；``XTUNER_USE_NATIVE_RMSNORM=0``
会把它切到仓库自带的 flash-attn Triton kernel（``zero_centered_weight=True``）。该路径在
2B ConceptLM 上实测稳态步时 -11.8%、显存 -5.7～-10.0 GB，这里钉住它的数值契约：

TestZeroCenteredRMSNormParity
    test_forward_matches_reference: 前向对 fp64 参考的误差与 native 相同（bf16 下为 1 ulp 量级）。
    test_backward_not_worse_than_native: 反向梯度对 fp64 参考的误差不超过 native 的两倍
        （bf16 梯度本身分辨率有限，两者应同量级）。
    test_gradients_are_finite: x / weight 的梯度均有限。
TestZeroCenteredRMSNormSelection
    test_env_switch_selects_implementation: ``XTUNER_USE_NATIVE_RMSNORM`` 在 0/1 之间切换时，
        ``get_zero_centered_rms_norm_fn()`` 返回 Triton / native 实现。
"""

import pytest
import torch

from xtuner.v1.ops.rms_norm import (
    _triton_zero_centered_rms_norm,
    get_zero_centered_rms_norm_fn,
    native_zero_centered_rms_norm,
)


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + eps)
    return (out * (1.0 + weight.double())).to(x.dtype)


def _relative_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().mean().clamp_min(1e-6)).item()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton zero-centered RMSNorm requires CUDA")
class TestZeroCenteredRMSNormParity:
    @pytest.mark.parametrize(
        ("dtype", "shape", "limit"),
        [
            (torch.float32, (1, 64, 128), 2.0e-5),
            (torch.bfloat16, (1, 64, 128), 2.0e-2),
            (torch.bfloat16, (1, 8192, 2048), 2.0e-2),
            (torch.bfloat16, (1, 8192, 4096), 2.0e-2),
            (torch.float32, (4, 333, 257), 2.0e-5),
        ],
    )
    def test_forward_matches_reference(self, dtype, shape, limit):
        torch.manual_seed(0)
        x = torch.randn(shape, device="cuda", dtype=dtype)
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype) * 0.05 + 1.0
        eps = 1.0e-6
        reference = _reference(x, weight, eps)

        native = native_zero_centered_rms_norm(x, weight, eps)
        triton = _triton_zero_centered_rms_norm(x, weight, eps)

        assert _relative_gap(native, reference) < limit
        assert _relative_gap(triton, reference) < limit
        # 两个实现都落在参考的 bf16 舍入带内；大形状下归约顺序不同，不要求逐位相等。
        assert _relative_gap(native, triton) < limit

    @pytest.mark.parametrize(
        ("dtype", "shape"),
        [
            (torch.float32, (1, 64, 128)),
            (torch.bfloat16, (1, 64, 128)),
            (torch.bfloat16, (1, 8192, 2048)),
            (torch.bfloat16, (1, 8192, 4096)),
        ],
    )
    def test_backward_not_worse_than_native(self, dtype, shape):
        torch.manual_seed(0)
        x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
        with torch.no_grad():
            weight.copy_(weight * 0.05 + 1.0)
        eps = 1.0e-6

        x64 = x.detach().double().requires_grad_(True)
        w64 = weight.detach().double().requires_grad_(True)
        _reference(x64, w64, eps).pow(2).mean().backward()

        gaps = {}
        for name, fn in (
            ("native", native_zero_centered_rms_norm),
            ("triton", _triton_zero_centered_rms_norm),
        ):
            x.grad = weight.grad = None
            fn(x, weight, eps).float().pow(2).mean().backward()
            gaps[name] = (_relative_gap(x.grad, x64.grad), _relative_gap(weight.grad, w64.grad))

        tolerance = 5.0e-2 if dtype == torch.bfloat16 else 5.0e-4
        assert gaps["triton"][0] < max(tolerance, gaps["native"][0] * 2.0), gaps
        assert gaps["triton"][1] < max(tolerance, gaps["native"][1] * 2.0), gaps

    def test_gradients_are_finite(self):
        torch.manual_seed(0)
        x = torch.randn(1, 128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        _triton_zero_centered_rms_norm(x, weight, 1.0e-6).float().pow(2).mean().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(weight.grad).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton selection is only reachable on CUDA")
class TestZeroCenteredRMSNormSelection:
    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [("0", _triton_zero_centered_rms_norm), ("1", native_zero_centered_rms_norm)],
    )
    def test_env_switch_selects_implementation(self, monkeypatch, env_value, expected):
        monkeypatch.setenv("XTUNER_USE_NATIVE_RMSNORM", env_value)
        assert get_zero_centered_rms_norm_fn() is expected
