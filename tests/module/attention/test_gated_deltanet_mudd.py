"""GatedDeltaNet MUDD QKV 投影的行切片回归测试。

TestGatedDeltaNetMUDDQKVProjection
    test_row_sliced_projection_matches_whole_weight_projection: 按 source 行切片的三次投影与整权重三次投影前向、梯度一致。
    test_row_sliced_projection_skips_whole_weight_calls: 行切片路径不再调用整权重投影。
    test_projection_falls_back_for_sharded_or_float8_weights: 分片权重或 float8 线性退回整权重投影。
TestGatedDeltaNetMuonSplitSizes
    test_muon_split_sizes_cover_the_fused_qkv_rows: MuonSplit 的 Q/K/V 行划分与 fused in_proj_qkv 一致且覆盖全部行。
"""

import pytest
import torch
import torch.nn as nn


try:
    from xtuner.v1.module.attention.gated_deltanet import GatedDeltaNet
except ImportError:  # GDN 依赖 fla/triton 等可选算子，CPU-only 环境下自动跳过
    GatedDeltaNet = None

pytestmark = pytest.mark.skipif(GatedDeltaNet is None, reason="GatedDeltaNet optional deps unavailable")

HIDDEN_SIZE = 16
KEY_DIM = 8
VALUE_DIM = 12
FUSED_DIM = KEY_DIM * 2 + VALUE_DIM


class _FusedProjection(nn.Module):
    """Stand-in for ``in_proj_qkv``: exposes ``weight`` and projects the full fused width."""

    def __init__(self, rows: int = FUSED_DIM, counting: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rows, HIDDEN_SIZE))
        self.calls = 0 if counting else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calls is not None:
            self.calls += 1
        return x @ self.weight.t()


class _ExplodingProjection(nn.Module):
    """Fails loudly if the row-sliced path still calls the whole-weight projection."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(FUSED_DIM, HIDDEN_SIZE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("the row-sliced path must not call the whole-weight projection")


class _StubRoute:
    """Minimal stand-in exposing only the attributes ``_mudd_qkv_projection`` touches."""

    def __init__(self, projection: nn.Module, float8_cfg=None) -> None:
        self.key_dim = KEY_DIM
        self.value_dim = VALUE_DIM
        self.float8_cfg = float8_cfg
        self.in_proj_qkv = projection


class _Float8ConfigStub:
    def __init__(self, granularity) -> None:
        self.scaling_granularity_gemm = granularity


def _build_pair() -> tuple[_StubRoute, _StubRoute]:
    """Two stubs sharing identical weights: the row-sliced one and the reference one."""
    torch.manual_seed(0)
    route = _StubRoute(_FusedProjection())
    reference = _StubRoute(_FusedProjection())
    reference.in_proj_qkv.weight.data.copy_(route.in_proj_qkv.weight.data)
    return route, reference


def _inputs(requires_grad: bool = False) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1)
    return tuple(torch.randn(2, 3, HIDDEN_SIZE, generator=generator, requires_grad=requires_grad) for _ in range(3))


def _whole_weight_reference(
    route: _StubRoute, query_input: torch.Tensor, key_input: torch.Tensor, value_input: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """优化前的实现：对每个 source 计算整权重投影，再切出该 source 的段。"""
    return (
        route.in_proj_qkv(query_input)[..., : route.key_dim],
        route.in_proj_qkv(key_input)[..., route.key_dim : 2 * route.key_dim],
        route.in_proj_qkv(value_input)[..., 2 * route.key_dim :],
    )


class TestGatedDeltaNetMUDDQKVProjection:
    def test_row_sliced_projection_matches_whole_weight_projection(self):
        route, reference_route = _build_pair()
        query_input, key_input, value_input = _inputs(requires_grad=True)
        outputs = GatedDeltaNet._mudd_qkv_projection(route, query_input, key_input, value_input)
        sum(output.float().square().sum() for output in outputs).backward()

        reference_inputs = [
            tensor.detach().clone().requires_grad_() for tensor in (query_input, key_input, value_input)
        ]
        expected = _whole_weight_reference(reference_route, *reference_inputs)
        sum(output.float().square().sum() for output in expected).backward()

        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-6)
        for actual_grad, reference_input in zip((query_input, key_input, value_input), reference_inputs):
            torch.testing.assert_close(actual_grad.grad, reference_input.grad, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(
            route.in_proj_qkv.weight.grad, reference_route.in_proj_qkv.weight.grad, rtol=1e-4, atol=1e-6
        )

    def test_row_sliced_projection_bf16_baseline(self):
        """量化 bf16 下行切片投影相对整权重投影的浮点偏差——模型级 Step-0 对齐阈值即取自此量级。"""
        route, reference_route = _build_pair()
        route.in_proj_qkv.to(torch.bfloat16)
        reference_route.in_proj_qkv.to(torch.bfloat16)
        query_input, key_input, value_input = (tensor.to(torch.bfloat16) for tensor in _inputs())

        outputs = GatedDeltaNet._mudd_qkv_projection(route, query_input, key_input, value_input)
        expected = _whole_weight_reference(reference_route, query_input, key_input, value_input)

        max_diff = max(
            (actual.float() - reference.float()).abs().max().item() for actual, reference in zip(outputs, expected)
        )
        scale = max(reference.float().abs().max().item() for reference in expected)
        # bf16 只有 8 位尾数：逐段 GEMM 与融合 GEMM 的累加顺序不同，差值应停在 bf16 分辨率量级，
        # 明显低于"接错线"会产生的偏差。
        assert max_diff <= 1.0e-2 * max(scale, 1.0), (
            f"bf16 row-sliced projection deviates by {max_diff:.3e} (scale {scale:.3e})"
        )

    def test_row_sliced_projection_skips_whole_weight_calls(self):
        route = _StubRoute(_ExplodingProjection())
        query_input, key_input, value_input = _inputs()
        outputs = GatedDeltaNet._mudd_qkv_projection(route, query_input, key_input, value_input)

        weight = route.in_proj_qkv.weight
        expected = (
            query_input @ weight[:KEY_DIM].t(),
            key_input @ weight[KEY_DIM : 2 * KEY_DIM].t(),
            value_input @ weight[2 * KEY_DIM :].t(),
        )
        assert [output.shape[-1] for output in outputs] == [KEY_DIM, KEY_DIM, VALUE_DIM]
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    def test_projection_falls_back_for_sharded_or_float8_weights(self):
        cases = {
            "sharded": _StubRoute(_FusedProjection(rows=FUSED_DIM // 2, counting=True)),
            "float8": _StubRoute(_FusedProjection(counting=True), float8_cfg=_Float8ConfigStub(object())),
        }
        for name, route in cases.items():
            query_input, key_input, value_input = _inputs()
            outputs = GatedDeltaNet._mudd_qkv_projection(route, query_input, key_input, value_input)
            assert route.in_proj_qkv.calls == 3, f"{name}: expected the whole-weight path"
            expected = _whole_weight_reference(route, query_input, key_input, value_input)
            for actual, reference in zip(outputs, expected):
                torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


class TestGatedDeltaNetMuonSplitSizes:
    def test_muon_split_sizes_cover_the_fused_qkv_rows(self):
        route = _StubRoute(_FusedProjection())
        splits = GatedDeltaNet.get_muon_split_sizes(route)
        assert list(splits) == [route.in_proj_qkv.weight]
        assert splits[route.in_proj_qkv.weight] == (KEY_DIM, KEY_DIM, VALUE_DIM)
        assert sum(splits[route.in_proj_qkv.weight]) == FUSED_DIM
