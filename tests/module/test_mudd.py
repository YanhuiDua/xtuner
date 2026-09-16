"""Qwen35MUDDRoute PrePostDANorm identity path 回归测试。

TestQwen35MUDDRouteIdentity
    test_zero_init_is_identity: 零初始化时四路输出严格等于 raw 当前层输出。
    test_update_matches_manual_formula: 扰动静态先验后输出与手算公式一致。
    test_gradient_flows_to_history_and_router: 历史 source 与路由器参数均可收到梯度。
    test_router_norm_affine_switch_matches_reference: router norm 的仿射开关决定是否施加可学习 scale。
TestQwen35MUDDRouteInit
    test_static_a_init_is_zero: 静态先验零初始化。
    test_init_values: 静态先验全 0,Pre/Post-DA scale 分别为 1 与 1e-3。
    test_init_extra_weights_restores_values: init_extra_weights 可将扰动后的参数恢复为约定初值。
TestQwen35MUDDRouteGlobalInit
    test_generic_initializer_keeps_zero_init_identity: 经全局初始化器后 w2 仍为零、四路输出仍严格恒等。
TestQwen35MUDDStack
    test_zero_init_stack_reads_back_layer_output: 零初始化时 finish_layer 后 final_read 严格等于当层输出。
TestQwen35MUDDRouteFusedParity
    test_fused_eager_matches_loop_reference: 融合 eager 路径与逐 source 循环参考实现输出一致。
TestQwen35MUDDRouteCompile
    test_compile_matches_eager_fused: compile 路径与融合 eager 路径输出一致。
    test_compile_zero_init_is_identity: compile 路径零初始化时四路输出严格等于 raw 当前层输出。
    test_compile_handles_multiple_widths: 按 source 数特化的编译路由可处理多种宽度且输出正确。
TestQwen35MUDDRouteTritonKernel
    test_triton_kernel_matches_loop_reference: Triton kernel 前向与逐 source 循环参考实现输出一致。
    test_triton_kernel_backward_matches_loop_reference: Triton kernel 反向梯度与循环参考实现一致。
    test_triton_kernel_zero_init_is_identity: Triton kernel 零初始化时四路输出严格等于 raw 当前层输出。
    test_triton_kernel_handles_multiple_widths: Triton kernel 可处理多种 source 宽度且输出正确。
    test_triton_kernel_bf16_smoke: bf16 下前向/反向可运行且梯度有限。
"""

import pytest
import torch
import torch.nn.functional as F

from xtuner.v1.module.mudd import Qwen35MUDD, Qwen35MUDDRoute


HIDDEN_SIZE = 8
EPS = 1e-6


def _build_route(num_sources: int = 3, router_norm_affine: bool = False) -> Qwen35MUDDRoute:
    torch.manual_seed(0)
    return Qwen35MUDDRoute(HIDDEN_SIZE, num_sources, EPS, router_norm_affine=router_norm_affine)


def _build_inputs(num_sources: int = 3) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    current_raw = torch.randn(2, 3, HIDDEN_SIZE, generator=generator)
    sources = [torch.randn(2, 3, HIDDEN_SIZE, generator=generator) for _ in range(num_sources - 1)]
    return current_raw, [*sources, current_raw]


class TestQwen35MUDDRouteIdentity:
    def test_zero_init_is_identity(self):
        route = _build_route()
        current_raw, sources = _build_inputs()
        outputs = route(current_raw, sources)
        assert len(outputs) == 4
        for output in outputs:
            assert torch.equal(output, current_raw)

    def test_update_matches_manual_formula(self):
        route = _build_route()
        current_raw, sources = _build_inputs()
        with torch.no_grad():
            route.static_a[0, 0] = 2.0
        outputs = route(current_raw, sources)
        expected_update = 2.0 * F.rms_norm(sources[0], (HIDDEN_SIZE,), weight=route.source_norm, eps=EPS)
        expected = current_raw + F.rms_norm(expected_update, (HIDDEN_SIZE,), weight=route.post_norm[0], eps=EPS)
        torch.testing.assert_close(outputs[0], expected)
        for output in outputs[1:]:
            assert torch.equal(output, current_raw)

    def test_gradient_flows_to_history_and_router(self):
        route = _build_route()
        generator = torch.Generator().manual_seed(1)
        current_raw = torch.randn(2, 3, HIDDEN_SIZE, generator=generator, requires_grad=True)
        sources = [torch.randn(2, 3, HIDDEN_SIZE, generator=generator, requires_grad=True) for _ in range(2)]
        sources.append(current_raw)
        with torch.no_grad():
            route.static_a[1, 0] = 1.0
        outputs = route(current_raw, sources)
        loss = torch.stack(outputs).sum()
        loss.backward()
        assert route.router_w2.grad is not None
        assert route.static_a.grad is not None
        assert route.post_norm.grad is not None and route.post_norm.grad[1].abs().sum() > 0
        assert sources[0].grad is not None and sources[0].grad.abs().sum() > 0
        assert current_raw.grad is not None and current_raw.grad.abs().sum() > 0

    def test_router_norm_affine_switch_matches_reference(self):
        current_raw, sources = _build_inputs(num_sources=4)

        # 非仿射(默认):router_norm 不参与归一化,但零值锚点让它仍能拿到梯度。
        non_affine = _build_route(num_sources=4)
        _perturb_route(non_affine)
        shifted = _build_route(num_sources=4)
        _perturb_route(shifted)
        with torch.no_grad():
            non_affine.router_norm.fill_(3.0)
        for output, reference in zip(non_affine(current_raw, sources), shifted(current_raw, sources)):
            assert torch.equal(output, reference)
        non_affine(current_raw, sources)[0].sum().backward()
        assert non_affine.router_norm.grad is not None
        assert not non_affine.router_norm.grad.any()

        # 仿射:router_norm 是归一化的可学习 scale,改动它会改变输出。
        affine = _build_route(num_sources=4, router_norm_affine=True)
        _perturb_route(affine)
        reference = _build_route(num_sources=4, router_norm_affine=True)
        _perturb_route(reference)
        with torch.no_grad():
            affine.router_norm.mul_(2.0)
        differences = [
            not torch.equal(output, expected)
            for output, expected in zip(affine(current_raw, sources), reference(current_raw, sources))
        ]
        assert any(differences)


class TestQwen35MUDDRouteInit:
    def test_static_a_init_is_zero(self):
        route = _build_route(num_sources=4)
        assert torch.equal(route.static_a, torch.zeros_like(route.static_a))

    def test_init_values(self):
        route = _build_route()
        assert torch.equal(route.static_a, torch.zeros_like(route.static_a))
        assert torch.equal(route.router_norm, torch.ones_like(route.router_norm))
        assert torch.equal(route.source_norm, torch.ones_like(route.source_norm))
        assert torch.equal(route.post_norm, torch.full_like(route.post_norm, 1.0e-3))

    def test_init_extra_weights_restores_values(self):
        route = _build_route()
        with torch.no_grad():
            route.static_a.fill_(7.0)
            route.router_w1.fill_(7.0)
            route.router_w2.fill_(7.0)
            route.router_norm.fill_(3.0)
            route.source_norm.fill_(4.0)
            route.post_norm.fill_(9.0)
        initialized = route.init_extra_weights()
        assert initialized == {"static_a", "router_w1", "router_w2", "router_norm", "source_norm", "post_norm"}
        assert torch.equal(route.static_a, torch.zeros_like(route.static_a))
        assert torch.equal(route.router_w2, torch.zeros_like(route.router_w2))
        assert torch.equal(route.router_norm, torch.ones_like(route.router_norm))
        assert torch.equal(route.source_norm, torch.ones_like(route.source_norm))
        assert torch.equal(route.post_norm, torch.full_like(route.post_norm, 1.0e-3))


class TestQwen35MUDDRouteGlobalInit:
    def test_generic_initializer_keeps_zero_init_identity(self):
        # 回归:default_init_weights 会对名字不含 "norm" 的子模块 .weight 施加
        # normal_(0, 0.02)。router_w1/router_w2 若挂在 nn.Linear 上就会被覆盖,
        # 零初始化恒等契约随之失效——而直接构造 route 的测试都看不到这一步。
        from xtuner.v1.utils import default_init_weights

        route = _build_route()
        # 裸 route 的模块名为空,返回值会带前导点;模型级调用没有这个前缀。
        initialized = {name.lstrip(".") for name in default_init_weights(route)}
        assert not {name for name, _ in route.named_parameters()} - initialized
        assert torch.equal(route.router_w2, torch.zeros_like(route.router_w2))
        current_raw, sources = _build_inputs()
        for output in route(current_raw, sources):
            assert torch.equal(output, current_raw)


class TestQwen35MUDDStack:
    def test_zero_init_stack_reads_back_layer_output(self):
        torch.manual_seed(0)
        mudd = Qwen35MUDD(HIDDEN_SIZE, num_layers=2, eps=EPS)
        generator = torch.Generator().manual_seed(0)
        embedding = torch.randn(2, 3, HIDDEN_SIZE, generator=generator)
        state = mudd.initial_state(embedding)
        layer0_output = torch.randn(2, 3, HIDDEN_SIZE, generator=generator)
        state = mudd.finish_layer(0, layer0_output, state)
        assert torch.equal(mudd.final_read(state), layer0_output)
        assert len(state.history) == 2
        layer1_output = torch.randn(2, 3, HIDDEN_SIZE, generator=generator)
        state = mudd.finish_layer(1, layer1_output, state)
        assert torch.equal(mudd.final_read(state), layer1_output)
        assert len(state.history) == 3


def _loop_reference(route: Qwen35MUDDRoute, current_raw: torch.Tensor, sources: list[torch.Tensor]):
    """逐 source 循环的参考实现,对应融合改造前的 eager 行为。"""
    if route.router_norm_affine:
        route_input = F.rms_norm(current_raw, (HIDDEN_SIZE,), weight=route.router_norm, eps=EPS)
    else:
        # 非仿射模式下生产代码用零值项把 router_norm 留在自动求导图里,参考实现同样带上:
        # 数值上无影响,但让两侧的 router_norm.grad 都是零而不是一侧零、一侧 None。
        route_input = F.rms_norm(current_raw, (HIDDEN_SIZE,), eps=EPS) + route.router_norm.sum() * 0.0
    sources = [F.rms_norm(source, (HIDDEN_SIZE,), weight=route.source_norm, eps=EPS) for source in sources]
    weights = F.linear(F.gelu(F.linear(route_input, route.router_w1)), route.router_w2)
    weights = weights.view(*weights.shape[:-1], route.num_streams, route.num_sources) + route.static_a
    updates = [
        sum(source * weights[..., way, source_idx : source_idx + 1] for source_idx, source in enumerate(sources))
        for way in range(route.num_streams)
    ]
    return [
        current_raw + F.rms_norm(update, (HIDDEN_SIZE,), weight=route.post_norm[way], eps=EPS)
        for way, update in enumerate(updates)
    ]


class TestQwen35MUDDRouteFusedParity:
    def test_fused_eager_matches_loop_reference(self):
        route = _build_route(num_sources=4)
        generator = torch.Generator().manual_seed(3)
        with torch.no_grad():
            route.router_w2.copy_(torch.randn(route.router_w2.shape, generator=generator) * 0.1)
            route.static_a.copy_(torch.randn(route.static_a.shape, generator=generator) * 0.1)
            route.router_norm.copy_(torch.randn(route.router_norm.shape, generator=generator).abs() + 0.5)
            route.source_norm.copy_(torch.randn(route.source_norm.shape, generator=generator).abs() + 0.5)
            route.post_norm.copy_(torch.randn(route.post_norm.shape, generator=generator) * 0.05)
        current_raw, sources = _build_inputs(num_sources=4)
        outputs = route(current_raw, sources)
        expected = _loop_reference(route, current_raw, sources)
        assert len(outputs) == len(expected) == 4
        for output, reference in zip(outputs, expected):
            torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-6)


def _compile_available() -> bool:
    try:
        probe = torch.compile(lambda x: x * 2, fullgraph=True)
        probe(torch.ones(1))
    except Exception:
        return False
    return True


def _build_compiled_route_like(reference: Qwen35MUDDRoute) -> Qwen35MUDDRoute:
    route = Qwen35MUDDRoute(
        HIDDEN_SIZE,
        reference.num_sources,
        EPS,
        compile_route=True,
        router_norm_affine=reference.router_norm_affine,
    )
    route.load_state_dict(reference.state_dict())
    return route


class TestQwen35MUDDRouteCompile:
    def test_compile_matches_eager_fused(self):
        if not _compile_available():
            pytest.skip("torch.compile / inductor unavailable in this environment")
        reference = _build_route(num_sources=4)
        generator = torch.Generator().manual_seed(3)
        with torch.no_grad():
            reference.router_w2.copy_(torch.randn(reference.router_w2.shape, generator=generator) * 0.1)
            reference.static_a.copy_(torch.randn(reference.static_a.shape, generator=generator) * 0.1)
            reference.router_norm.copy_(torch.randn(reference.router_norm.shape, generator=generator).abs() + 0.5)
            reference.source_norm.copy_(torch.randn(reference.source_norm.shape, generator=generator).abs() + 0.5)
            reference.post_norm.copy_(torch.randn(reference.post_norm.shape, generator=generator) * 0.05)
        route = _build_compiled_route_like(reference)
        current_raw, sources = _build_inputs(num_sources=4)
        outputs = route(current_raw, sources)
        expected = reference(current_raw, sources)
        assert len(outputs) == len(expected) == 4
        for output, reference_output in zip(outputs, expected):
            torch.testing.assert_close(output, reference_output, rtol=1e-3, atol=1e-5)

    def test_compile_zero_init_is_identity(self):
        if not _compile_available():
            pytest.skip("torch.compile / inductor unavailable in this environment")
        reference = _build_route()
        route = _build_compiled_route_like(reference)
        current_raw, sources = _build_inputs()
        outputs = route(current_raw, sources)
        assert len(outputs) == 4
        for output in outputs:
            assert torch.equal(output, current_raw)

    def test_compile_handles_multiple_widths(self):
        if not _compile_available():
            pytest.skip("torch.compile / inductor unavailable in this environment")
        for num_sources in (2, 3, 7, 33):
            reference = _build_route(num_sources=num_sources)
            generator = torch.Generator().manual_seed(5)
            with torch.no_grad():
                reference.router_w2.copy_(torch.randn(reference.router_w2.shape, generator=generator) * 0.1)
                reference.static_a.copy_(torch.randn(reference.static_a.shape, generator=generator) * 0.1)
            route = _build_compiled_route_like(reference)
            current_raw, sources = _build_inputs(num_sources=num_sources)
            outputs = route(current_raw, sources)
            expected = _loop_reference(route, current_raw, sources)
            assert len(outputs) == len(expected) == 4
            for output, reference_output in zip(outputs, expected):
                torch.testing.assert_close(output, reference_output, rtol=1e-3, atol=1e-5)


def _triton_kernel_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _build_kernel_route_like(reference: Qwen35MUDDRoute) -> Qwen35MUDDRoute:
    route = Qwen35MUDDRoute(
        HIDDEN_SIZE,
        reference.num_sources,
        EPS,
        route_kernel=True,
        router_norm_affine=reference.router_norm_affine,
    )
    route.load_state_dict(reference.state_dict())
    return route


def _perturb_route(route: Qwen35MUDDRoute, seed: int = 3) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        route.router_w2.copy_(torch.randn(route.router_w2.shape, generator=generator) * 0.1)
        route.static_a.copy_(torch.randn(route.static_a.shape, generator=generator) * 0.1)
        route.router_norm.copy_(torch.randn(route.router_norm.shape, generator=generator).abs() + 0.5)
        route.source_norm.copy_(torch.randn(route.source_norm.shape, generator=generator).abs() + 0.5)
        route.post_norm.copy_(torch.randn(route.post_norm.shape, generator=generator) * 0.05)


class TestQwen35MUDDRouteTritonKernel:
    def test_triton_kernel_matches_loop_reference(self):
        if not _triton_kernel_available():
            pytest.skip("triton or CUDA unavailable in this environment")
        reference = _build_route(num_sources=4)
        _perturb_route(reference)
        route = _build_kernel_route_like(reference).cuda()
        reference = reference.cuda()
        current_raw, sources = _build_inputs(num_sources=4)
        current_raw = current_raw.cuda()
        sources = [source.cuda() for source in sources]
        outputs = route(current_raw, sources)
        expected = _loop_reference(reference, current_raw, sources)
        assert len(outputs) == len(expected) == 4
        for output, reference_output in zip(outputs, expected):
            torch.testing.assert_close(output, reference_output, rtol=1e-4, atol=1e-6)

    def test_triton_kernel_backward_matches_loop_reference(self):
        if not _triton_kernel_available():
            pytest.skip("triton or CUDA unavailable in this environment")
        reference = _build_route(num_sources=4)
        _perturb_route(reference)
        route = _build_kernel_route_like(reference).cuda()
        reference = reference.cuda()
        generator = torch.Generator().manual_seed(7)
        current_raw, sources = _build_inputs(num_sources=4)
        upstream = [torch.randn(2, 3, HIDDEN_SIZE, generator=generator).cuda() for _ in range(4)]
        # Two identical sets of leaves so each path gets its own .grad.
        kernel_current = current_raw.cuda().requires_grad_()
        kernel_sources = [source.cuda().detach().requires_grad_() for source in sources[:-1]] + [kernel_current]
        reference_current = current_raw.cuda().detach().requires_grad_()
        reference_sources = [source.cuda().detach().requires_grad_() for source in sources[:-1]] + [reference_current]
        kernel_loss = sum((output * up).sum() for output, up in zip(route(kernel_current, kernel_sources), upstream))
        kernel_loss.backward()
        reference_loss = sum(
            (output * up).sum()
            for output, up in zip(_loop_reference(reference, reference_current, reference_sources), upstream)
        )
        reference_loss.backward()
        leaf_pairs = [(kernel_current.grad, reference_current.grad)] + [
            (kernel_source.grad, reference_source.grad)
            for kernel_source, reference_source in zip(kernel_sources[:-1], reference_sources[:-1])
        ]
        paired = [
            (route.router_w1.grad, reference.router_w1.grad),
            (route.router_w2.grad, reference.router_w2.grad),
            (route.static_a.grad, reference.static_a.grad),
        ]
        paired += [
            (route.router_norm.grad, reference.router_norm.grad),
            (route.source_norm.grad, reference.source_norm.grad),
            (route.post_norm.grad, reference.post_norm.grad),
        ]
        for kernel_grad, reference_grad in leaf_pairs + paired:
            torch.testing.assert_close(kernel_grad, reference_grad, rtol=1e-4, atol=1e-6)

    def test_triton_kernel_zero_init_is_identity(self):
        if not _triton_kernel_available():
            pytest.skip("triton or CUDA unavailable in this environment")
        for dtype in (torch.float32, torch.bfloat16):
            route = _build_kernel_route_like(_build_route()).cuda().to(dtype)
            current_raw, sources = _build_inputs()
            current_raw = current_raw.cuda().to(dtype)
            sources = [source.cuda().to(dtype) for source in sources]
            outputs = route(current_raw, sources)
            assert len(outputs) == 4
            for output in outputs:
                assert torch.equal(output, current_raw)

    def test_triton_kernel_handles_multiple_widths(self):
        if not _triton_kernel_available():
            pytest.skip("triton or CUDA unavailable in this environment")
        for num_sources in (2, 3, 7, 33):
            reference = _build_route(num_sources=num_sources)
            _perturb_route(reference, seed=5)
            route = _build_kernel_route_like(reference).cuda()
            reference = reference.cuda()
            current_raw, sources = _build_inputs(num_sources=num_sources)
            current_raw = current_raw.cuda()
            sources = [source.cuda() for source in sources]
            outputs = route(current_raw, sources)
            expected = _loop_reference(reference, current_raw, sources)
            assert len(outputs) == len(expected) == 4
            for output, reference_output in zip(outputs, expected):
                torch.testing.assert_close(output, reference_output, rtol=1e-4, atol=1e-6)

    def test_triton_kernel_bf16_smoke(self):
        if not _triton_kernel_available():
            pytest.skip("triton or CUDA unavailable in this environment")
        reference = _build_route(num_sources=4)
        _perturb_route(reference)
        route = _build_kernel_route_like(reference).cuda().to(torch.bfloat16)
        current_raw, sources = _build_inputs(num_sources=4)
        current_raw = current_raw.cuda().to(torch.bfloat16).requires_grad_()
        sources = [source.cuda().to(torch.bfloat16).detach().requires_grad_() for source in sources[:-1]]
        sources.append(current_raw)
        outputs = route(current_raw, sources)
        assert len(outputs) == 4
        for output in outputs:
            assert output.shape == current_raw.shape
            assert output.dtype == torch.bfloat16
            assert torch.isfinite(output.float()).all()
        loss = torch.stack([output.float() for output in outputs]).sum()
        loss.backward()
        assert route.router_w2.grad is not None and torch.isfinite(route.router_w2.grad.float()).all()
        assert route.static_a.grad is not None and torch.isfinite(route.static_a.grad.float()).all()
        assert current_raw.grad is not None and torch.isfinite(current_raw.grad.float()).all()
        for source in sources[:-1]:
            assert source.grad is not None and torch.isfinite(source.grad.float()).all()
        assert torch.isfinite(route.source_norm.grad.float()).all()
        assert torch.isfinite(route.post_norm.grad.float()).all()
