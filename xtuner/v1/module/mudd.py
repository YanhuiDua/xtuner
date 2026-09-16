"""Qwen3.5 four-way full-history MUDD routing for dense training."""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xtuner.v1.utils import init_params


try:  # Optional fused Triton route kernel; the fused eager path stays the default.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


_COMPILED_ROUTES: dict[tuple[int, int, bool], Callable] = {}


def _raise_dynamo_limits() -> None:
    """Give Dynamo room for one graph per layer width without tripping its guard."""
    for name, floor in (
        ("cache_size_limit", 128),
        ("recompile_limit", 128),
        ("accumulated_recompile_limit", 512),
    ):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), floor))


def _compiled_route(
    num_sources: int,
    num_streams: int,
    router_norm_affine: bool,
    hidden_size: int,
    eps: float,
) -> Callable:
    """Return a compiled route specialised to one layer's source count.

    The whole route sits inside the compiled region -- router MLP included -- the
    endpoints are passed one at a time rather than pre-stacked, and the graph is
    static. The reference implementation specialises per source count for the same
    reason: one dynamically shaped graph shared by every layer is measurably
    slower, and a layer's source count is known when the route is built.

    Args:
        num_sources (int): Number of endpoints this layer's route combines.
        num_streams (int): Number of residual streams produced.
        router_norm_affine (bool): Whether the router's input norm carries a learnable scale.
        hidden_size (int): Hidden dimension.
        eps (float): RMS normalization epsilon.

    Returns:
        Callable: Compiled route returning one ``[..., num_streams, hidden]`` tensor.
    """
    key = (int(num_sources), int(num_streams), bool(router_norm_affine))
    cached = _COMPILED_ROUTES.get(key)
    if cached is not None:
        return cached

    def route_fn(
        current_raw: Tensor,
        router_norm: Tensor,
        router_w1: Tensor,
        router_w2: Tensor,
        static_a: Tensor,
        source_norm: Tensor,
        post_norm: Tensor,
        *history: Tensor,
    ) -> Tensor:
        if router_norm_affine:
            route_input = F.rms_norm(current_raw, (hidden_size,), weight=router_norm, eps=eps)
        else:
            route_input = F.rms_norm(current_raw, (hidden_size,), eps=eps) + router_norm.sum() * 0.0
        weights = F.linear(F.gelu(F.linear(route_input, router_w1)), router_w2)
        weights = weights.view(*weights.shape[:-1], num_streams, num_sources) + static_a
        # Normalise and combine endpoint by endpoint: no [..., num_sources, hidden]
        # stack is ever materialised, and Dynamo unrolls both loops into one fused
        # reduction per stream.
        normed = tuple(F.rms_norm(source, (hidden_size,), weight=source_norm, eps=eps) for source in history)
        streams = []
        for stream_idx in range(num_streams):
            update = sum(
                normed[source_idx] * weights[..., stream_idx, source_idx].unsqueeze(-1)
                for source_idx in range(num_sources)
            )
            normed_update = F.rms_norm(update, (hidden_size,), eps=eps)
            streams.append(current_raw + normed_update * post_norm[stream_idx])
        return torch.stack(streams, dim=-2)

    _raise_dynamo_limits()
    compiled = torch.compile(route_fn, fullgraph=True, dynamic=False)
    _COMPILED_ROUTES[key] = compiled
    return compiled


@torch._dynamo.disable
def _invoke_compiled_route(compiled_fn: Callable, *args: Tensor) -> Tensor:
    """Call a compiled route from outside any enclosing Dynamo frame.

    Invoking it from inside a traced region makes Dynamo guard on the call site
    instead of the graph, which is what exhausts the recompile limit.
    """
    return compiled_fn(*args)


if _HAS_TRITON:

    @triton.jit
    def _mudd_combine_fwd_kernel(
        current_ptr,
        stacked_ptr,
        weights_ptr,
        source_norm_ptr,
        post_norm_ptr,
        out_ptr,
        inv_rms_ptr,
        eps,
        N: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        PREPOST: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_H)
        offs4 = tl.arange(0, 4)
        h_mask = offs_h < H
        if PREPOST:
            source_norm = tl.load(source_norm_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        acc = tl.zeros((4, BLOCK_H), dtype=tl.float32)
        for n in tl.static_range(N):
            x = tl.load(stacked_ptr + row * N * H + n * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
            if PREPOST:
                inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
                # Hand the per-source normalization to the backward pass, which
                # would otherwise redo this full-row reduction per source. That
                # reduction is what caps the kernel's bandwidth (measured about
                # 1.4 TB/s with it against 4.3 TB/s without), so paying 4 bytes
                # per source here buys a dependency-free backward loop.
                tl.store(inv_rms_ptr + row * N + n, inv_rms)
                x = x * inv_rms * source_norm
            w = tl.load(weights_ptr + row * 4 * N + offs4 * N + n).to(tl.float32)
            acc += w[:, None] * x[None, :]
        if PREPOST:
            inv_rms_u = tl.rsqrt(tl.sum(acc * acc, axis=1) / H + eps)
            post_norm = tl.load(post_norm_ptr + offs4[:, None] * H + offs_h, mask=h_mask[None, :], other=0.0).to(
                tl.float32
            )
            current = tl.load(current_ptr + row * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
            acc = current[None, :] + acc * inv_rms_u[:, None] * post_norm
        tl.store(
            out_ptr + row * 4 * H + offs4[:, None] * H + offs_h,
            acc.to(out_ptr.dtype.element_ty),
            mask=h_mask[None, :],
        )

    @triton.jit
    def _mudd_combine_bwd_kernel(
        grad_out_ptr,
        stacked_ptr,
        weights_ptr,
        source_norm_ptr,
        post_norm_ptr,
        inv_rms_ptr,
        grad_stacked_ptr,
        grad_weights_ptr,
        grad_current_ptr,
        grad_source_norm_ptr,
        grad_post_norm_ptr,
        eps,
        ROWS,
        N: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROWS_PER_PROG: tl.constexpr,
        PREPOST: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_H)
        offs4 = tl.arange(0, 4)
        h_mask = offs_h < H
        mask4h = h_mask[None, :]
        if PREPOST:
            source_norm = tl.load(source_norm_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
            post_norm = tl.load(post_norm_ptr + offs4[:, None] * H + offs_h, mask=mask4h, other=0.0).to(tl.float32)
            # Parameter grads accumulate in registers across the rows this program
            # handles; a single atomic set at the end replaces one atomic vector per
            # row (thousands of programs would otherwise serialize on these few
            # addresses).
            dg_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
            dp_acc = tl.zeros((4, BLOCK_H), dtype=tl.float32)
        for i in range(ROWS_PER_PROG):
            row = pid * ROWS_PER_PROG + i
            valid = row < ROWS
            safe_row = tl.minimum(row, ROWS - 1)
            gout = tl.load(
                grad_out_ptr + safe_row * 4 * H + offs4[:, None] * H + offs_h, mask=mask4h & valid, other=0.0
            ).to(tl.float32)
            # Recompute the forward accumulation: the normed sources are not kept in
            # memory, so only one copy of ``stacked`` stays live in the autograd graph.
            # Per-source normalization comes from the saved forward inv_rms, so this
            # loop carries no cross-lane reduction and streams at full bandwidth.
            acc = tl.zeros((4, BLOCK_H), dtype=tl.float32)
            for n in tl.static_range(N):
                x = tl.load(stacked_ptr + safe_row * N * H + n * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                if PREPOST:
                    x = x * tl.load(inv_rms_ptr + safe_row * N + n) * source_norm
                w = tl.load(weights_ptr + safe_row * 4 * N + offs4 * N + n).to(tl.float32)
                acc += w[:, None] * x[None, :]
            if PREPOST:
                rms_u = tl.sqrt(tl.sum(acc * acc, axis=1) / H + eps)
                a = gout * post_norm
                coeff = tl.sum(a * acc, axis=1) / (H * rms_u * rms_u * rms_u)
                du = a / rms_u[:, None] - acc * coeff[:, None]
                dp_acc += gout * acc / rms_u[:, None]
                tl.store(
                    grad_current_ptr + row * H + offs_h,
                    tl.sum(gout, axis=0).to(grad_current_ptr.dtype.element_ty),
                    mask=h_mask & valid,
                )
            else:
                du = gout
            for n in tl.static_range(N):
                x = tl.load(stacked_ptr + safe_row * N * H + n * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
                if PREPOST:
                    inv_rms = tl.load(inv_rms_ptr + safe_row * N + n)
                    y = x * inv_rms * source_norm
                else:
                    y = x
                w = tl.load(weights_ptr + safe_row * 4 * N + offs4 * N + n).to(tl.float32)
                # The only cross-lane reduction left in this loop: y is needed for
                # both the weight gradient and the source-norm backward.
                grad_w = tl.sum(du * y[None, :], axis=1)
                tl.store(
                    grad_weights_ptr + row * 4 * N + offs4 * N + n,
                    grad_w.to(grad_weights_ptr.dtype.element_ty),
                    mask=(offs4 < 4) & valid,
                )
                e = tl.sum(w[:, None] * du, axis=0)
                if PREPOST:
                    # proj is the source-norm rms backward projection. Written as a
                    # sum over h of (e*source_norm)*x it would need a second
                    # cross-lane reduction; since y = x*inv_rms*source_norm, it is
                    # algebraically identical to this contraction of grad_w with w,
                    # which reuses the reduction just computed and halves the
                    # barrier count of the loop.
                    proj = tl.sum(w * grad_w, axis=0) / (H * inv_rms)
                    eg = e * source_norm
                    dx = eg * inv_rms - x * (proj * inv_rms * inv_rms * inv_rms)
                    dg_acc += e * x * inv_rms
                else:
                    dx = e
                tl.store(
                    grad_stacked_ptr + row * N * H + n * H + offs_h,
                    dx.to(grad_stacked_ptr.dtype.element_ty),
                    mask=h_mask & valid,
                )
        if PREPOST:
            tl.atomic_add(grad_source_norm_ptr + offs_h, dg_acc, mask=h_mask)
            tl.atomic_add(grad_post_norm_ptr + offs4[:, None] * H + offs_h, dp_acc, mask=mask4h)


def _kernel_num_warps(block_h: int) -> int:
    if block_h >= 4096:
        return 16
    if block_h >= 2048:
        return 8
    return 4


class _MUDDCombineTriton(torch.autograd.Function):
    """Autograd wrapper around the fused Triton route-combine kernels.

    Sources arrive as individual arguments rather than as one pre-stacked tensor
    so that the backward can rebuild the stack from endpoints the graph keeps
    alive anyway. Saving the stack instead pins a ``[rows, num_sources, hidden]``
    buffer per layer, and ``num_sources`` grows by one per layer, so a 32-layer
    model holds ~560 hidden-sized slices for the whole backward -- at
    8192x4096 bf16 that is ~35 GiB per sequence, which dominated the route's
    footprint and made the route, not the layers, the memory ceiling.
    """

    @staticmethod
    def forward(
        ctx,
        current_raw: Tensor,
        weights: Tensor,
        source_norm: Tensor,
        post_norm: Tensor,
        eps: float,
        prepost: bool,
        *sources: Tensor,
    ) -> Tensor:
        stacked = torch.stack(sources, dim=-2)
        num_sources, hidden_size = stacked.shape[-2], stacked.shape[-1]
        rows = stacked.numel() // (num_sources * hidden_size)
        out = torch.empty(rows, 4, hidden_size, device=stacked.device, dtype=stacked.dtype)
        # Per-source inverse RMS from the forward, reused by the backward so its
        # loops stay free of cross-lane reductions. 4 bytes per source per row.
        inv_rms = torch.empty(rows, num_sources, device=stacked.device, dtype=torch.float32)
        block_h = triton.next_power_of_2(hidden_size)
        _mudd_combine_fwd_kernel[(rows,)](
            current_raw.reshape(rows, hidden_size),
            stacked.reshape(rows, num_sources, hidden_size),
            weights.reshape(rows, 4, num_sources),
            source_norm,
            post_norm,
            out,
            inv_rms,
            eps,
            N=num_sources,
            H=hidden_size,
            BLOCK_H=block_h,
            PREPOST=prepost,
            num_warps=_kernel_num_warps(block_h),
        )
        ctx.save_for_backward(current_raw, weights, source_norm, post_norm, inv_rms, *sources)
        ctx.eps = eps
        ctx.prepost = prepost
        return out.reshape(*stacked.shape[:-2], 4, hidden_size)

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        saved = ctx.saved_tensors
        current_raw, weights, source_norm, post_norm, inv_rms = saved[:5]
        sources = saved[5:]
        # Rebuilt on demand; the endpoints themselves are alive regardless.
        stacked = torch.stack(sources, dim=-2)
        num_sources, hidden_size = stacked.shape[-2], stacked.shape[-1]
        rows = stacked.numel() // (num_sources * hidden_size)
        grad_out = grad_out.reshape(rows, 4, hidden_size).contiguous()
        grad_stacked = torch.empty_like(stacked)
        grad_weights = torch.empty_like(weights)
        block_h = triton.next_power_of_2(hidden_size)
        if ctx.prepost:
            grad_current = torch.empty_like(current_raw)
            grad_source_norm = torch.zeros(hidden_size, device=stacked.device, dtype=torch.float32)
            grad_post_norm = torch.zeros(4, hidden_size, device=stacked.device, dtype=torch.float32)
        else:
            # Unused pointer placeholders; the kernel never touches them when PREPOST=False.
            grad_current = grad_source_norm = grad_post_norm = grad_stacked
        rows_per_prog = 8
        _mudd_combine_bwd_kernel[(triton.cdiv(rows, rows_per_prog),)](
            grad_out,
            stacked.reshape(rows, num_sources, hidden_size),
            weights.reshape(rows, 4, num_sources),
            source_norm,
            post_norm,
            inv_rms,
            grad_stacked,
            grad_weights,
            grad_current,
            grad_source_norm,
            grad_post_norm,
            ctx.eps,
            rows,
            N=num_sources,
            H=hidden_size,
            BLOCK_H=block_h,
            ROWS_PER_PROG=rows_per_prog,
            PREPOST=ctx.prepost,
            num_warps=_kernel_num_warps(block_h),
        )
        if ctx.prepost:
            grad_source_norm = grad_source_norm.to(source_norm.dtype)
            grad_post_norm = grad_post_norm.to(post_norm.dtype)
        else:
            grad_current = grad_source_norm = grad_post_norm = None
        # One gradient per endpoint; the stack has no node of its own to feed now.
        return grad_current, grad_weights, grad_source_norm, grad_post_norm, None, None, *grad_stacked.unbind(dim=-2)


@dataclass
class Qwen35MUDDState:
    history: list[Tensor]
    streams: tuple[Tensor, Tensor, Tensor, Tensor]


class Qwen35MUDDRoute(nn.Module):
    """Route all preceding layer endpoints into four residual streams.

    The route follows the PrePostDANorm contract: the whole candidate pool is
    normalized with a learnable Pre-DA scale, each stream increment is normalized
    with its own learnable Post-DA scale, and every stream adds the raw
    ``current_raw`` back explicitly, so a zero-initialized route is exactly the
    identity path rather than a renormalized copy of it.

    Args:
        hidden_size (int): Hidden dimension.
        num_sources (int): Number of available history endpoints.
        eps (float): RMS normalization epsilon.
        route_kernel (bool): Use the fused Triton route kernel, the supported path. Takes precedence over
            ``compile_route`` and falls back to the fused eager path when Triton or CUDA is unavailable.
        compile_route (bool): Compile the route with torch.compile, reproducing the reference implementation's
            formulation. Opt-in only: it is slower per step and peaks higher than the kernel at the doc
            configuration, and pays a one-off per-source-count compilation cost.
        router_norm_affine (bool): Apply the learnable scale to the router's own input norm. The reference
            implementation's best-validated configuration runs it affine-free, so that is the default; the
            parameter is kept either way so it stays in the autograd graph.
    """

    num_streams = 4

    def __init__(
        self,
        hidden_size: int,
        num_sources: int,
        eps: float,
        *,
        compile_route: bool = False,
        route_kernel: bool = False,
        router_norm_affine: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_sources = int(num_sources)
        self.eps = float(eps)
        self.compile_route = bool(compile_route)
        self.route_kernel = bool(route_kernel)
        self.router_norm_affine = bool(router_norm_affine)
        route_width = self.num_streams * self.num_sources
        # Plain parameters rather than ``nn.Linear``: the generic initializer rewrites the
        # ``weight`` of every submodule whose name does not contain "norm", which would
        # clobber the zero initialisation the identity contract depends on. Keeping the
        # router weights as parameters of this module puts them out of that walk's reach
        # and lets ``init_extra_weights`` own them, the way A_log/dt_bias already do.
        self.router_w1 = nn.Parameter(torch.empty(route_width, self.hidden_size))
        self.router_w2 = nn.Parameter(torch.empty(route_width, route_width))
        self.static_a = nn.Parameter(torch.zeros(self.num_streams, self.num_sources))
        self.router_norm = nn.Parameter(torch.ones(self.hidden_size))
        self.source_norm = nn.Parameter(torch.ones(self.hidden_size))
        self.post_norm = nn.Parameter(torch.full((self.num_streams, self.hidden_size), 1.0e-3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.router_w1, mean=0.0, std=1.0 / self.hidden_size**0.5)
        nn.init.zeros_(self.router_w2)
        with torch.no_grad():
            # The raw current state is added back explicitly, so the static prior
            # stays zero instead of pinning the latest source.
            self.static_a.zero_()
            nn.init.ones_(self.router_norm)
            nn.init.ones_(self.source_norm)
            nn.init.constant_(self.post_norm, 1.0e-3)

    def _router_norm(self, current_raw: Tensor) -> Tensor:
        """Normalize the router input, affine only when configured.

        The affine-free variant keeps ``router_norm`` in the graph through a
        zero-valued term so the parameter still receives a gradient; the
        reference implementation does the same.
        """
        if self.router_norm_affine:
            return F.rms_norm(current_raw, (current_raw.shape[-1],), weight=self.router_norm, eps=self.eps)
        return F.rms_norm(current_raw, (current_raw.shape[-1],), eps=self.eps) + self.router_norm.sum() * 0.0

    @torch.no_grad()
    def init_extra_weights(self) -> set[str]:
        """Initialise the route parameters the generic initializer must not own.

        ``router_w1``/``router_w2`` are part of the MUDD contract (``router_w2`` must be
        zero so the zero-initialised route is exactly the identity), so they are asserted
        here rather than left to the generic walk.

        Returns:
            set[str]: Names of initialized non-standard parameters.
        """

        def init_static_a(param: Tensor) -> None:
            nn.init.zeros_(param)

        def init_router_w1(param: Tensor) -> None:
            nn.init.normal_(param, mean=0.0, std=1.0 / self.hidden_size**0.5)

        def init_post_norm(param: Tensor) -> None:
            nn.init.constant_(param, 1.0e-3)

        init_params(self.static_a, init_static_a)
        init_params(self.router_w1, init_router_w1)
        init_params(self.router_w2, nn.init.zeros_)
        init_params(self.router_norm, nn.init.ones_)
        init_params(self.source_norm, nn.init.ones_)
        init_params(self.post_norm, init_post_norm)
        return {"static_a", "router_w1", "router_w2", "router_norm", "source_norm", "post_norm"}

    def _route_weights(self, current_raw: Tensor) -> Tensor:
        """Route coefficients for every (stream, source) pair."""
        weights = F.linear(F.gelu(F.linear(self._router_norm(current_raw), self.router_w1)), self.router_w2)
        return weights.view(*weights.shape[:-1], self.num_streams, self.num_sources) + self.static_a

    def forward(self, current_raw: Tensor, sources: list[Tensor]) -> tuple[Tensor, ...]:
        if len(sources) != self.num_sources:
            raise RuntimeError(f"MUDD expects {self.num_sources} sources, got {len(sources)}")

        if self.route_kernel and _HAS_TRITON and current_raw.is_cuda:
            # Endpoints are handed over one by one so the kernel wrapper never has to
            # retain a stacked copy. The kernel still carries a non-PrePostDANorm
            # branch from when the route had two contracts; it is unreachable now.
            fused = _MUDDCombineTriton.apply(
                current_raw,
                self._route_weights(current_raw),
                self.source_norm,
                self.post_norm,
                self.eps,
                True,
                *sources,
            )
            return tuple(fused.unbind(dim=-2))

        if self.compile_route:
            route_fn = _compiled_route(
                self.num_sources,
                self.num_streams,
                self.router_norm_affine,
                self.hidden_size,
                self.eps,
            )
            fused = _invoke_compiled_route(
                route_fn,
                current_raw,
                self.router_norm,
                self.router_w1,
                self.router_w2,
                self.static_a,
                self.source_norm,
                self.post_norm,
                *sources,
            )
            return tuple(fused.unbind(dim=-2))

        # Fused eager path: stack sources once (GEMM-ready [..., N, H] layout,
        # so the einsum feeds the batched GEMM without a permute copy) and run
        # one rms_norm plus one einsum instead of O(num_sources) small kernels.
        stacked = torch.stack(sources, dim=-2)
        normed = F.rms_norm(stacked, (stacked.shape[-1],), weight=self.source_norm, eps=self.eps)
        updates = torch.einsum("...sn,...nh->...sh", self._route_weights(current_raw), normed)
        updates = F.rms_norm(updates, (updates.shape[-1],), eps=self.eps)
        return tuple((current_raw.unsqueeze(-2) + updates * self.post_norm).unbind(dim=-2))


class Qwen35MUDD(nn.Module):
    """Base full-history Qwen3.5 MUDD residual route.

    Args:
        hidden_size (int): Hidden dimension.
        num_layers (int): Number of decoder layers.
        eps (float): RMS normalization epsilon.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        eps: float,
        *,
        compile_route: bool = False,
        route_kernel: bool = False,
        router_norm_affine: bool = False,
    ) -> None:
        super().__init__()
        self.routes = nn.ModuleList(
            [
                Qwen35MUDDRoute(
                    hidden_size,
                    layer_idx + 2,
                    eps,
                    compile_route=compile_route,
                    route_kernel=route_kernel,
                    router_norm_affine=router_norm_affine,
                )
                for layer_idx in range(int(num_layers))
            ]
        )

    def initial_state(self, decoder_input: Tensor) -> Qwen35MUDDState:
        """Create the initial route state.

        Args:
            decoder_input (Tensor): Decoder input embeddings.

        Returns:
            Qwen35MUDDState: Initial full-history state.
        """
        return Qwen35MUDDState(history=[decoder_input], streams=(decoder_input,) * 4)

    def finish_layer(self, layer_idx: int, current_raw: Tensor, state: Qwen35MUDDState) -> Qwen35MUDDState:
        """Route one completed decoder layer.

        Args:
            layer_idx (int): Decoder layer index.
            current_raw (Tensor): Raw decoder layer output.
            state (Qwen35MUDDState): Previous route state.

        Returns:
            Qwen35MUDDState: Updated route state.
        """
        sources = [*state.history, current_raw]
        state.streams = self.routes[layer_idx](current_raw, sources)
        state.history.append(current_raw)
        return state

    @staticmethod
    def final_read(state: Qwen35MUDDState) -> Tensor:
        """Read the residual stream used by the next decoder layer.

        Args:
            state (Qwen35MUDDState): Current route state.

        Returns:
            Tensor: Fourth MUDD stream.
        """
        return state.streams[3]
