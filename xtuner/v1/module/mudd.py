"""Qwen3.5 four-way full-history MUDD routing for dense training."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xtuner.v1.utils import init_params


@dataclass
class Qwen35MUDDState:
    history: list[Tensor]
    streams: tuple[Tensor, Tensor, Tensor, Tensor]


class Qwen35MUDDRoute(nn.Module):
    """Route all preceding layer endpoints into four residual streams.

    Args:
        hidden_size (int): Hidden dimension.
        num_sources (int): Number of available history endpoints.
        eps (float): RMS normalization epsilon.
    """

    num_streams = 4

    def __init__(self, hidden_size: int, num_sources: int, eps: float) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_sources = int(num_sources)
        self.eps = float(eps)
        route_width = self.num_streams * self.num_sources
        self.w1 = nn.Linear(self.hidden_size, route_width, bias=False)
        self.w2 = nn.Linear(route_width, route_width, bias=False)
        self.static_a = nn.Parameter(torch.zeros(self.num_streams, self.num_sources))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w1.weight, mean=0.0, std=1.0 / self.hidden_size**0.5)
        nn.init.zeros_(self.w2.weight)
        with torch.no_grad():
            self.static_a.zero_()
            self.static_a[:, -1] = 1.0

    @torch.no_grad()
    def init_extra_weights(self) -> set[str]:
        """Initialize the identity route parameter.

        Returns:
            set[str]: Names of initialized non-standard parameters.
        """

        def init_static_a(param: Tensor) -> None:
            nn.init.zeros_(param)
            param[:, -1] = 1.0

        init_params(self.static_a, init_static_a)
        return {"static_a"}

    def forward(self, current_raw: Tensor, sources: list[Tensor]) -> tuple[Tensor, ...]:
        if len(sources) != self.num_sources:
            raise RuntimeError(f"MUDD expects {self.num_sources} sources, got {len(sources)}")
        route_input = F.rms_norm(current_raw, (current_raw.shape[-1],), eps=self.eps)
        weights = self.w2(F.gelu(self.w1(route_input)))
        weights = weights.view(*weights.shape[:-1], self.num_streams, self.num_sources)
        weights = weights + self.static_a.view(1, 1, self.num_streams, self.num_sources)
        return tuple(
            sum(source * weights[..., way, source_idx : source_idx + 1] for source_idx, source in enumerate(sources))
            for way in range(self.num_streams)
        )


class Qwen35MUDD(nn.Module):
    """Base full-history Qwen3.5 MUDD residual route.

    Args:
        hidden_size (int): Hidden dimension.
        num_layers (int): Number of decoder layers.
        eps (float): RMS normalization epsilon.
    """

    def __init__(self, hidden_size: int, num_layers: int, eps: float) -> None:
        super().__init__()
        self.routes = nn.ModuleList(
            [Qwen35MUDDRoute(hidden_size, layer_idx + 2, eps) for layer_idx in range(int(num_layers))]
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
