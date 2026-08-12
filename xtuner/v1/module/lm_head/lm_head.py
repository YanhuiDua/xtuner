from typing import Any, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from typing_extensions import overload

from xtuner.v1.loss.ce_loss import LMHeadLossContext


Loss: TypeAlias = torch.Tensor
Logits: TypeAlias = torch.Tensor
Weight: TypeAlias = torch.Tensor | DTensor
Bias: TypeAlias = torch.Tensor | DTensor | None
HiddenStates: TypeAlias = torch.Tensor
Labels: TypeAlias = torch.Tensor


class LMHead(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        logical_out_features: int | None = None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if logical_out_features is None:
            logical_out_features = out_features
        if not 0 < logical_out_features <= out_features:
            raise ValueError("logical_out_features must be positive and no larger than out_features")
        self.logical_out_features = logical_out_features

    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Logits | None, dict[str, Any]]]: ...

    @overload  # type: ignore[override]
    def forward(
        self, hidden_states: HiddenStates, loss_ctx: LMHeadLossContext
    ) -> tuple[Loss, tuple[Logits | None, dict[str, Any]]]: ...

    def forward(  # type: ignore[override]
        self, hidden_states: torch.Tensor, loss_ctx: LMHeadLossContext | None = None
    ) -> tuple[Loss | None, tuple[Logits | None, dict[str, Any]]]:
        """Forward pass of the language model head."""
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                assert isinstance(self.bias, DTensor), "Bias should be a DTensor if weight is a DTensor"
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        if loss_ctx is None:
            logits = F.linear(hidden_states, w, b)
            logits = logits[..., : self.logical_out_features]
            return None, (logits.float(), {})
        elif self.logical_out_features < self.out_features:
            if loss_ctx.__class__ is not LMHeadLossContext:
                raise NotImplementedError(
                    "Padded LM heads currently support only the plain CE loss context"
                )
            return loss_ctx.forward(
                hidden_states,
                w,
                b,
                valid_vocab_size=self.logical_out_features,
            )
        else:
            return loss_ctx.forward(hidden_states, w, b)

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: None = None
    ) -> tuple[None, tuple[Logits | None, dict[str, Any]]]: ...

    @overload  # type: ignore
    def __call__(
        self, hidden_states: HiddenStates, loss_ctx: LMHeadLossContext
    ) -> tuple[Loss, tuple[Logits | None, dict[str, Any]]]: ...

    __call__ = nn.Module.__call__
