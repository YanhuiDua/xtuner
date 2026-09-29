# Copyright (c) OpenMMLab. All rights reserved.
"""FSDP-aware FLA ``FusedRMSNormGated`` shared by GatedDeltaNet and KDA."""

from __future__ import annotations

import torch
import torch.nn as nn


_fused_rms_norm_gated_import_error: BaseException | None = None
has_fused_rms_norm_gated = False


class FusedRMSNormGated(nn.Module):
    """Typed stand-in used when ``fla`` is not installed.

    The ``try`` block below replaces this class with the FLA subclass. Keeping a
    real class here lets mypy treat the name as callable and as an ``isinstance``
    target; assigning ``None`` made every later attribute in ``__init__`` untyped.
    """

    weight: nn.Parameter
    bias: nn.Parameter | None
    eps: float
    activation: str

    def __init__(self, hidden_size: int, eps: float = 1e-5, activation: str = "silu") -> None:
        super().__init__()
        del hidden_size
        self.eps = eps
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del x, g, residual, prenorm, residual_in_fp32
        raise RuntimeError("Please install fla (`pip install -U fla-core`)")


try:
    from fla.modules import FusedRMSNormGated as FLA_FusedRMSNormGated

    from .dtensor import dtensor_to_local
    from .rms_norm_gated import rms_norm_gated

    class FusedRMSNormGated(FLA_FusedRMSNormGated):  # type: ignore[no-redef]
        def forward(
            self,
            x: torch.Tensor,
            g: torch.Tensor,
            residual: torch.Tensor | None = None,
            prenorm: bool = False,
            residual_in_fp32: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            weight = dtensor_to_local(self.weight)
            return rms_norm_gated(
                x,
                g,
                weight,
                self.bias,
                self.activation,
                residual=residual,
                eps=self.eps,
                prenorm=prenorm,
                residual_in_fp32=residual_in_fp32,
            )

    has_fused_rms_norm_gated = True
except (ImportError, ModuleNotFoundError) as e:
    _fused_rms_norm_gated_import_error = e
