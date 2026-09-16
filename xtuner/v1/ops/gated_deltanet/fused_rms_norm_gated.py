# Copyright (c) OpenMMLab. All rights reserved.
"""FSDP-aware FLA ``FusedRMSNormGated`` shared by GatedDeltaNet and KDA."""

from __future__ import annotations

import torch

_fused_rms_norm_gated_import_error: BaseException | None = None
FusedRMSNormGated = None  # type: ignore[assignment,misc]
has_fused_rms_norm_gated = False

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
