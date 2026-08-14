import inspect
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.optim.adamw import adamw


_INIT_GROUP_TAKES_AMSGRAD = "amsgrad" in inspect.signature(torch.optim.AdamW._init_group).parameters


class LocalDTensorAdamW(torch.optim.AdamW):
    """Fused AdamW that dispatches directly on DTensor local shards.

    Parameters and optimizer states remain DTensors owned by the regular
    ``torch.optim.AdamW`` implementation. Only the tensor lists passed to the
    fused pointwise kernel are unwrapped to aliasing local views, avoiding the
    per-step DTensor operator-propagation overhead without changing state-dict
    structure.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._validate_fused_groups()

    @staticmethod
    def _local(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _validate_fused_groups(self) -> None:
        if not all(group.get("fused") is True for group in self.param_groups):
            raise ValueError("LocalDTensorAdamW requires fused=True for every parameter group")

    def load_state_dict(self, state_dict):
        param_groups = state_dict.get("param_groups")
        if param_groups is not None and not all(group.get("fused") is True for group in param_groups):
            raise ValueError("LocalDTensorAdamW requires fused=True for every parameter group")
        result = super().load_state_dict(state_dict)
        self._validate_fused_groups()
        return result

    @torch.no_grad()
    def step(self, closure=None):
        self._validate_fused_groups()
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            beta1, beta2 = group["betas"]

            init_group_args = (
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            if _INIT_GROUP_TAKES_AMSGRAD:
                # PyTorch 2.6/2.7 include ``amsgrad`` before the state lists.
                init_group_args = (*init_group_args[:3], group["amsgrad"], *init_group_args[3:])
            has_complex = self._init_group(*init_group_args)
            adamw(
                [self._local(tensor) for tensor in params_with_grad],
                [self._local(tensor) for tensor in grads],
                [self._local(tensor) for tensor in exp_avgs],
                [self._local(tensor) for tensor in exp_avg_sqs],
                [self._local(tensor) for tensor in max_exp_avg_sqs],
                [self._local(tensor) for tensor in state_steps],
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=self._local(getattr(self, "grad_scale", None)),
                found_inf=self._local(getattr(self, "found_inf", None)),
            )

        return loss
