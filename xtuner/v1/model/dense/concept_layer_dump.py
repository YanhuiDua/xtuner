"""Dump per-layer hidden states and attention outputs for Megatron alignment.

Enabled only when CONCEPT_LAYER_DUMP=1. Rank 0 saves the first microbatch of each of the first CONCEPT_LAYER_DUMP_STEPS
optimizer steps (default 5).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


_ENABLED = os.environ.get("CONCEPT_LAYER_DUMP", "0").strip().lower() in {"1", "true", "yes", "on"}
_TAG = os.environ.get("CONCEPT_LAYER_DUMP_TAG", "xtuner_fix")
_DIR = Path(
    os.environ.get(
        "CONCEPT_LAYER_DUMP_DIR",
        f"/mnt/shared-storage-user/llmrazor-share/heqiushi/concept_olmo/layer_dump_dd5/{_TAG}",
    )
)
_ACCUM = int(os.environ.get("CONCEPT_LAYER_DUMP_ACCUM", "64"))
_MAX_STEPS = int(os.environ.get("CONCEPT_LAYER_DUMP_STEPS", "5"))


class _State:
    forward_index = 0
    capturing = False
    step = 0
    tensors: dict[str, torch.Tensor] = {}
    stats: dict[str, dict] = {}


_state = _State()


def _rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def begin_forward() -> None:
    if not _ENABLED:
        return
    _state.forward_index += 1
    if not _rank0():
        _state.capturing = False
        return
    step = (_state.forward_index - 1) // _ACCUM
    micro = (_state.forward_index - 1) % _ACCUM
    _state.capturing = step < _MAX_STEPS and micro == 0
    _state.step = step
    _state.tensors = {}
    _state.stats = {}


def end_forward() -> None:
    if not _state.capturing:
        return
    _state.capturing = False
    _DIR.mkdir(parents=True, exist_ok=True)
    step = _state.step
    torch.save(_state.tensors, _DIR / f"step_{step:02d}.pt")
    (_DIR / f"step_{step:02d}_stats.json").write_text(json.dumps(_state.stats, indent=2, sort_keys=True))
    print(f"[concept_layer_dump] wrote {_DIR / f'step_{step:02d}.pt'} ({len(_state.tensors)} tensors)", flush=True)
    _state.tensors = {}
    _state.stats = {}


def record(name: str | None, kind: str, tensor: torch.Tensor | None) -> None:
    if not _state.capturing or name is None or tensor is None:
        return
    if torch.compiler.is_compiling():
        return
    value = tensor.detach()
    if value.ndim == 3:
        if value.shape[0] == 1:
            value = value[0]
        elif value.shape[1] == 1:
            value = value[:, 0]
    flat = value.float()
    key = f"{name}.{kind}"
    _state.stats[key] = {
        "shape": list(value.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "absmax": float(flat.abs().max()),
    }
    _state.tensors[key] = value.to(dtype=torch.bfloat16, device="cpu")


def tag_stages(encoder_stages, concept_stages, decoder_stages) -> None:
    for prefix, stages in (
        ("encoder", encoder_stages),
        ("concept", concept_stages),
        ("decoder", decoder_stages),
    ):
        for index, stage in enumerate(stages):
            name = f"{prefix}.{index}"
            stage.dump_name = name
            stage.transformer.concept_dump_name = name
