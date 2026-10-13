#!/usr/bin/env python3
"""Generate a tiny bf16 Kimi Linear checkpoint for load / smoke tests.

Usage:
  PYTHONPATH=. python .dev_scripts/gen_kimi_linear_tiny_bf16.py [out_dir]
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "work_dirs" / "kimi_linear_tiny_bf16"
    out_dir.mkdir(parents=True, exist_ok=True)

    from xtuner.v1.model.moe.kimi_linear import KimiLinearTinyConfig

    cfg = KimiLinearTinyConfig()
    config_source = repo / ".cache" / "kimi_k3_hf" / "configuration_kimi_k3.py"
    if not config_source.is_file():
        raise FileNotFoundError(f"Missing official Kimi config implementation: {config_source}")
    spec = importlib.util.spec_from_file_location("_xtuner_kimi_k3_config", config_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official Kimi config implementation: {config_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg._hf_config_cls = module.KimiLinearConfig
    # Prefer CPU meta-free init for tiny model
    model = cfg.build()
    model.to(dtype=torch.bfloat16)
    model.init_weights() if hasattr(model, "init_weights") else None

    # Materialize parameters if still on meta
    for p in model.parameters():
        if p.device.type == "meta":
            raise RuntimeError("Model still on meta device; build without meta context")

    if hasattr(model, "save_hf"):
        model.save_hf(str(out_dir))
    else:
        # Fallback: dump state dict
        torch.save(model.state_dict(), out_dir / "pytorch_model.bin")

    # Copy remote-code stubs so trust_remote_code loaders can resolve model_type
    src = repo / ".cache" / "kimi_k3_hf"
    for name in (
        "configuration_kimi_k3.py",
        "modeling_kimi_linear.py",
        "tokenization_kimi.py",
        "tokenizer_config.json",
        "tiktoken.model",
    ):
        p = src / name
        if p.exists():
            shutil.copy2(p, out_dir / name)

    print(f"Wrote tiny bf16 Kimi Linear checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
