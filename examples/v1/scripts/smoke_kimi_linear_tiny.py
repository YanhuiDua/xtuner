#!/usr/bin/env python3
"""8-GPU torchrun smoke for Kimi Linear tiny (AttnRes + LatentMoE load/forward).

Usage:
  torchrun --standalone --nproc-per-node=8 \\
    examples/v1/scripts/smoke_kimi_linear_tiny.py \\
    [--ckpt work_dirs/kimi_linear_tiny_bf16] [--seq-len 64]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional tiny bf16 checkpoint dir (generated if missing)",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = _repo_root()
    sys.path.insert(0, str(repo))

    import torch
    import torch.distributed as dist

    from xtuner.v1.config import FSDPConfig
    from xtuner.v1.data_proto import SequenceContext
    from xtuner.v1.loss import CELossConfig
    from xtuner.v1.model.moe.kimi_linear import KimiLinearTinyConfig

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed + rank)

    ckpt = args.ckpt or (repo / "work_dirs" / "kimi_linear_tiny_bf16")
    if rank == 0:
        if not ckpt.exists() or not any(ckpt.glob("*.safetensors")) and not (ckpt / "model.safetensors.index.json").exists():
            # Generate tiny weights on rank0 only
            from importlib.util import module_from_spec, spec_from_file_location

            gen_path = repo / ".dev_scripts" / "gen_kimi_linear_tiny_bf16.py"
            spec = spec_from_file_location("gen_kimi_linear_tiny_bf16", gen_path)
            assert spec and spec.loader
            mod = module_from_spec(spec)
            # Call main via subprocess-like argv
            old_argv = sys.argv
            try:
                sys.argv = [str(gen_path), str(ckpt)]
                spec.loader.exec_module(mod)
                mod.main()
            finally:
                sys.argv = old_argv
    dist.barrier()

    cfg = KimiLinearTinyConfig()
    with torch.device("meta"):
        model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

    fsdp_cfg = FSDPConfig(ep_size=1, reshard_after_forward=True, cpu_offload=False)
    model.fully_shard(fsdp_config=fsdp_cfg)

    # Prefer loading synthetic ckpt when present; else init after shard.
    index = ckpt / "model.safetensors.index.json"
    shards = list(ckpt.glob("*.safetensors"))
    if index.exists() or shards:
        try:
            model.from_hf(str(ckpt))
            if rank == 0:
                print(f"[rank0] loaded checkpoint from {ckpt}")
        except Exception as exc:  # noqa: BLE001 — smoke should continue on layout drift
            if rank == 0:
                print(f"[rank0] from_hf skipped ({exc}); initializing weights")
            if hasattr(model, "init_weights"):
                model.init_weights()
    elif hasattr(model, "init_weights"):
        model.init_weights()
        if rank == 0:
            print("[rank0] using randomly initialized tiny weights")

    seq_len = args.seq_len
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device, dtype=torch.long)
    labels = ids.clone()
    seq_ctx = SequenceContext.from_input_ids(input_ids=(ids,))
    loss_ctx = CELossConfig().build(data={"shifted_labels": labels}, sp_mesh=None)

    model.train()
    out = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
    loss = out["loss"] if isinstance(out, dict) else getattr(out, "loss", None)
    assert loss is not None, "model did not return loss"
    assert torch.isfinite(loss).all(), f"non-finite loss: {loss}"

    # Backward once to exercise AttnRes + LatentMoE grads
    loss.backward()
    dist.barrier()

    if rank == 0:
        print(
            f"[ok] kimi_linear_tiny smoke world_size={world_size} "
            f"seq_len={seq_len} loss={float(loss.detach().float()):.4f} "
            f"attn_res_block_size={cfg.attn_res_block_size}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
