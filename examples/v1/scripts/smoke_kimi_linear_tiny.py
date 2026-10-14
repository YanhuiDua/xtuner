#!/usr/bin/env python3
"""8-GPU torchrun smoke for Kimi Linear tiny (AttnRes + LatentMoE next-token train).

Generate the tiny checkpoint in a single process before torchrun (save_hf must
not run inside an already-initialized process group):

  PYTHONPATH=. python .dev_scripts/gen_kimi_linear_tiny_bf16.py work_dirs/kimi_linear_tiny_bf16
  torchrun --standalone --nproc-per-node=8 \\
    examples/v1/scripts/smoke_kimi_linear_tiny.py \\
    [--ckpt work_dirs/kimi_linear_tiny_bf16] [--seq-len 64] [--steps 5]

Or use examples/v1/scripts/run_kimi_linear_tiny_8gpu.sh, which generates first.
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
        help="Tiny bf16 checkpoint dir; from_hf must succeed",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5, help="AdamW optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-3)
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
    from xtuner.v1.loss import CELossConfig, CELossContext
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

    cfg = KimiLinearTinyConfig()
    with torch.device("meta"):
        model = cfg.build()._to_device_dtype(dtype=torch.bfloat16, skip_buffers_dtype=True)

    fsdp_cfg = FSDPConfig(ep_size=1, reshard_after_forward=True, cpu_offload=False)
    model.fully_shard(fsdp_config=fsdp_cfg)

    model.from_hf(str(ckpt))
    if rank == 0:
        print(f"[rank0] loaded checkpoint from {ckpt}")

    seq_len = args.seq_len
    if seq_len < 2:
        raise ValueError("--seq-len must be >= 2 for next-token loss")

    # Match collator / HF causal LM: input t predicts token t+1.
    tokens = torch.randint(0, cfg.vocab_size, (1, seq_len + 1), device=device, dtype=torch.long)
    input_ids = tokens[:, :-1].contiguous()
    shifted_labels = tokens[:, 1:].contiguous()

    params = [p for _, p in model.trainable_parameters()]
    optim = torch.optim.AdamW(params, lr=args.lr, foreach=False)
    model.train()

    losses: list[float] = []
    for step in range(args.steps):
        seq_ctx = SequenceContext.from_input_ids(input_ids=(input_ids,))
        loss_ctx = CELossConfig().build(data={"shifted_labels": shifted_labels}, sp_mesh=None)
        assert loss_ctx is not None
        loss_ctx = CELossContext.build_batches([loss_ctx])[0]

        out = model(seq_ctx=seq_ctx, loss_ctx={"lm": loss_ctx})
        loss = out["loss"] if isinstance(out, dict) else getattr(out, "loss", None)
        assert loss is not None, "model did not return loss"
        assert torch.isfinite(loss).all(), f"non-finite loss at step {step}: {loss}"

        loss.backward()
        model.scale_and_reduce_grad()
        optim.step()
        optim.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().float()))
        if rank == 0:
            print(f"[rank0] step {step + 1}/{args.steps} loss={losses[-1]:.4f}")

    dist.barrier()

    if rank == 0:
        print(
            f"[ok] kimi_linear_tiny smoke world_size={world_size} "
            f"seq_len={seq_len} steps={args.steps} "
            f"loss_first={losses[0]:.4f} loss_last={losses[-1]:.4f} "
            f"attn_res_block_size={cfg.attn_res_block_size}"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
