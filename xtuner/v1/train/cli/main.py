import os
import ray
import argparse

import torch.distributed as dist
from xtuner.v1.train.rl_trainer import RLTrainer

from xtuner.v1.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Intern-S1 Training Script")
    parser.add_argument("cfg", type=str)
    parser.add_argument("--ray-cluster-url", type=str, default="")
    parser.add_argument("--work-dir", type=str, default="work_dir")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main(args):
    if not ray.is_initialized():
        master_addr = os.getenv("RAY_MASTER_ADDR", "127.0.0.1")
        client_port = os.getenv("RAY_CLIENT_PORT", "10001")
        ray_head_address = f"ray://{master_addr}:{client_port}"
        ray.init(address=ray_head_address)

    cfg = Config.fromfile(args.cfg)
    cfg.trainer.work_dir = args.work_dir
    cfg.trainer.resources.num_workers = args.num_workers
    trainer = RLTrainer.from_config(cfg.trainer)
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
