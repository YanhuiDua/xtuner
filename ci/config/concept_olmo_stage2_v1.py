# Smoke SFT config for Concept OLMo Stage-2 V1.
#
# Usage (single node, 8 GPUs):
#
#   export CONCEPT_OLMO_STAGE2_V1_PATH=/path/to/concept_olmo_stage_2_V1
#   export ALPACA_PATH=/path/to/alpaca
#   torchrun --nproc-per-node=8 -m xtuner.v1.train.cli.sft \
#       --config ci/config/concept_olmo_stage2_v1.py
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import ConceptOlmoStage2V1Config
from xtuner.v1.train import TrainerConfig


CONCEPT_OLMO_STAGE2_V1_PATH = os.environ["CONCEPT_OLMO_STAGE2_V1_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]

model_cfg = ConceptOlmoStage2V1Config(compile_cfg=False)
optim_cfg = AdamWConfig(lr=6e-5)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(max_length=8192),
    },
]
dataloader_config = DataloaderConfig(pack_max_length=8192)
loss_cfg = CELossConfig()

trainer = TrainerConfig(
    load_from=CONCEPT_OLMO_STAGE2_V1_PATH,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=CONCEPT_OLMO_STAGE2_V1_PATH,
    global_batch_size=16,
    total_epoch=1,
    work_dir="/tmp/concept_olmo_stage2_v1",
    seed=0,
)
