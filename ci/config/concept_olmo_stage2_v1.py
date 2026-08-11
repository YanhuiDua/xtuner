# SFT config for Concept OLMo Stage-2 V1, based on qwen3_moe_30BA3.py.
#
# Required environment variables:
#
#   export CONCEPT_OLMO_STAGE2_V1_PATH=/path/to/concept_olmo_stage_2_V1
#   export ALPACA_PATH=/path/to/openai_sft.jsonl
#
# Optional environment variables:
#
#   TOTAL_STEP=50 SP_SIZE=2 SEQUENCE_LENGTH=8192 \
#     WORK_DIR=/tmp/concept_olmo_stage2_v1_sft
#
# Launch from the XTuner repository root with:
#
#   torchrun --nproc-per-node=8 xtuner/v1/train/cli/sft.py \
#     --config ci/config/concept_olmo_stage2_v1.py
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import ConceptOlmoStage2V1Config
from xtuner.v1.train import MFUConfig, TrainerConfig


CONCEPT_OLMO_STAGE2_V1_PATH = os.environ["CONCEPT_OLMO_STAGE2_V1_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]
TOTAL_STEP = int(os.getenv("TOTAL_STEP", "50"))
SP_SIZE = int(os.getenv("SP_SIZE", "2"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "8192"))
# The SFT collator shifts labels by one token, so an exact 8192-token model
# sequence needs 8193 source tokens before the shift.
SOURCE_PACK_LENGTH = SEQUENCE_LENGTH + 1
# The trainer detects the accelerator's dense BF16/FP16 peak from its runtime
# device name. This optional override is useful for unknown or quantized GPUs.
_peak_tflops_override = os.getenv("PEAK_TFLOPS_PER_GPU")
PEAK_TFLOPS_PER_GPU = float(_peak_tflops_override) if _peak_tflops_override else None
WORK_DIR = os.getenv("WORK_DIR", "/tmp/concept_olmo_stage2_v1_sft")

model_cfg = ConceptOlmoStage2V1Config()
optim_cfg = AdamWConfig(lr=6e-5)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        "tokenize_fn": FTDPTokenizeFnConfig(chat_template="qwen", max_length=SOURCE_PACK_LENGTH),
    },
]
dataloader_config = DataloaderConfig(
    collator="single_sequence_sft_llm_collator",
    pack_level="hard",
    pack_max_length=SOURCE_PACK_LENGTH,
    pack_to_max_length=True,
)
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
    # Derive global batch size as dp_size * intra_layer_micro_batch. With
    # intra_layer_micro_batch=1, each DP rank receives one 8192-token sequence.
    global_batch_size=None,
    intra_layer_micro_batch=1,
    sp_size=SP_SIZE,
    total_step=TOTAL_STEP,
    work_dir=WORK_DIR,
    seed=0,
    mfu_cfg=MFUConfig(
        sequence_length=SEQUENCE_LENGTH,
        peak_tflops_per_gpu=PEAK_TFLOPS_PER_GPU,
    ),
)
