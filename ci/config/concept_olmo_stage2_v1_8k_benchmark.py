# Single-node 8-GPU throughput benchmark for Concept OLMo Stage-2 V1.
#
# Each data-parallel rank receives one continuous 8192-token model sequence.
# Sequence parallelism is disabled. Toggle torch.compile with:
#
#   TORCH_COMPILE=1  # compile on (default)
#   TORCH_COMPILE=0  # compile off
#   RECOMPUTE_RATIO=1.0  # full activation recompute (default)
#   RECOMPUTE_RATIO=0.0  # disable activation recompute
#   RESHARD_AFTER_FORWARD=0  # retain full parameters through backward
#   PROFILE_STEP=18  # profile one training step (disabled by default)
#   PROFILE_TIME=1  # enable the time profiler (default)
#   PROFILE_MEMORY=0  # disable the memory profiler (default)
#   SKIP_SAVE=1  # skip both HuggingFace and DCP checkpoint exports
#   LM_HEAD_VOCAB_PADDING_MULTIPLE=8  # align the physical LM-head vocabulary (default)
#
# Required environment variables:
#
#   CONCEPT_OLMO_STAGE2_V1_PATH=/path/to/concept_olmo_stage_2_V1
#   ALPACA_PATH=/path/to/alpaca
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import ConceptOlmoStage2V1Config
from xtuner.v1.train import MFUConfig, TrainerConfig


def _parse_bool_env(name: str, default: str) -> bool:
    value = os.getenv(name, default).lower()
    if value not in {"0", "1", "false", "true", "off", "on"}:
        raise ValueError(f"{name} must be one of: 0, 1, false, true, off, on")
    return value in {"1", "true", "on"}


CONCEPT_OLMO_STAGE2_V1_PATH = os.environ["CONCEPT_OLMO_STAGE2_V1_PATH"]
ALPACA_PATH = os.environ["ALPACA_PATH"]
TOTAL_STEP = int(os.getenv("TOTAL_STEP", "20"))
RECOMPUTE_RATIO = float(os.getenv("RECOMPUTE_RATIO", "1.0"))
if not 0.0 <= RECOMPUTE_RATIO <= 1.0:
    raise ValueError("RECOMPUTE_RATIO must be between 0.0 and 1.0")
_profile_step_value = os.getenv("PROFILE_STEP")
PROFILE_STEP = int(_profile_step_value) if _profile_step_value else None
if PROFILE_STEP is not None and PROFILE_STEP <= 0:
    raise ValueError("PROFILE_STEP must be positive")
# Trainer checks its zero-based current-step counter before incrementing it for
# logging, so convert the user-facing one-based step here.
TRAINER_PROFILE_STEP = PROFILE_STEP - 1 if PROFILE_STEP is not None else None
SEQUENCE_LENGTH = 8192
# The causal LM collator consumes one source token while shifting labels, so
# 8193 source tokens produce an exact 8192-token model sequence.
SOURCE_PACK_LENGTH = SEQUENCE_LENGTH + 1

TORCH_COMPILE = _parse_bool_env("TORCH_COMPILE", "1")
RESHARD_AFTER_FORWARD = _parse_bool_env("RESHARD_AFTER_FORWARD", "1")
PROFILE_TIME = _parse_bool_env("PROFILE_TIME", "1")
PROFILE_MEMORY = _parse_bool_env("PROFILE_MEMORY", "0")
SKIP_SAVE = _parse_bool_env("SKIP_SAVE", "0")
LM_HEAD_VOCAB_PADDING_MULTIPLE = int(os.getenv("LM_HEAD_VOCAB_PADDING_MULTIPLE", "8"))
if LM_HEAD_VOCAB_PADDING_MULTIPLE <= 0:
    raise ValueError("LM_HEAD_VOCAB_PADDING_MULTIPLE must be positive")

_peak_tflops_override = os.getenv("PEAK_TFLOPS_PER_GPU")
PEAK_TFLOPS_PER_GPU = float(_peak_tflops_override) if _peak_tflops_override else None
WORK_DIR = os.getenv(
    "WORK_DIR",
    f"/tmp/concept_olmo_stage2_v1_8k_compile_{'on' if TORCH_COMPILE else 'off'}",
)

model_cfg = ConceptOlmoStage2V1Config(
    compile_cfg=None if TORCH_COMPILE else False,
    lm_head_vocab_padding_multiple=LM_HEAD_VOCAB_PADDING_MULTIPLE,
)
optim_cfg = AdamWConfig(lr=6e-5)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    recompute_ratio=RECOMPUTE_RATIO,
    reshard_after_forward=RESHARD_AFTER_FORWARD,
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

trainer = TrainerConfig(
    load_from=CONCEPT_OLMO_STAGE2_V1_PATH,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(),
    tokenizer_path=CONCEPT_OLMO_STAGE2_V1_PATH,
    # With 8 GPUs, SP=1 and one sequence per rank, global batch size is 8.
    global_batch_size=None,
    intra_layer_micro_batch=1,
    sp_size=1,
    total_step=TOTAL_STEP,
    work_dir=WORK_DIR,
    profile_step=TRAINER_PROFILE_STEP,
    profile_time=PROFILE_TIME,
    profile_memory=PROFILE_MEMORY,
    debug_skip_save=SKIP_SAVE,
    seed=0,
    mfu_cfg=MFUConfig(
        sequence_length=SEQUENCE_LENGTH,
        peak_tflops_per_gpu=PEAK_TFLOPS_PER_GPU,
    ),
)
