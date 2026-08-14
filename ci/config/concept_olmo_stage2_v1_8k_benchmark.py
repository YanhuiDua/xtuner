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
#   FUSED_ADAMW=1  # use torch's fused AdamW implementation
#   FUSED_ADAMW=1 FUSED_GRAD_CLIP=1  # fold clipping and finite checks into fused AdamW
#   FUSED_ADAMW=1 LOCAL_DTENSOR_ADAMW=1  # bypass DTensor dispatch around fused AdamW
#   LOSS_MODE=eager  # eager, chunk, or liger; liger requires LM_HEAD_VOCAB_PADDING_MULTIPLE=1
#   STAGE_GROUP_SIZE=4 RECOMPUTE_RATIO=0 RESHARD_AFTER_FORWARD=0
#       compile/shard four consecutive stages as one unit
#   FLOAT8_MODE=tensorwise PEAK_TFLOPS_PER_GPU=1979  # H200 dense FP8 peak
#
# Required environment variables:
#
#   CONCEPT_OLMO_STAGE2_V1_PATH=/path/to/concept_olmo_stage_2_V1
#   ALPACA_PATH=/path/to/alpaca
import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
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
FUSED_ADAMW = _parse_bool_env("FUSED_ADAMW", "0")
FUSED_GRAD_CLIP = _parse_bool_env("FUSED_GRAD_CLIP", "0")
LOCAL_DTENSOR_ADAMW = _parse_bool_env("LOCAL_DTENSOR_ADAMW", "0")
LOSS_MODE = os.getenv("LOSS_MODE", "eager")
LM_HEAD_VOCAB_PADDING_MULTIPLE = int(os.getenv("LM_HEAD_VOCAB_PADDING_MULTIPLE", "8"))
if LM_HEAD_VOCAB_PADDING_MULTIPLE <= 0:
    raise ValueError("LM_HEAD_VOCAB_PADDING_MULTIPLE must be positive")
STAGE_GROUP_SIZE = int(os.getenv("STAGE_GROUP_SIZE", "1"))
if STAGE_GROUP_SIZE <= 0:
    raise ValueError("STAGE_GROUP_SIZE must be positive")
if STAGE_GROUP_SIZE > 1 and (RECOMPUTE_RATIO != 0 or RESHARD_AFTER_FORWARD):
    raise ValueError("STAGE_GROUP_SIZE > 1 requires RECOMPUTE_RATIO=0 and RESHARD_AFTER_FORWARD=0")
if (FUSED_GRAD_CLIP or LOCAL_DTENSOR_ADAMW) and not FUSED_ADAMW:
    raise ValueError("FUSED_GRAD_CLIP and LOCAL_DTENSOR_ADAMW require FUSED_ADAMW=1")
FLOAT8_MODE = os.getenv("FLOAT8_MODE", "none").lower()
float8_cfg = {
    "none": None,
    "tensorwise": Float8Config(scaling_granularity_gemm=ScalingGranularity.TENSORWISE),
}.get(FLOAT8_MODE)
if FLOAT8_MODE not in {"none", "tensorwise"}:
    raise ValueError("FLOAT8_MODE must be one of: none, tensorwise")

_peak_tflops_override = os.getenv("PEAK_TFLOPS_PER_GPU")
PEAK_TFLOPS_PER_GPU = float(_peak_tflops_override) if _peak_tflops_override else None
if float8_cfg is not None and PEAK_TFLOPS_PER_GPU is None:
    raise ValueError("FP8 training requires PEAK_TFLOPS_PER_GPU for a valid MFU denominator")
WORK_DIR = os.getenv(
    "WORK_DIR",
    f"/tmp/concept_olmo_stage2_v1_8k_compile_{'on' if TORCH_COMPILE else 'off'}",
)

model_cfg = ConceptOlmoStage2V1Config(
    compile_cfg=None if TORCH_COMPILE else False,
    lm_head_vocab_padding_multiple=LM_HEAD_VOCAB_PADDING_MULTIPLE,
    stage_group_size=STAGE_GROUP_SIZE,
    float8_cfg=float8_cfg,
)
optim_cfg = AdamWConfig(
    lr=6e-5,
    fused=True if FUSED_ADAMW else None,
    fused_grad_clip=FUSED_GRAD_CLIP,
    local_dtensor_step=LOCAL_DTENSOR_ADAMW,
)
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
    loss_cfg=CELossConfig(mode=LOSS_MODE),
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
