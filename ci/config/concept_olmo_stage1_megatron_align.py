# Concept-OLMo Stage-1 pretrain config aligned to the Megatron 8-GPU recipe
# in concept_v22_vq_olmo3_config.sh / pretrain_concept_olmo_stage_1.sh.
#
# This tree is layer-DD only (no cumsum). Pair LOAD_FROM with the dd dump:
#   .../data/megatron_init_hf_dd
#
# One optimizer step on both sides consumes:
#   GBS=512 sequences x SEQ_LENGTH=8192 tokens = 4,194,304 tokens
# 8 GPUs, SP=1, MBS=1 -> 64 microbatches/rank. Rank assignment matches
# MegatronPretrainingSampler (index % 8). Compare by step:
#   Megatron `lm loss`  (run with OLMO3_Z_LOSS_MULTIPLIER=0)
#   XTuner  `reduced_llm_loss`  (CELossConfig loss_reduction="token")
#
# Same samples as Megatron: dump first, then point MEGATRON_REPLAY_PATH at it.
# Same init weights: dump Megatron init to HF, then set LOAD_FROM:
#   LOAD_FROM=.../data/megatron_init_hf_dd bash examples/v1/scripts/run_concept_olmo_stage1_8gpu.sh
#
# Optional:
#   CONCEPT_OLMO_STAGE2_V1_PATH / LOAD_FROM / TOKENIZER_PATH
#   TOTAL_STEP GLOBAL_BATCH_SIZE SEQ_LENGTH LR MIN_LR WARMUP_ITERS WORK_DIR
#   HLM_LOSS_WEIGHT VQ_LOSS_WEIGHT  (default 1.0; model class default is 0)
import os
from pathlib import Path

import torch

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss.ce_loss import CELossConfig
from xtuner.v1.model import ConceptOlmoStage2V1Config
from xtuner.v1.train import MFUConfig, TrainerConfig


def _parse_bool_env(name: str, default: str) -> bool:
    value = os.getenv(name, default).lower()
    if value not in {"0", "1", "false", "true", "off", "on"}:
        raise ValueError(f"{name} must be one of: 0, 1, false, true, off, on")
    return value in {"1", "true", "on"}


TOTAL_STEP = int(os.getenv("TOTAL_STEP", "100"))
SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", "8192"))
# Causal collator drops one source token. 8193 indexed tokens -> 8192 model tokens,
# matching Megatron --seq-length 8192.
SOURCE_LENGTH = SEQ_LENGTH + 1
GLOBAL_BATCH_SIZE = int(os.getenv("GLOBAL_BATCH_SIZE", "512"))
INTRA_LAYER_MICRO_BATCH = int(os.getenv("INTRA_LAYER_MICRO_BATCH", "1"))
SP_SIZE = int(os.getenv("SP_SIZE", "1"))
LR = float(os.getenv("LR", "6.0e-5"))
MIN_LR = float(os.getenv("MIN_LR", "6.0e-6"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.1"))
CLIP_GRAD = float(os.getenv("CLIP_GRAD", "1.0"))
ADAM_BETA1 = float(os.getenv("ADAM_BETA1", "0.9"))
ADAM_BETA2 = float(os.getenv("ADAM_BETA2", "0.95"))
ADAM_EPS = float(os.getenv("ADAM_EPS", "1.0e-8"))
# Megatron --muon-momentum 0.9, --muon-scale-mode spectral, nesterov off.
# rms_norm with extra_scale_factor=1 is sqrt(max(fan_out, fan_in)). QKV is split per head.
MUON_MOMENTUM = float(os.getenv("MUON_MOMENTUM", "0.9"))

# Megatron uses absolute warmup iters and asserts warmup < decay. Mirror that:
# full recipe 2000; smoke TRAIN_ITERS=100 clamps to 50.
_warmup_default = 2000
if TOTAL_STEP <= 1:
    _warmup_default = 0
elif TOTAL_STEP <= 2000:
    _warmup_default = 50 if TOTAL_STEP > 50 else max(TOTAL_STEP - 1, 0)
WARMUP_ITERS = int(os.getenv("WARMUP_ITERS", str(_warmup_default)))
if WARMUP_ITERS >= TOTAL_STEP:
    WARMUP_ITERS = 50 if TOTAL_STEP > 50 else max(TOTAL_STEP - 1, 0)

# MEGATRON_REPLAY_PATH is a dumped indexed prefix (no .bin/.idx suffix).
# Without it, XTuner cannot match Megatron blend/shuffle/document packing.
MEGATRON_REPLAY_PATH = os.getenv("MEGATRON_REPLAY_PATH")
DATA_PATH = MEGATRON_REPLAY_PATH or os.getenv(
    "DATA_PATH",
    "/mnt/shared-storage-gpfs2/plm-gpfs/public_data_model/data/dolma3_mix-150B-1025/megatron_olmo3_8192/data_args_path.txt",
)
if not MEGATRON_REPLAY_PATH:
    raise ValueError(
        "Set MEGATRON_REPLAY_PATH to the dump prefix from dump_megatron_align_samples.sh. "
        "Reading data_args_path.txt directly does not reproduce Megatron's blend, "
        "document packing, or shuffle_index (seed=1234)."
    )

LOAD_FROM = os.getenv("LOAD_FROM") or os.getenv("CONCEPT_OLMO_STAGE2_V1_PATH") or None
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH") or LOAD_FROM
WORK_DIR = os.getenv("WORK_DIR", "./work_dirs/concept_olmo_stage1_megatron_align_dd")
SKIP_SAVE = _parse_bool_env("SKIP_SAVE", "1")
TORCH_COMPILE = _parse_bool_env("TORCH_COMPILE", "1")
RESHARD_AFTER_FORWARD = _parse_bool_env("RESHARD_AFTER_FORWARD", "1")
RECOMPUTE_RATIO = float(os.getenv("RECOMPUTE_RATIO", "1.0"))
# Megatron default --seed 1234. With replay + shuffle=False this only affects init/RNG.
SEED = int(os.getenv("SEED", "1234"))
_peak_tflops_override = os.getenv("PEAK_TFLOPS_PER_GPU")
PEAK_TFLOPS_PER_GPU = float(_peak_tflops_override) if _peak_tflops_override else None

model_cfg = ConceptOlmoStage2V1Config(
    compile_cfg=False,
    concept_hlm_loss_weight=float(os.getenv("HLM_LOSS_WEIGHT", "1.0")),
    concept_vq_loss_weight=float(os.getenv("VQ_LOSS_WEIGHT", "1.0")),
)
optim_cfg = MuonConfig(
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=CLIP_GRAD,
    momentum=MUON_MOMENTUM,
    betas=(ADAM_BETA1, ADAM_BETA2),
    eps=ADAM_EPS,
    clip_grad_mode="all",
    adjust_lr="rms_norm",
    extra_scale_factor=1.0,
    nesterov=False,
    embedding_weight_decay=0.0,
    one_d_weight_decay=0.0,
)
# warmup_ratio >= 1 is treated as an absolute step count.
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=float(WARMUP_ITERS), lr_min=MIN_LR)
fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    recompute_ratio=RECOMPUTE_RATIO,
    reshard_after_forward=RESHARD_AFTER_FORWARD,
    gradient_accumulation_no_sync=True,
    reduce_dtype=torch.float32,
)

dataset_config = [
    {
        "dataset": DatasetConfig(
            name="megatron_replay",
            anno_path=DATA_PATH,
            class_name="MegatronIndexedDataset",
            # Dumped samples are already seq_length+1 tokens each. Do not recut.
            sequence_length=None,
            sample_ratio=1.0,
            enable_sequential_sampler=True,
            cache_dir=str(Path(WORK_DIR) / "data_cache"),
        ),
        "tokenize_fn": PretrainTokenizeFunctionConfig(
            add_eos_token=False,
            add_bos_token=False,
        ),
    }
]
dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    # Exact 8193-token dump samples: keep them, causal-shift to 8192, no pad.
    # pack_to_max_length must stay True so torch.compile is allowed; this
    # collator disables padding internally after the shift.
    collator="single_sequence_sft_llm_collator",
    pack_level="none",
    pack_max_length=SOURCE_LENGTH,
    pack_to_max_length=True,
    group_by_length=False,
    shuffle=False,
    num_workers=int(os.getenv("NUM_WORKERS", "0")),
)

trainer = TrainerConfig(
    load_from=LOAD_FROM,
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="eager", loss_reduction="token"),
    tokenizer_path=TOKENIZER_PATH,
    global_batch_size=GLOBAL_BATCH_SIZE,
    intra_layer_micro_batch=INTRA_LAYER_MICRO_BATCH,
    sp_size=SP_SIZE,
    total_step=TOTAL_STEP,
    work_dir=WORK_DIR,
    debug_skip_save=SKIP_SAVE,
    seed=SEED,
    mfu_cfg=MFUConfig(
        sequence_length=SEQ_LENGTH,
        peak_tflops_per_gpu=PEAK_TFLOPS_PER_GPU,
    ),
)
