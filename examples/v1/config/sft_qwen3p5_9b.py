"""Qwen3.5-9B dense pretraining and memory-smoke configuration."""

import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3_5_VLTextDense9BConfig
from xtuner.v1.module.mtp import MTPConfig
from xtuner.v1.train import TrainerConfig


def _bool_env(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


def _profile_step() -> list[int] | None:
    # Trainer compares this with its zero-based pre-step counter, so 4 profiles
    # the user-visible fifth optimization step.
    raw = os.environ.get("PROFILE_STEP", "4")
    return [int(step.strip()) for step in raw.split(",") if step.strip()] or None


repo_root = Path(__file__).resolve().parents[3]
work_dir = Path(
    os.environ.get(
        "WORK_DIR",
        repo_root / "work_dirs/qwen3p5_9b_8gpu_gbs16_20steps_step5_trace_ncp_mtp_on",
    )
)
data_path = Path(
    os.environ.get(
        "DATA_PATH",
        "/mnt/shared-storage-user/llmrazor-share/geruijun/ncp-dataset/megatron_qwen35_8192/indexed",
    )
)
seq_len = int(os.environ.get("SEQ_LEN", "8192"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))
hsdp_sharding_size_env = os.environ.get("HSDP_SHARDING_SIZE")
hsdp_sharding_size = int(hsdp_sharding_size_env) if hsdp_sharding_size_env else None
optimizer = os.environ.get("OPTIMIZER", "muon")
data_format = os.environ.get("DATA_FORMAT", "megatron_indexed")
model_compile = _bool_env("MODEL_COMPILE", False)
torch_compile_mode = os.environ.get("TORCH_COMPILE_MODE")
loss_mode = os.environ.get("LOSS_MODE", "chunk")
loss_reduction = os.environ.get("LOSS_REDUCTION", "square")

if model_compile and torch_compile_mode:
    compile_cfg = {
        "xtuner.v1.module.decoder_layer.dense_decoder_layer.DenseDecoderLayer.forward": {
            "fullgraph": True,
            "mode": torch_compile_mode,
        }
    }
else:
    compile_cfg = model_compile

model_cfg = Qwen3_5_VLTextDense9BConfig(compile_cfg=compile_cfg)
model_cfg.mudd_enabled = _bool_env("MUDD_ENABLED", False)
if _bool_env("MTP_ENABLED", True):
    model_cfg.mtp_config = MTPConfig(num_layers=1, loss_scaling_factor=0.1)

if optimizer == "muon":
    optim_cfg = MuonConfig(
        lr=float(os.environ.get("LR", "2e-5")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.05")),
        enable_all2all=_bool_env("MUON_ENABLE_ALL2ALL", False),
        adamw_foreach=_bool_env("MUON_ADAMW_FOREACH", False),
    )
elif optimizer == "adamw":
    optim_cfg = AdamWConfig(
        lr=float(os.environ.get("LR", "2e-5")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.05")),
        foreach=_bool_env("ADAMW_FOREACH", False),
        swap_optimizer=_bool_env("SWAP_OPTIMIZER", False),
    )
else:
    raise ValueError(f"Unsupported OPTIMIZER={optimizer!r}; expected 'muon' or 'adamw'")

if data_format == "jsonl":
    dataset_class_name = "JsonlDataset"
    indexed_sequence_length = None
elif data_format == "megatron_indexed":
    dataset_class_name = "MegatronIndexedDataset"
    indexed_sequence_length = seq_len
else:
    raise ValueError(f"Unsupported DATA_FORMAT={data_format!r}; expected 'jsonl' or 'megatron_indexed'")

dataset_config = [
    {
        "dataset": DatasetConfig(
            name=f"qwen3p5-9b-{data_format}",
            anno_path=data_path,
            class_name=dataset_class_name,
            sequence_length=indexed_sequence_length,
            cache_dir=work_dir / "data_cache",
            enable_sequential_sampler=True,
        ),
        "tokenize_fn": PretrainTokenizeFunctionConfig(
            add_eos_token=False,
            add_bos_token=False,
        ),
    }
]

dataloader_cfg = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_level="none",
    pack_max_length=seq_len,
    pack_to_max_length=True,
    group_by_length=False,
    num_workers=0,
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=None,
    tokenizer_path=None,
    dataloader_cfg=dataloader_cfg,
    optim_cfg=optim_cfg,
    lr_cfg=LRConfig(lr_type="constant", warmup_ratio=0.0, lr_min=0.0),
    loss_cfg=CELossConfig(
        mode=loss_mode,
        chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "256")),
        loss_reduction=loss_reduction,
    ),
    fsdp_cfg=FSDPConfig(
        tp_size=1,
        ep_size=1,
        hsdp_sharding_size=hsdp_sharding_size,
        cpu_offload=False,
        reshard_after_forward=_bool_env("RESHARD_AFTER_FORWARD", True),
        gradient_accumulation_no_sync=_bool_env("FSDP_GRAD_ACCUMULATION_NO_SYNC", False),
        recompute_ratio=float(os.environ.get("RECOMPUTE_RATIO", "1.0")),
        checkpoint_preserve_rng_state=False,
        torch_compile=model_compile,
    ),
    sp_size=1,
    global_batch_size=global_batch_size,
    total_step=int(os.environ.get("TOTAL_STEP", "20")),
    checkpoint_interval=None,
    hf_interval=None,
    work_dir=work_dir,
    profile_memory=_bool_env("PROFILE_MEMORY", False),
    profile_step=_profile_step(),
    debug_skip_save=True,
    seed=42,
)
