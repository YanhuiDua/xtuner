"""Qwen3.5-2B ConceptLM pretraining / throughput-smoke configuration.

Mirrors the reference recipe (``train_conceptlm_qwen35_2b.sh``): three Qwen3.5-2B
hybrid stacks (encoder 12 / predictor 8 / decoder 12), chunk-4 mean pooling,
32x128 residual VQ, HLM objective, and MUDD switchable per stack (the reference
turns all three on with ``CONCEPTLM_MUDD_COMPILE=1``).  Optimizer and data
plumbing follow the Qwen3.5 dense examples so the numbers stay comparable with
``run_qwen3p5_9b_8gpu.sh`` / ``run_qwen3p5_2b_8gpu.sh``.

Environment switches:

* ``CONCEPTLM_MUDD_COMPILE=1`` - enable MUDD in all three stacks (reference
  "best" configuration);
* ``CONCEPTLM_{ENCODER,PREDICTOR,DECODER}_MUDD=1`` - enable MUDD per stack;
* ``MUDD_ROUTE_KERNEL`` / ``MUDD_COMPILE`` / ``MUDD_ROUTER_NORM_AFFINE`` -
  route implementation switches, same defaults as the dense examples.
"""

import os
from pathlib import Path

from xtuner.v1.config import FSDPConfig, LRConfig, MuonConfig
from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import ConceptLMQwen35Config
from xtuner.v1.train import TrainerConfig


def _bool_env(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


def _profile_step() -> list[int] | None:
    raw = os.environ.get("PROFILE_STEP", "4")
    return [int(step.strip()) for step in raw.split(",") if step.strip()] or None


repo_root = Path(__file__).resolve().parents[3]
work_dir = Path(os.environ.get("WORK_DIR", repo_root / "work_dirs/conceptlm_qwen3p5_2b"))
data_path = Path(os.environ.get("DATA_PATH", work_dir / "data/indexed"))
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

model_cfg = ConceptLMQwen35Config(compile_cfg=compile_cfg)
# ConceptLM knobs (defaults already follow the reference launcher).
model_cfg.concept_encoder_layers = int(os.environ.get("CONCEPTLM_ENCODER_LAYERS", "12"))
model_cfg.concept_predictor_layers = int(os.environ.get("CONCEPTLM_PREDICTOR_LAYERS", "8"))
model_cfg.concept_decoder_layers = int(os.environ.get("CONCEPTLM_DECODER_LAYERS", "12"))
model_cfg.concept_chunk_size = int(os.environ.get("CONCEPTLM_CHUNK_SIZE", "4"))
model_cfg.concept_codebook_size = int(os.environ.get("CONCEPTLM_CODEBOOK_SIZE", "128"))
model_cfg.concept_num_codebooks = int(os.environ.get("CONCEPTLM_NUM_CODEBOOKS", "32"))
model_cfg.concept_commitment_cost = float(os.environ.get("CONCEPTLM_COMMITMENT_COST", "0.25"))
model_cfg.concept_vq_loss_weight = float(os.environ.get("CONCEPTLM_VQ_LOSS_WEIGHT", "1.0"))
model_cfg.concept_hlm_loss_weight = float(os.environ.get("CONCEPTLM_HLM_LOSS_WEIGHT", "1.0"))
model_cfg.concept_shift_feature = _bool_env("CONCEPTLM_SHIFT_FEATURE", True)
model_cfg.concept_detach_hlm_target = _bool_env("CONCEPTLM_DETACH_HLM_TARGET", True)

# MUDD: per-stack switches, plus the reference's one-flag all-stack mode.
model_cfg.mudd_route_kernel = _bool_env("MUDD_ROUTE_KERNEL", True)
model_cfg.mudd_compile = _bool_env("MUDD_COMPILE", False)
model_cfg.mudd_router_norm_affine = _bool_env("MUDD_ROUTER_NORM_AFFINE", False)
model_cfg.mudd_route_scatter = _bool_env("MUDD_ROUTE_SCATTER", True)
model_cfg.concept_encoder_mudd = _bool_env("CONCEPTLM_ENCODER_MUDD", False)
model_cfg.concept_predictor_mudd = _bool_env("CONCEPTLM_PREDICTOR_MUDD", False)
model_cfg.concept_decoder_mudd = _bool_env("CONCEPTLM_DECODER_MUDD", False)
if _bool_env("CONCEPTLM_MUDD_COMPILE", False):
    model_cfg.concept_encoder_mudd = True
    model_cfg.concept_predictor_mudd = True
    model_cfg.concept_decoder_mudd = True

if optimizer == "muon":
    optim_cfg = MuonConfig(
        lr=float(os.environ.get("LR", "2e-5")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.05")),
        enable_all2all=_bool_env("MUON_ENABLE_ALL2ALL", False),
        adamw_foreach=_bool_env("MUON_ADAMW_FOREACH", False),
    )
else:
    raise ValueError(f"Unsupported OPTIMIZER={optimizer!r}; the ConceptLM recipe uses muon")

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
            name=f"conceptlm-qwen3p5-2b-{data_format}",
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
        chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "4096")),
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
