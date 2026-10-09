#!/usr/bin/env bash
set -euo pipefail

# 8-GPU Concept-OLMo Stage-1 run aligned to Megatron GBS=512 / seq=8192.
# Layer-DD only (this tree has no cumsum). Run inside the already-started
# rjob replica. Do not call submit_rjob.sh.
#
# The Megatron conda env (megatron-h200-cu128-py312) has torch/CUDA but not
# XTuner runtime deps (loguru, mmengine, cyclopts, ...). This script overlays
# those into a writable venv with --system-site-packages. Do not pip-install
# into jxxiong's shared conda prefix.
#
# Required: dump Megatron's first 100*512 samples first:
#   bash .../dump_megatron_align_samples.sh
#   MEGATRON_REPLAY_PATH=<dump prefix> bash examples/v1/scripts/run_concept_olmo_stage1_8gpu.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH="${1:-${REPO_ROOT}/ci/config/concept_olmo_stage1_megatron_align.py}"
export WORK_DIR="${2:-${WORK_DIR:-${REPO_ROOT}/work_dirs/concept_olmo_stage1_megatron_align_dd}}"

DEFAULT_REPLAY="${DEFAULT_REPLAY:-/mnt/shared-storage-user/llmrazor-share/heqiushi/concept_olmo/full_train_dd_diag_256g_0611/data/megatron_replay_100x512/samples}"
export MEGATRON_REPLAY_PATH="${MEGATRON_REPLAY_PATH:-$DEFAULT_REPLAY}"
DEFAULT_LOAD_FROM="${DEFAULT_LOAD_FROM:-/mnt/shared-storage-user/llmrazor-share/heqiushi/concept_olmo/full_train_dd_diag_256g_0611/data/megatron_init_hf_dd}"
if [[ -z "${LOAD_FROM:-}" && -f "${DEFAULT_LOAD_FROM}/config.json" ]]; then
  export LOAD_FROM="$DEFAULT_LOAD_FROM"
fi
if [[ -n "${LOAD_FROM:-}" ]]; then
  export TOKENIZER_PATH="${TOKENIZER_PATH:-$LOAD_FROM}"
fi
if [[ ! -f "${MEGATRON_REPLAY_PATH}.idx" || ! -f "${MEGATRON_REPLAY_PATH}.bin" ]]; then
  echo "missing Megatron replay dump: ${MEGATRON_REPLAY_PATH}.bin/.idx" >&2
  echo "dump first with dump_megatron_align_samples.sh, using the same DATA_ARGS_PATH and DATA_CACHE_PATH as the Megatron 100-step run" >&2
  exit 2
fi

export TOTAL_STEP="${TOTAL_STEP:-100}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export INTRA_LAYER_MICRO_BATCH="${INTRA_LAYER_MICRO_BATCH:-1}"
export SP_SIZE="${SP_SIZE:-1}"
export LR="${LR:-6.0e-5}"
export MIN_LR="${MIN_LR:-6.0e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}"
export ADAM_BETA2="${ADAM_BETA2:-0.95}"
export ADAM_EPS="${ADAM_EPS:-1.0e-8}"
export SEED="${SEED:-1234}"
export SKIP_SAVE="${SKIP_SAVE:-1}"
export TORCH_COMPILE="${TORCH_COMPILE:-1}"
export RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-1.0}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Overlay XTuner-only packages without mutating jxxiong's conda prefix.
# pip --user + PYTHONUSERBASE is conda-safe (no ensurepip / venv). conda's
# torchrun shebang still uses conda python, which sees this via PYTHONPATH.
XTUNER_PYTHON="${XTUNER_PYTHON:-python}"
if [[ "${SKIP_XTUNER_OVERLAY:-0}" != "1" ]]; then
  export PYTHONUSERBASE="${XTUNER_USERBASE:-${REPO_ROOT}/.xtuner_userbase}"
  PYVER="$("${XTUNER_PYTHON}" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
  XTUNER_SITE="${PYTHONUSERBASE}/lib/${PYVER}/site-packages"
  mkdir -p "${XTUNER_SITE}"
  export PYTHONPATH="${XTUNER_SITE}:${PYTHONPATH}"
  unset PYTHONNOUSERSITE
  "${XTUNER_PYTHON}" - <<'PY'
import importlib
import os
import subprocess
import sys

needed = {
    "loguru": "loguru",
    "mmengine": "mmengine>=0.10",
    "cyclopts": "cyclopts",
    "addict": "addict",
    "yaml": "pyyaml",
    "einops": "einops",
    "pydantic": "pydantic",
    "tensorboard": "tensorboard",
    "xxhash": "xxhash",
    "datasets": "datasets<4.0.0",
    "tiktoken": "tiktoken",
    "sentencepiece": "SentencePiece",
    "peft": "peft",
    "codetiming": "codetiming",
    "git": "GitPython",
    "pyarrow": "pyarrow",
    "more_itertools": "more_itertools",
}
missing = []
for module_name, spec in needed.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing.append(spec)
if not missing:
    raise SystemExit(0)
print("installing XTuner runtime deps into", os.environ["PYTHONUSERBASE"], ":", " ".join(missing))
cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--user", *missing]
try:
    subprocess.check_call(cmd)
except subprocess.CalledProcessError:
    site = os.path.join(
        os.environ["PYTHONUSERBASE"],
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )
    print("pip --user failed; falling back to --target", site)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--target", site, *missing]
    )
PY
fi

export XTUNER_ACTIVATION_OFFLOAD="${XTUNER_ACTIVATION_OFFLOAD:-0}"
export XTUNER_GC_ENABLE="${XTUNER_GC_ENABLE:-1}"
export XTUNER_USE_FA3="${XTUNER_USE_FA3:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

NNODES="${NNODES:-${NODE_COUNT:-1}}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"

mkdir -p "${WORK_DIR}"
ulimit -n 65536

echo "XTuner_fix Concept-OLMo Stage-1 Megatron-align (layer-DD)"
echo "  python=${XTUNER_PYTHON}"
echo "  pythonpath_overlay=${XTUNER_SITE:-}"
echo "  config=${CONFIG_PATH}"
echo "  work_dir=${WORK_DIR}"
echo "  megatron_replay_path=${MEGATRON_REPLAY_PATH}"
echo "  load_from=${LOAD_FROM:-}"
echo "  gbs=${GLOBAL_BATCH_SIZE} seq=${SEQ_LENGTH} mbs=${INTRA_LAYER_MICRO_BATCH} sp=${SP_SIZE}"
echo "  tokens_per_step=$((GLOBAL_BATCH_SIZE * SEQ_LENGTH)) total_step=${TOTAL_STEP} seed=${SEED}"
echo "  lr=${LR} min_lr=${MIN_LR} optimizer=muon momentum=${MUON_MOMENTUM:-0.9} nesterov=0 adjust_lr=rms_norm extra_scale_factor=1 qkv_split=1 reduce_dtype=fp32"
echo "  dd_mode=dd compile=off (DepthDD / ResidualRoute / ConceptRoute)"
echo "  compare: XTuner reduced_llm_loss vs Megatron lm loss (OLMO3_Z_LOSS_MULTIPLIER=0)"

"${XTUNER_PYTHON}" -m torch.distributed.run \
  --nproc-per-node="${NPROC_PER_NODE:-8}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --tee 3 \
  -m xtuner.v1.train.cli.sft \
  --config "${CONFIG_PATH}" \
  2>&1 | tee -a "${WORK_DIR}/node_${NODE_RANK}.txt"
