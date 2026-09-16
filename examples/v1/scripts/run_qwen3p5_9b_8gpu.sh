#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export WORK_DIR="${WORK_DIR:-$REPO_ROOT/work_dirs/qwen3p5_9b_8gpu_gbs16_20steps_step5_trace_ncp_mtp_on}"

mkdir -p "$WORK_DIR" "$WORK_DIR/data"
export DATA_FORMAT="${DATA_FORMAT:-megatron_indexed}"
export SEQ_LEN="${SEQ_LEN:-8192}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
export TOTAL_STEP="${TOTAL_STEP:-20}"
# Trainer checks this value against its zero-based pre-step counter, so 4
# captures the user-visible Step 5/20.
export PROFILE_STEP="${PROFILE_STEP:-4}"
export RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0}"
export RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-0}"
export FSDP_GRAD_ACCUMULATION_NO_SYNC="${FSDP_GRAD_ACCUMULATION_NO_SYNC:-1}"
export MUDD_ENABLED="${MUDD_ENABLED:-0}"
export MUDD_ROUTE_KERNEL="${MUDD_ROUTE_KERNEL:-1}"
export MUDD_COMPILE="${MUDD_COMPILE:-0}"
export MUDD_ROUTER_NORM_AFFINE="${MUDD_ROUTER_NORM_AFFINE:-0}"
export MTP_ENABLED="${MTP_ENABLED:-1}"
export OPTIMIZER="${OPTIMIZER:-muon}"
export MODEL_COMPILE="${MODEL_COMPILE:-1}"
export LOSS_MODE="${LOSS_MODE:-chunk}"
export LOSS_REDUCTION="${LOSS_REDUCTION:-square}"
export LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-4096}"
export MUON_ENABLE_ALL2ALL="${MUON_ENABLE_ALL2ALL:-1}"
export MUON_ADAMW_FOREACH="${MUON_ADAMW_FOREACH:-1}"
export XTUNER_MUON_USE_TRITON="${XTUNER_MUON_USE_TRITON:-1}"
export XTUNER_USE_FA3=1

if [[ "$DATA_FORMAT" == "jsonl" ]]; then
    export DATA_PATH="${DATA_PATH:-$WORK_DIR/data/synthetic.jsonl}"
    synthetic_samples="${SYNTHETIC_SAMPLES:-$((TOTAL_STEP * GLOBAL_BATCH_SIZE + GLOBAL_BATCH_SIZE))}"
    python - "$DATA_PATH" "${SEQ_LEN:-8192}" "$synthetic_samples" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
seq_len = int(sys.argv[2])
sample_count = int(sys.argv[3])
with path.open("w", encoding="ascii") as file:
    for _ in range(max(16, sample_count)):
        json.dump({"content": "x" * seq_len}, file)
        file.write("\n")
PY
else
    export DATA_PATH="${DATA_PATH:-$WORK_DIR/data/indexed}"
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    -m xtuner.v1.train.cli.sft \
    --config "$REPO_ROOT/examples/v1/config/sft_qwen3p5_9b.py"
