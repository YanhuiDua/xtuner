#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${WORK_DIR:?WORK_DIR must point to workspace-backed storage}"

mkdir -p "$WORK_DIR" "$WORK_DIR/data"
export DATA_FORMAT="${DATA_FORMAT:-jsonl}"
export SEQ_LEN="${SEQ_LEN:-8192}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export TOTAL_STEP="${TOTAL_STEP:-3}"
export RECOMPUTE_RATIO="${RECOMPUTE_RATIO:-0}"
export RESHARD_AFTER_FORWARD="${RESHARD_AFTER_FORWARD:-0}"
export MUDD_ENABLED="${MUDD_ENABLED:-0}"
export MTP_ENABLED="${MTP_ENABLED:-1}"
export OPTIMIZER="${OPTIMIZER:-muon}"
export MODEL_COMPILE="${MODEL_COMPILE:-0}"
export LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-256}"
export MUON_ENABLE_ALL2ALL="${MUON_ENABLE_ALL2ALL:-0}"
export MUON_ADAMW_FOREACH="${MUON_ADAMW_FOREACH:-0}"

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
    : "${DATA_PATH:?DATA_PATH is required when DATA_FORMAT=$DATA_FORMAT}"
    export DATA_PATH
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    -m xtuner.v1.train.cli.sft \
    --config "$REPO_ROOT/examples/v1/config/sft_qwen3p5_9b.py"
