#!/usr/bin/env bash
# 8-GPU torchrun smoke for Kimi Linear tiny (AttnRes cross-layer forward).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CKPT="${CKPT:-${REPO_ROOT}/work_dirs/kimi_linear_tiny_bf16}"
WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work_dirs/kimi_linear_tiny_8gpu_smoke}"
SEQ_LEN="${SEQ_LEN:-64}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"

mkdir -p "${WORK_DIR}" "${CKPT}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export XTUNER_ACTIVATION_OFFLOAD="${XTUNER_ACTIVATION_OFFLOAD:-0}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --tee 3 \
  "${REPO_ROOT}/examples/v1/scripts/smoke_kimi_linear_tiny.py" \
  --ckpt "${CKPT}" \
  --seq-len "${SEQ_LEN}" \
  2>&1 | tee -a "${WORK_DIR}/smoke.log"
