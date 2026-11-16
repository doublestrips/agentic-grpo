#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
: "${APPROVED_GPU_UUIDS:?Set GPU UUID(s) explicitly after owner allocation}"
: "${POLICY_MODEL_PATH:?Set downloaded immutable model snapshot path}"
: "${POLICY_MODEL:?Set served model name}"
export CUDA_VISIBLE_DEVICES="$APPROVED_GPU_UUIDS"
export VLLM_CACHE_ROOT="$AGENTIC_ROOT/cache/vllm"
port="${POLICY_PORT:-18762}"
tp="${TENSOR_PARALLEL_SIZE:-1}"
# Deliberately fail instead of touching any existing process on this port/GPU.
exec "$AGENTIC_ROOT/envs/vllm/bin/vllm" serve "$POLICY_MODEL_PATH" \
  --served-model-name "$POLICY_MODEL" --host 127.0.0.1 --port "$port" \
  --max-model-len 8192 --dtype bfloat16 --tensor-parallel-size "$tp" \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.85 --max-num-seqs 4 --generation-config vllm
