#!/usr/bin/env bash
# Source this file; do not modify shared shell startup files.
export AGENTIC_ROOT="${AGENTIC_ROOT:-/traindata2/ganmuv3}"
export AGENTIC_PROJECT="$AGENTIC_ROOT/projects/agentic-grpo"
export HF_HOME="$AGENTIC_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$AGENTIC_ROOT/cache/xdg"
export PIP_CACHE_DIR="$AGENTIC_ROOT/cache/pip"
export UV_CACHE_DIR="$AGENTIC_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$AGENTIC_ROOT/tools/python"
export UV_PYTHON_BIN_DIR="$AGENTIC_ROOT/tools/bin"
export TORCH_HOME="$AGENTIC_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$AGENTIC_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$AGENTIC_ROOT/cache/triton"
export CUDA_CACHE_PATH="$AGENTIC_ROOT/cache/cuda"
export TMPDIR="$AGENTIC_ROOT/tmp"
export RAY_TMPDIR="$AGENTIC_ROOT/tmp/ray"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$AGENTIC_PROJECT:$AGENTIC_ROOT/sources/tau-bench"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=offline
export PATH="$AGENTIC_ROOT/tools/bin:$PATH"
if [ -f "$AGENTIC_ROOT/envs/eval/bin/activate" ]; then
  source "$AGENTIC_ROOT/envs/eval/bin/activate"
fi
