#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
exec > >(tee -a logs/install-vllm.log) 2>&1
trap 'code=$?; printf "%s\n" "$code" > logs/install-vllm.exit' EXIT
if [ ! -x "$AGENTIC_ROOT/envs/vllm/bin/python" ]; then
  uv venv --python 3.11.14 "$AGENTIC_ROOT/envs/vllm"
fi
if [ ! -f configs/requirements-vllm.lock ]; then
  uv pip compile configs/requirements-vllm.in --python-version 3.11 --generate-hashes -o configs/requirements-vllm.lock
fi
uv pip sync --python "$AGENTIC_ROOT/envs/vllm/bin/python" configs/requirements-vllm.lock
uv pip freeze --python "$AGENTIC_ROOT/envs/vllm/bin/python" > artifacts/vllm-freeze.txt
"$AGENTIC_ROOT/envs/vllm/bin/python" -c 'import torch,vllm; print("torch",torch.__version__,"runtime",torch.version.cuda,"vllm",vllm.__version__)'
