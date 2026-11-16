#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
exec > >(tee -a logs/model-relay.log) 2>&1
rm -f logs/model-relay.exit
trap 'code=$?; printf "%s\n" "$code" > logs/model-relay.exit' EXIT
date -Is
export MODEL_RELAY_URL="${MODEL_RELAY_URL:-http://127.0.0.1:18761/hf}"
python -m scripts.download_models
