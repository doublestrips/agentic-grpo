#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
exec > >(tee -a logs/model-prefetch.log) 2>&1
rm -f logs/model-prefetch.exit
trap 'code=$?; printf "%s\n" "$code" > logs/model-prefetch.exit' EXIT
python -m scripts.prefetch_weights --indices 3 4 --workers 2
