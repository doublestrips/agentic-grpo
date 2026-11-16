#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ssh -S "$PWD/logs/196-api-tunnel.sock" -O exit 196
