#!/usr/bin/env bash
# Run on the personal computer; no API credential is sent by SSH.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
socket="$PWD/logs/196-api-tunnel.sock"
ssh -M -S "$socket" -fN -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:18761:127.0.0.1:18761 196
printf 'Tunnel control socket: %s\n' "$socket"
