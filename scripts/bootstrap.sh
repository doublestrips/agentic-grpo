#!/usr/bin/env bash
set -euo pipefail
umask 077
cd "$(dirname "$0")/.."
source scripts/activate.sh
mkdir -p "$AGENTIC_ROOT"/{sources,models,envs,cache,tmp,tools/bin,tools/downloads} "$TMPDIR" logs artifacts
exec > >(tee -a logs/bootstrap.log) 2>&1
rm -f logs/bootstrap.exit
trap 'code=$?; printf "%s\n" "$code" > logs/bootstrap.exit' EXIT
date -Is
upstream="$AGENTIC_ROOT/sources/tau-bench"
upstream_commit=59a200c6d575d595120f1cb70fea53cef0632f6b
if [ ! -d "$upstream/.git" ]; then
  git clone https://github.com/sierra-research/tau-bench.git "$upstream"
  git -C "$upstream" checkout --detach "$upstream_commit"
fi
test "$(git -C "$upstream" rev-parse HEAD)" = "$upstream_commit"
archive="$AGENTIC_ROOT/tools/downloads/uv-0.9.9-linux-x86_64.tar.gz"
if [ ! -x "$AGENTIC_ROOT/tools/bin/uv" ]; then
  curl -fsSL --retry 3 --connect-timeout 20 --max-time 300 \
    https://github.com/astral-sh/uv/releases/download/0.9.9/uv-x86_64-unknown-linux-gnu.tar.gz -o "$archive"
  curl -fsSL --retry 3 --connect-timeout 20 --max-time 60 \
    https://github.com/astral-sh/uv/releases/download/0.9.9/uv-x86_64-unknown-linux-gnu.tar.gz.sha256 -o "$archive.sha256"
  expected="$(awk '{print $1}' "$archive.sha256")"
  actual="$(sha256sum "$archive" | awk '{print $1}')"
  test "$expected" = "$actual"
  tar -xzf "$archive" -C "$AGENTIC_ROOT/tools/downloads"
  cp "$AGENTIC_ROOT/tools/downloads/uv-x86_64-unknown-linux-gnu/uv" "$AGENTIC_ROOT/tools/bin/uv"
fi
uv --version
uv python install 3.11.14 --no-bin
if [ ! -x "$AGENTIC_ROOT/envs/eval/bin/python" ]; then
  uv venv --python 3.11.14 "$AGENTIC_ROOT/envs/eval"
fi
if [ ! -f configs/requirements-eval.lock ]; then
  uv pip compile configs/requirements-eval.in --python-version 3.11 --generate-hashes -o configs/requirements-eval.lock
fi
uv pip sync --python "$AGENTIC_ROOT/envs/eval/bin/python" configs/requirements-eval.lock
"$AGENTIC_ROOT/envs/eval/bin/python" -c 'import openai,litellm,numpy,pydantic; from tau_bench.envs import get_env; print("evaluation imports OK")'
uv pip freeze --python "$AGENTIC_ROOT/envs/eval/bin/python" > artifacts/eval-freeze.txt
"$AGENTIC_ROOT/envs/eval/bin/python" --version
date -Is
