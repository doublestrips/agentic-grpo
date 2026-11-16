"""Inventory project-owned installations and source versions without secrets."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.storage import atomic_json


def command(args):
    output = subprocess.run(args, capture_output=True, text=True)
    return {"exit": output.returncode, "stdout": output.stdout, "stderr": output.stderr}


def main():
    root = Path(os.environ["AGENTIC_ROOT"])
    project = root / "projects/agentic-grpo"
    paths = ["projects/agentic-grpo", "sources/tau-bench", "models", "envs/eval", "envs/vllm", "cache", "tmp", "tools", "config"]
    record = {"at": datetime.now(timezone.utc).isoformat(), "namespace": str(root), "git_author": "jack <wanxueyu1999@163.com>",
              "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "managed_paths": [str(root / p) for p in paths],
              "git_local": command(["git", "-C", str(project), "config", "--local", "--list"]),
              "upstream_commit": command(["git", "-C", str(root / "sources/tau-bench"), "rev-parse", "HEAD"]),
              "gpu_snapshot": command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv"]),
              "tmux": command(["tmux", "-L", "ganmuv3-agentic", "list-sessions"]),
              "locks": {}}
    for file in (project / "configs").glob("*.lock"):
        record["locks"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    atomic_json(project / "artifacts/provenance.json", record)
    print(json.dumps({"managed_paths": record["managed_paths"], "locks": record["locks"]}, indent=2))


if __name__ == "__main__":
    main()
