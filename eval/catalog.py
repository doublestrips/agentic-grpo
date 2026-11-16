import argparse
import subprocess
from pathlib import Path

from envs.tau import make_env
from eval.storage import atomic_json, digest


def describe(domain, split):
    env = make_env(domain, split, 0)
    tasks = [task.model_dump() for task in env.tasks]
    return {"domain": domain, "split": split, "count": len(tasks),
            "task_ids": list(range(len(tasks))), "tasks_sha256": digest(tasks),
            "policy_sha256": digest(env.wiki), "tools_sha256": digest(env.tools_info),
            "task_sha256": [digest(t) for t in tasks]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/task-catalog.json")
    args = parser.parse_args()
    entries = []
    for domain in ("retail", "airline"):
        for split in ("train", "dev", "test"):
            try:
                entries.append(describe(domain, split))
            except (ValueError, AssertionError, ModuleNotFoundError) as e:
                entries.append({"domain": domain, "split": split, "available": False, "error": str(e)})
    atomic_json(args.output, entries)
    for e in entries:
        print(e["domain"], e["split"], e.get("count", "unavailable"))


if __name__ == "__main__":
    main()
