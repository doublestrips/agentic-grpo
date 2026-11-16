#!/usr/bin/env python3
"""Rules with explicit evidence; semantic policy errors stay unknown absent review."""
import argparse
import json
from collections import Counter
from pathlib import Path

from eval.storage import atomic_json


def classify(record):
    labels = []

    def add(category, rule, evidence, confidence="high"):
        labels.append({"category": category, "rule": rule, "evidence": evidence, "confidence": confidence})

    term = record["termination"]
    if "USER_SIM" in term or term.startswith("simulator_"):
        add("用户模拟器问题", "simulator-terminal-error", term)
    elif record["status"] == "system_error":
        add("运行环境或策略服务异常", "system-terminal-error", term)
    if record["status"] == "truncated" or "truncated" in term:
        add("超长截断", "explicit-truncation", term)
    if record["status"] == "max_steps":
        add("步数上限", "explicit-step-limit", term)
    if term == "invalid_tool_arguments":
        add("参数错", "arguments-not-json-object", term)
    for i, step in enumerate(record.get("observations", [])):
        obs = str(step["observation"])
        if obs.startswith("Unknown action "):
            add("工具选错", "unknown-tool-name", {"observation_index": i, "text": obs})
        elif obs.startswith("Error:") and any(s in obs for s in ("unexpected keyword argument", "missing 1 required", "required positional argument")):
            add("参数错", "python-call-signature-error", {"observation_index": i, "text": obs})
    # Optional annotations must be explicitly human verified, never inferred from reward.
    for annotation in record.get("review_annotations", []):
        if annotation.get("human_verified") is True and annotation.get("category") in ("违反 policy", "提前结束", "工具选错", "参数错"):
            add(annotation["category"], "human-reviewed", annotation.get("evidence"))
    if not labels:
        add("未确定", "insufficient-evidence", term, "unknown")
    return {"task_id": record["task_id"], "trial": record["trial"],
            "primary": labels[0]["category"], "labels": labels,
            "human_reviewed": any(a.get("human_verified") is True for a in record.get("review_annotations", []))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    records = [json.loads(path.read_text()) for path in sorted((args.run / "trials").glob("*.json"))]
    failures = [r for r in records if r["status"] != "completed" or r["reward"] != 1]
    classified = [classify(r) for r in failures]
    counts = Counter(r["primary"] for r in classified)
    data = {"failure_count": len(failures), "recorded_trial_count": len(records),
            "system_errors": sum(r["status"] == "system_error" for r in failures),
            "primary_counts": dict(counts),
            "primary_proportions": {k: v / len(failures) for k, v in counts.items()},
            "proportion_denominator": "all recorded non-success trials, including system errors",
            "human_review_status": "pending unless explicitly annotated", "trajectories": classified}
    atomic_json(args.run / "failure_taxonomy.json", data)
    print(json.dumps({k: v for k, v in data.items() if k != "trajectories"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
