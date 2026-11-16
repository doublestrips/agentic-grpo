#!/usr/bin/env python3
"""CLI: python -m eval.run_eval ... (from the project directory)."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tau_bench.types import Action

from envs.clients import Endpoint, EndpointError
from envs.tau import Simulator, make_env
from eval.catalog import describe
from eval.metrics import aggregate
from eval.storage import atomic_json, digest


def clean_message(message):
    result = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        result["tool_calls"] = [call.model_dump() for call in message.tool_calls]
    return result


def rollout(config, task_id, trial, policy, user):
    started = time.monotonic()
    seed = config["seed"] + task_id * config["trials"] + trial
    sim = Simulator(user, seed, temperature=config["user_temperature"])
    events, messages, observations = [], [], []
    record = {"task_id": task_id, "trial": trial, "seed": seed, "status": "system_error",
              "reward": None, "termination": "uninitialized", "rounds": 0, "tool_calls": 0,
              "config_sha256": digest(config), "messages": messages, "observations": observations}
    try:
        env = make_env(config["domain"], config["split"], task_id)
        env.user = sim
        obs = env.reset(task_id).observation
        messages.extend([{"role": "system", "content": env.wiki}, {"role": "user", "content": obs}])
        record["reward"] = 0.0
        for step in range(config["max_steps"]):
            response = policy.complete(messages, events, tools=env.tools_info, tool_choice="auto",
                parallel_tool_calls=False, temperature=config["temperature"], top_p=config["top_p"],
                max_tokens=config["max_tokens"], seed=seed + step)
            record["rounds"] += 1
            choice = response.choices[0]
            message = clean_message(choice.message)
            if choice.finish_reason == "length":
                messages.append(message)
                record.update(status="truncated", termination="policy_generation_length")
                break
            if choice.finish_reason not in ("stop", "tool_calls"):
                record.update(status="system_error", termination="policy_invalid_finish_reason")
                messages.append(message)
                break
            calls = message.get("tool_calls", [])
            if calls:
                # Preserve official TC behavior: executes the first tool only.
                # Raw response retains any additional calls for inspection.
                message["tool_calls"] = calls[:1]
                call = calls[0]
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be an object")
                except (ValueError, TypeError):
                    messages.append(message)
                    record.update(status="completed", termination="invalid_tool_arguments")
                    break
                action = Action(name=call["function"]["name"], kwargs=arguments)
                record["tool_calls"] += 1
            else:
                if not isinstance(message["content"], str) or not message["content"]:
                    record.update(status="system_error", termination="policy_empty_response")
                    break
                action = Action(name="respond", kwargs={"content": message["content"]})
            messages.append(message)
            # Keep the official state transition and official terminal reward.
            response_env = env.step(action)
            observations.append({"step": step, "action": action.model_dump(),
                                 "observation": response_env.observation, "done": response_env.done})
            if calls:
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "name": action.name, "content": response_env.observation})
            else:
                messages.append({"role": "user", "content": response_env.observation})
            if response_env.done:
                info = response_env.info.model_dump()
                # User cost interface placeholder is NOT an actual monetary cost.
                info.pop("user_cost", None)
                record.update(status="completed", termination="tool_terminated" if calls else "user_stop",
                              reward=response_env.reward, official_info=info)
                break
        else:
            record.update(status="max_steps", termination="step_limit")
    except EndpointError as error:
        record.update(status="truncated" if error.kind == "context_overflow" else "system_error",
                      termination=f"{error.source}:{error.kind}", error=str(error))
    except Exception as error:
        # SDK unexpected exceptions are kept to type; do not leak raw remote data.
        kind = str(error) if str(error) in ("simulator_truncated", "simulator_invalid_response") else type(error).__name__
        record.update(status="system_error", termination=kind,
                      error_frames=[{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                                    for frame in traceback.extract_tb(error.__traceback__)])
    finally:
        record["api_events"] = events + sim.events
        record["simulator_messages"] = sim.messages
        record["elapsed_seconds"] = time.monotonic() - started
    return record


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("retail", "airline"), required=True)
    p.add_argument("--split", choices=("train", "dev", "test"), default="test")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--task-ids", type=int, nargs="+")
    selection.add_argument("--task-range", type=int, nargs=2, metavar=("START", "END"))
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--user-concurrency", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--user-temperature", type=float, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.trials, args.concurrency, args.user_concurrency, args.max_steps, args.max_tokens) <= 0:
        p.error("counts and limits must be positive")
    if args.max_tokens >= 8192:
        p.error("max-tokens must leave room for prompt within 8192")
    catalog = describe(args.domain, args.split)
    ids = args.task_ids if args.task_ids is not None else (list(range(*args.task_range)) if args.task_range else catalog["task_ids"])
    if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(catalog["task_ids"]):
        p.error("task selection must contain unique existing task IDs")
    policy = Endpoint("POLICY", args.concurrency)
    user = Endpoint("USER_SIM", args.user_concurrency)
    root = Path(os.environ["AGENTIC_ROOT"])
    config = {k: v for k, v in vars(args).items() if k not in ("output", "resume", "task_ids", "task_range")}
    config.update(task_ids=ids, context_limit=8192, policy=policy.public_config(), user=user.public_config(),
                  catalog=catalog, upstream_revision=revision(root / "sources/tau-bench"),
                  code_revision=revision(Path(__file__).resolve().parents[1]))
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], text=True).strip():
        p.error("commit source/config changes before evaluating")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("w") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = args.output / "config.json"
        if config_path.exists():
            if not args.resume or json.loads(config_path.read_text()) != config:
                p.error("existing output requires --resume and exactly matching config")
        else:
            if any(args.output.glob("trials/*.json")):
                p.error("orphan trial files without config")
            atomic_json(config_path, config)
        trial_dir = args.output / "trials"
        trial_dir.mkdir(exist_ok=True)
        records = [json.loads(path.read_text()) for path in trial_dir.glob("*.json")]
        for r in records:
            if r["config_sha256"] != digest(config):
                p.error("trial config mismatch")
        aggregate(records, ids, args.trials)  # validates duplicates and selection
        done = {(r["task_id"], r["trial"]) for r in records}
        pending = [(task, trial) for task in ids for trial in range(args.trials) if (task, trial) not in done]
        # Terminal system errors remain recorded; rerun them in a separate run.
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(rollout, config, t, j, policy, user): (t, j) for t, j in pending}
            for future in as_completed(futures):
                result = future.result()
                atomic_json(trial_dir / f"task-{result['task_id']:04d}-trial-{result['trial']:02d}.json", result)
                records.append(result)
                atomic_json(args.output / "metrics.json", aggregate(records, ids, args.trials))
                print(json.dumps({k: result[k] for k in ("task_id", "trial", "status", "reward", "termination")}), flush=True)
        metrics = aggregate(records, ids, args.trials)
        atomic_json(args.output / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
        if metrics["missing_trials"] or metrics["status_counts"].get("system_error", 0):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
