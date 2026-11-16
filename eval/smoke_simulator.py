"""One real simulator dialogue, not a benchmark score."""
import argparse
import json

from envs.clients import Endpoint
from envs.tau import Simulator
from eval.storage import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/simulator-smoke/result.json")
    args = p.parse_args()
    endpoint = Endpoint("USER_SIM", 1, retries=0)
    user = Simulator(endpoint, 42, max_tokens=128)
    record = {"kind": "simulator connectivity smoke; not benchmark", "config": endpoint.public_config()}
    try:
        first = user.reset("You want to know the opening time. Ask the agent. Once the agent tells you the opening time, your goal is satisfied.")
        second = user.step("We open at 9 AM.")
        record.update(status="ok", initial_message=first, final_message=second, stop_observed="###STOP###" in second)
    except Exception as e:
        record.update(status="error", error_type=type(e).__name__)
    finally:
        record["api_events"] = user.events
        record["messages"] = user.messages
        atomic_json(args.output, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("api_events", "messages")}, ensure_ascii=False))
    if record["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
