"""Official tool-calling solver with separate endpoint routing only.

Does not claim unmodified upstream run.py: no upstream files are changed, but
the two completion clients are adapted for local policy + external simulator.
"""
import argparse
from types import SimpleNamespace
from unittest.mock import patch

from tau_bench.agents.tool_calling_agent import ToolCallingAgent

from envs.clients import Endpoint
from envs.tau import Simulator, make_env
from eval.storage import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=("retail", "airline"), default="retail")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--output", default="results/official-smoke/result.json")
    args = p.parse_args()
    policy, user = Endpoint("POLICY", 1, retries=0), Endpoint("USER_SIM", 1, retries=0)
    env = make_env(args.domain, "test", args.task_id)
    sim = Simulator(user, 42)
    env.user = sim
    events = []

    def completion(**kwargs):
        result = policy.complete(kwargs["messages"], events, tools=kwargs["tools"],
                                  temperature=kwargs["temperature"], max_tokens=1024,
                                  parallel_tool_calls=False, seed=42)
        return SimpleNamespace(choices=result.choices, _hidden_params={"response_cost": None})

    record = {"kind": "official ToolCallingAgent with endpoint adapters", "task_id": args.task_id, "domain": args.domain}
    try:
        with patch("tau_bench.agents.tool_calling_agent.completion", side_effect=completion):
            result = ToolCallingAgent(env.tools_info, env.wiki, policy.model, "openai").solve(env, args.task_id)
        record.update(status="ok", official_result=result.model_dump())
        # Interface placeholders for monetary cost are not measured results.
        record["official_result"].pop("total_cost", None)
        record["official_result"]["info"].pop("user_cost", None)
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__)
    finally:
        record.update(policy_events=events, simulator_events=sim.events, simulator_messages=sim.messages)
        atomic_json(args.output, record)
    print(record["status"])
    if record["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
