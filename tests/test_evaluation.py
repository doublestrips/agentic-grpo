"""Offline correctness tests. Synthetic scores here are never benchmark results."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from analysis.failure_taxonomy import classify
from envs.tau import Simulator, make_env
from eval.metrics import aggregate
from eval.run_eval import rollout
from eval.storage import atomic_json


def response(message, finish="stop"):
    return ChatCompletion.model_validate({"id": "offline-fixture", "object": "chat.completion", "created": 0,
        "model": "offline-fixture", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})


class ScriptedEndpoint:
    def __init__(self, outputs, source):
        self.outputs, self.source = iter(outputs), source

    def complete(self, messages, events, **kwargs):
        result = next(self.outputs)
        events.append({"source": self.source, "usage": result.usage.model_dump(), "response": result.model_dump()})
        return result


class MetricsTests(unittest.TestCase):
    def records(self):
        return [{"task_id": i, "trial": j, "reward": float(j < c), "status": "completed",
                 "termination": "user_stop", "rounds": 2, "tool_calls": 1}
                for i, c in enumerate((0, 1, 2, 4)) for j in range(4)]

    def test_pass_power_not_pass_at_k(self):
        metrics = aggregate(self.records(), [0, 1, 2, 3], 4)
        for k, expected in {"1": 7 / 16, "2": 7 / 24, "3": 1 / 4, "4": 1 / 4}.items():
            self.assertAlmostEqual(metrics["pass_k"][k], expected)

    def test_missing_or_duplicate_cannot_inflate_score(self):
        rows = self.records()
        self.assertEqual(aggregate(rows[:-1], [0, 1, 2, 3], 4)["pass_k"], {})
        with self.assertRaises(ValueError):
            aggregate(rows + rows[:1], [0, 1, 2, 3], 4)

    def test_missing_usage_is_not_estimated(self):
        row = self.records()[0]
        row["api_events"] = [{"source": "POLICY", "response": {}, "usage": None}]
        self.assertIsNone(aggregate([row], [0], 1)["tokens"]["POLICY"]["mean_reported_total_tokens_per_attempted_trial"])

    def test_atomic_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            atomic_json(path, {"round": 1})
            atomic_json(path, {"round": 2})
            self.assertEqual(json.loads(path.read_text()), {"round": 2})
            self.assertEqual(len(list(Path(folder).iterdir())), 1)


class TaxonomyTests(unittest.TestCase):
    def test_business_error_does_not_prove_wrong_arguments(self):
        row = {"task_id": 0, "trial": 0, "status": "completed", "reward": 0, "termination": "user_stop",
               "observations": [{"observation": "Error: Order cannot be cancelled"}]}
        self.assertEqual(classify(row)["primary"], "未确定")

    def test_source_error(self):
        row = {"task_id": 0, "trial": 0, "status": "system_error", "termination": "USER_SIM:APIConnectionError"}
        self.assertEqual(classify(row)["primary"], "用户模拟器问题")


class EnvironmentTests(unittest.TestCase):
    def test_real_env_instances_do_not_share_state(self):
        one, two = make_env("retail", "test", 0), make_env("retail", "test", 0)
        one.data["__offline_isolation_test__"] = True
        self.assertNotIn("__offline_isolation_test__", two.data)

    def test_rollout_matches_official_tc_on_same_script(self):
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent
        env = make_env("retail", "test", 0)
        tool_msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "fixture-call", "type": "function",
                    "function": {"name": "get_user_details", "arguments": json.dumps({"user_id": env.task.user_id})}}]}
        outputs = [response(tool_msg, "tool_calls"), response({"role": "assistant", "content": "Here are your account details."})]

        def make_user():
            return ScriptedEndpoint([response({"role": "assistant", "content": "Please help with my account."}),
                                     response({"role": "assistant", "content": "###STOP###"})], "USER_SIM")

        policy = ScriptedEndpoint(outputs, "POLICY")
        config = {"seed": 42, "trials": 4, "user_temperature": 0, "domain": "retail", "split": "test",
                  "max_steps": 30, "max_tokens": 1024, "temperature": 0.7, "top_p": 1.0}
        actual = rollout(config, 0, 0, policy, make_user())
        env.user = Simulator(make_user(), 42)
        iterator = iter(outputs)

        def official_completion(**kwargs):
            out = next(iterator)
            return SimpleNamespace(choices=out.choices, _hidden_params={"response_cost": 0})

        with patch("tau_bench.agents.tool_calling_agent.completion", side_effect=official_completion):
            expected = ToolCallingAgent(env.tools_info, env.wiki, "offline-fixture", "openai").solve(env, 0)
        self.assertEqual(actual["status"], "completed")
        self.assertEqual(actual["reward"], expected.reward)
        self.assertEqual(actual["tool_calls"], 1)
        self.assertEqual(actual["rounds"], 2)
        self.assertEqual(actual["messages"][-1]["content"], "###STOP###")

    def test_generation_truncation_keeps_raw_response(self):
        policy = ScriptedEndpoint([response({"role": "assistant", "content": "partial"}, "length")], "POLICY")
        user = ScriptedEndpoint([response({"role": "assistant", "content": "Hello"})], "USER_SIM")
        config = {"seed": 42, "trials": 4, "user_temperature": 0, "domain": "airline", "split": "test",
                  "max_steps": 30, "max_tokens": 1024, "temperature": 0.7, "top_p": 1.0}
        result = rollout(config, 0, 0, policy, user)
        self.assertEqual(result["status"], "truncated")
        self.assertEqual(result["messages"][-1]["content"], "partial")


if __name__ == "__main__":
    unittest.main()
