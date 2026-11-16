from collections import Counter
from math import comb
from statistics import mean


def aggregate(records, task_ids, trials):
    by_key = {(r["task_id"], r["trial"]): r for r in records}
    if len(by_key) != len(records):
        raise ValueError("duplicate task/trial results")
    expected = {(task, trial) for task in task_ids for trial in range(trials)}
    if not set(by_key).issubset(expected):
        raise ValueError("unexpected task/trial results")
    full_tasks = [t for t in task_ids if all((t, j) in by_key for j in range(trials))]
    pass_k = {}
    # Do not publish a partial-subset success score as the planned benchmark.
    if len(by_key) == len(expected) and expected:
        successes = [sum(by_key[t, j]["status"] == "completed" and by_key[t, j]["reward"] == 1 for j in range(trials)) for t in task_ids]
        pass_k = {str(k): mean(comb(c, k) / comb(trials, k) for c in successes) for k in range(1, min(trials, 4) + 1)}
    measured = [r for r in records if r["status"] in ("completed", "truncated", "max_steps")]
    token_stats = {}
    for source in ("POLICY", "USER_SIM"):
        all_events = [e for r in records for e in r.get("api_events", []) if e["source"] == source]
        calls = [e for e in all_events if "response" in e]
        usages = [e["usage"] for e in calls if e.get("usage") is not None]
        sums = {key: sum(u.get(key, 0) or 0 for u in usages) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        token_stats[source] = {"source": "API usage", "successful_calls": len(calls), "usage_present_calls": len(usages),
                               "error_attempts": sum("error" in e for e in all_events),
                               "usage_complete": len(calls) == len(usages), "reported_totals": sums,
                               "mean_reported_total_tokens_per_attempted_trial": sums["total_tokens"] / len(records) if records and len(calls) == len(usages) else None}
    return {"expected_trials": len(expected), "recorded_trials": len(records), "complete_tasks": len(full_tasks),
            "missing_trials": len(expected - set(by_key)), "status_counts": dict(Counter(r["status"] for r in records)),
            "termination_counts": dict(Counter(r["termination"] for r in records)),
            "pass_k": pass_k, "pass_k_scope": "all planned trials; system errors count as non-success; empty until all recorded",
            "mean_rounds": mean(r["rounds"] for r in measured) if measured else None,
            "mean_tool_calls": mean(r["tool_calls"] for r in measured) if measured else None,
            "rounds_denominator": len(measured), "tokens": token_stats}
