"""Offline replay of per-call thinking triggers over recorded agent traces (no GPU, read-only).

Reads the ATIF traces and task_results.jsonl of each run given (typically one thinking-off and one
thinking-on run), and reports:
- call hygiene: stale repeats, malformed arguments, error results, source edits;
- how bad calls chain: P(bad | previous bad) against P(bad | previous clean and ok);
- how often each trigger of a per-call thinking switch would fire;
- for a run that thinks, how the model's own output tokens (mostly thinking) are spread over those states;
- for a run with an advisor sub-agent (agentic_v2), where the model calls it.

Definitions:
- malformed: argument names don't match the tool's (unknown or missing required), as in the main-track
  repo's baseline/diagnose_run.py (TOOL_ARGS).
- stale repeat: the same tool and arguments as an earlier call, with no successful edit_file or write_file
  in between. A re-run of the same test after an edit is not stale.
- bad: stale repeat or malformed. Both can be detected on the drafted call, before it executes.
- before the call: triggers a controller can read off the conversation before the model writes anything for this call
  (the first call; the previous call was bad or returned an error). After a draft: triggers that need the drafted
  call itself, kept for comparison.

Usage: one or more run dirs, each with shard_*/traces/ and shard_*/task_results.jsonl. The numbers in
README.md come from the main-track repo's audit copies of two v28 runs, plus agentic_v2's run, whose traces and
task results were pulled from /scratch/$USER/gemma4/results/agentic_v2_v28_zip:
    python -I trigger_replay.py <Gemma4_kaggle_repo>/logs/explorer_audit_20261005/baseline_nothink_v28_zip \
        <Gemma4_kaggle_repo>/logs/explorer_audit_20261005/base_v28_zip <pulled>/agentic_v2_v28_zip
"""
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAIN_AGENT = "swe_baseline_agent"
# (required, optional) argument names; the main-track repo's baseline/diagnose_run.py TOOL_ARGS
TOOL_ARGS = {
    "run_command": ({"command"}, set()),
    "read_file": ({"filepath"}, {"start_line", "end_line"}),
    "edit_file": ({"filepath", "old_string", "new_string"}, {"allow_multiple"}),
    "write_file": ({"filepath", "content"}, set()),
    "submit_patch": (set(), set()),
    "get_status": (set(), set()),
    "get_code_neighbors": ({"node"}, {"edge_type", "max_neighbors"}),
    "search_similar_code": ({"query"}, {"k"}),
    "get_code_subgraph": ({"nodes"}, set()),
    "list_skills": (set(), set()),
    "load_skill": ({"skill_name"}, set()),
    "load_skill_resource": ({"skill_name", "file_path"}, set()),
    "run_skill_script": ({"skill_name", "file_path"}, {"args", "short_options", "positional_args"}),
    "code_analyzer_agent": ({"request"}, set()),
    "advisor": ({"request"}, set()),
}
TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


def source_edit(fn: str, args: dict) -> bool:
    """An edit of the library's source: not a test, and not a new file at the repo root (scratch scripts)."""
    path = re.sub(r"^(/workspace/|\./)+", "", str(args.get("filepath") or ""))
    return fn in ("edit_file", "write_file") and bool(path) and not TEST_PATH.search(path) and (fn == "edit_file" or "/" in path)


def trace_calls(t: dict) -> list[dict]:
    """The main agent's calls in order, with the facts each trigger needs."""
    out, last_seen, n_edits = [], {}, 0
    for s in t["steps"]:
        if s.get("source") != "agent" or (s.get("extra") or {}).get("author") not in (None, MAIN_AGENT):
            continue
        tcs = s.get("tool_calls") or []
        if not tcs:
            continue
        fn, args = tcs[-1].get("function_name"), tcs[-1].get("arguments") or {}
        key = (fn, json.dumps(args, sort_keys=True, default=str))
        req, opt = TOOL_ARGS.get(fn, (set(), set()))
        try:
            d = json.loads((s.get("observation") or {}).get("content") or "")
        except ValueError:
            d = None
        err = isinstance(d, dict) and (d.get("status") == "error" or ("error" in d and "status" not in d))
        out.append({
            "fn": fn,
            "src_edited_before": any(c["src_edit"] for c in out),
            "stale": last_seen.get(key) == n_edits,
            "malformed": fn in TOOL_ARGS and bool((set(args) - req - opt) or (req - set(args))),
            "err": bool(err),
            "src_edit": source_edit(fn, args) and not err,
            "out": (s.get("metrics") or {}).get("completion_tokens") or 0,
        })
        last_seen[key] = n_edits
        if fn in ("edit_file", "write_file") and isinstance(d, dict) and d.get("status") == "ok":
            n_edits += 1
    return out


def load(run: Path) -> tuple[dict, dict]:
    res, calls = {}, {}
    for shard in sorted(run.glob("shard_*")):
        for line in (shard / "task_results.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                res[r["instance_id"]] = r
        for f in (shard / "traces").glob("trace_*.json"):
            calls[f.stem[len("trace_"):]] = trace_calls(json.loads(f.read_text(encoding="utf-8")))
    return res, calls


def pct(k: float, n: float) -> str:
    return f"{100 * k / n:.1f}%" if n else "-"


def bad(c: dict) -> bool:
    return c["stale"] or c["malformed"]


def report(name: str, res: dict, calls: dict) -> None:
    allc = [c for cs in calls.values() for c in cs]
    n = len(allc)
    print(f"\n== {name}: {len(calls)} tasks, {n} calls, {sum(bool(r.get('resolved')) for r in res.values())} resolved")
    print(f"stale repeat {pct(sum(c['stale'] for c in allc), n)}, malformed {pct(sum(c['malformed'] for c in allc), n)}, "
          f"either (bad) {pct(sum(bad(c) for c in allc), n)}, error result {pct(sum(c['err'] for c in allc), n)}, "
          f"source edit {pct(sum(c['src_edit'] for c in allc), n)}")
    pairs = [(cs[i - 1], cs[i]) for cs in calls.values() for i in range(1, len(cs))]
    after_bad = [b for a, b in pairs if bad(a)]
    after_clean = [b for a, b in pairs if not bad(a) and not a["err"]]
    print(f"P(bad | previous bad) = {pct(sum(map(bad, after_bad)), len(after_bad))}; "
          f"P(bad | previous clean and ok) = {pct(sum(map(bad, after_clean)), len(after_clean))}; "
          f"P(error result | previous clean and ok, this call not bad) = "
          f"{pct(sum(b['err'] for b in after_clean if not bad(b)), sum(1 for b in after_clean if not bad(b)))}")
    firsts = [next(i for i, c in enumerate(cs) if bad(c)) + 1 for cs in calls.values() if any(map(bad, cs))]
    q = st.quantiles(firsts, n=4)
    print(f"tasks with a bad call: {len(firsts)}/{len(calls)}; the first at call {st.median(firsts):.0f} (median; IQR {q[0]:.0f}-{q[2]:.0f})")
    after_fail = lambda cs, i: i > 0 and (bad(cs[i - 1]) or cs[i - 1]["err"])
    triggers = {
        "before the call: previous call bad or errored": after_fail,
        "before the call: that, or the first call": lambda cs, i: i == 0 or after_fail(cs, i),
        "after a draft: draft is bad": lambda cs, i: bad(cs[i]),
        "after a draft: draft bad or a source edit, or either before-call trigger":
            lambda cs, i: i == 0 or bad(cs[i]) or cs[i]["src_edit"] or after_fail(cs, i),
    }
    for label, fire in triggers.items():
        hits = sum(fire(cs, i) for cs in calls.values() for i in range(len(cs)))
        print(f"  trigger '{label}': {pct(hits, n)} of these calls")
    advisor_placement(calls)


def advisor_placement(calls: dict) -> None:
    """Where the model calls its advisor sub-agent itself (agentic_v2): the model-decides arm."""
    where, after_bad = Counter(), Counter()
    for cs in calls.values():
        for i, c in enumerate(cs):
            if c["fn"] == "advisor":
                where["after a bad or errored call" if i > 0 and (bad(cs[i - 1]) or cs[i - 1]["err"])
                      else "after a source edit, previous call clean" if c["src_edited_before"]
                      else "before the first source edit"] += 1
            if i > 0 and bad(cs[i - 1]):
                after_bad["the advisor" if c["fn"] == "advisor" else "bad again" if bad(c) else "something else"] += 1
    if not where:
        return
    per_task = Counter(sum(c["fn"] == "advisor" for c in cs) for cs in calls.values())
    print(f"  advisor: {sum(where.values())} calls in {len(calls) - per_task[0]} tasks "
          f"(calls per task: {dict(sorted(per_task.items()))}); " + ", ".join(f"{k} {v}" for k, v in where.most_common()))
    n = sum(after_bad.values())
    print(f"  the call right after a bad call: " + ", ".join(f"{k} {v} ({pct(v, n)})" for k, v in after_bad.most_common()))


def thinking_spread(calls: dict) -> None:
    groups = defaultdict(list)
    long_by = Counter()
    for cs in calls.values():
        for i, c in enumerate(cs):
            if i == 0:
                g = "first call"
            elif bad(cs[i - 1]) or cs[i - 1]["err"]:
                g = "after a bad or errored call"
            elif c["src_edit"]:
                g = "source edit"
            else:
                g = "routine"
            groups[g].append(c["out"])
            if c["out"] > 1000:
                long_by[g if g != "routine" else f"routine {c['fn']}"] += 1
    total = sum(sum(v) for v in groups.values())
    ncalls = sum(len(v) for v in groups.values())
    print("   output tokens per call by state (a plain thinking-off call is ~20-40 tokens)")
    for g, v in sorted(groups.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {g:28s} {pct(len(v), ncalls):>6s} of calls, median {st.median(v):4.0f}, mean {st.mean(v):4.0f}, "
              f"{pct(sum(v), total):>6s} of all output tokens")
    print(f"  calls over 1,000 output tokens: {sum(long_by.values())}, by state: {dict(long_by.most_common())}")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        run = Path(arg)
        res, calls = load(run)
        report(run.name, res, calls)
        outs = [c["out"] for cs in calls.values() for c in cs]
        if outs and st.mean(outs) > 100:   # a thinking run; plain calls are ~20-40 output tokens
            thinking_spread(calls)


if __name__ == "__main__":
    main()
