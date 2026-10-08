"""The OPSD paper's baseline tables (baseline/README.md): every arm of arms.tsv, over all of its runs.

Per arm, on the 70 dev tasks:
- resolved per run, and the tasks resolved in every run and in any run;
- against the arm's reference (arms.tsv `ref`), task by task:
  - each task's resolve rate, averaged over each arm's runs;
  - the tasks where the arm does better and where it does worse;
  - the difference in tasks resolved, with a bootstrap 95% CI over tasks;
  - a two-sided sign test over the tasks that differ;
- from the traces: calls and agent time per task, the first source edit, stale repeats, malformed calls, and how bad
  calls chain. The rules are those of ../adaptive_inference/trigger_replay.py, with diagnose_run.py's full tool list;
- from calls.jsonl, where the run dir has it: output tokens per task, and the share that is thinking. That share is
  Gemma 4's <|channel> ... <channel|> block (ids 100 and 101), counted as baseline/diagnose_run.py counts it.

Only unpaced runs in the PINS=1 env count, as every v28 baseline is. Any other run is listed and left out.
stdlib only. On Explorer it runs through srun (opsd_baselines.sh report). Locally it runs on pulled run dirs,
where traces and task_results.jsonl suffice:
    python baseline_report.py [--results DIR] [--ids FILE] [--arms FILE] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(f"/scratch/{os.environ.get('USER', '')}/gemma4")
SKIPPED = "SKIPPED_DEADLINE"
THINK_OPEN, THINK_CLOSE = 100, 101   # Gemma 4's <|channel> and <channel|> token ids (baseline/diagnose_run.py)
MAIN_AGENT = "swe_baseline_agent"
# (required, optional) argument names of every tool our zips declare: baseline/diagnose_run.py TOOL_ARGS
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
BOOTSTRAP = 10_000


# ---- arms and runs
def read_arms(path: Path) -> list[dict]:
    keys = ("arm", "agent", "zip", "first_run", "extra_runs", "ref", "kaggle", "about")
    arms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            row = dict(zip(keys, line.split("\t")))
            row["extra_runs"] = [] if row["extra_runs"] == "-" else row["extra_runs"].split(",")
            arms.append(row)
    return arms


def arm_runs(arm: dict, results: Path) -> list[Path]:
    """The first run, the extra runs, then <first_run>_r<k> in order of k; only those that exist."""
    first = arm["first_run"]
    later = sorted((p for p in results.glob(f"{first}_r*") if re.fullmatch(re.escape(first) + r"_r\d+", p.name)),
                   key=lambda p: int(p.name.rsplit("_r", 1)[1]))
    return [p for p in [results / first, *(results / r for r in arm["extra_runs"]), *later] if p.is_dir()]


# ---- one run
def jsonl(path: Path):
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def source_edit(fn: str, args: dict) -> bool:
    """An edit of the library's source: not a test, and not a new file at the repo root (scratch scripts)."""
    path = re.sub(r"^(/workspace/|\./)+", "", str(args.get("filepath") or ""))
    return fn in ("edit_file", "write_file") and bool(path) and not TEST_PATH.search(path) and (fn == "edit_file" or "/" in path)


def trace_facts(t: dict) -> dict:
    """The main agent's calls, with what the hygiene rules need, and the task's time span (diagnose_run.py's)."""
    calls, last_seen, n_edits = [], {}, 0
    start_s, end_s, first_edit = None, 0.0, None
    for s in t.get("steps", []):
        ex = s.get("extra") or {}
        el = ex.get("elapsed_s")
        if el is not None:
            end_s = max(end_s, el)
            if start_s is None and ex.get("event_type") == "task_prompt":
                start_s = el
        tcs = s.get("tool_calls") or []
        if s.get("source") != "agent" or ex.get("author") not in (None, MAIN_AGENT) or not tcs:
            continue
        fn, args = tcs[-1].get("function_name"), tcs[-1].get("arguments") or {}
        key = (fn, json.dumps(args, sort_keys=True, default=str))
        req, opt = TOOL_ARGS.get(fn, (set(), set()))
        try:
            d = json.loads((s.get("observation") or {}).get("content") or "")
        except ValueError:
            d = None
        ok = isinstance(d, dict) and d.get("status") == "ok"
        err = isinstance(d, dict) and (d.get("status") == "error" or ("error" in d and "status" not in d))
        calls.append({"stale": last_seen.get(key) == n_edits, "err": bool(err),
                      "malformed": fn in TOOL_ARGS and bool((set(args) - req - opt) or (req - set(args)))})
        last_seen[key] = n_edits
        if fn in ("edit_file", "write_file") and ok:
            n_edits += 1
            if first_edit is None and source_edit(fn, args):
                first_edit = (tcs[-1].get("extra") or {}).get("elapsed_s")
    start_s = start_s or 0.0
    return {"calls": calls, "start_s": start_s, "end_s": end_s,
            "first_edit_s": None if first_edit is None else max(0.0, first_edit - start_s)}


def out_ids(c: dict) -> list:
    ids = c.get("completion_ids") or []
    return ids[0] if ids and isinstance(ids[0], list) else ids


def n_think(c: dict) -> int:
    ids = out_ids(c)
    if not ids or ids[0] != THINK_OPEN:
        return 0
    return ids.index(THINK_CLOSE) + 1 if THINK_CLOSE in ids else len(ids)


def load_run(run: Path, ids: set | None) -> dict:
    meta, rows, shard_of = {}, {}, {}
    for shard in sorted(p for p in run.glob("shard_*") if p.is_dir()):
        if not meta and (shard / "run_meta.json").is_file():
            meta = json.loads((shard / "run_meta.json").read_text(encoding="utf-8"))
        for r in jsonl(shard / "task_results.jsonl"):
            if r.get("error") != SKIPPED and (ids is None or r["instance_id"] in ids):
                rows[r["instance_id"]], shard_of[r["instance_id"]] = r, shard   # a later rerun wins
    budget_s = 60.0 * float((meta.get("budget") or {}).get("max_time_minutes") or 4)
    tasks = {}
    for t, r in rows.items():
        f = shard_of[t] / "traces" / f"trace_{t}.json"
        facts = trace_facts(json.loads(f.read_text(encoding="utf-8"))) if f.is_file() else None
        x = {"resolved": bool(r.get("resolved")), "facts": facts, "out": 0, "think": 0}
        if facts:   # a task stopped by the clock used its whole budget (diagnose_run.py)
            x["agent_s"] = budget_s if "session timeout" in (r.get("error") or "") else facts["end_s"] - facts["start_s"]
        tasks[t] = x
    have_calls = False
    for shard in sorted(set(shard_of.values())):
        mine = {t for t, s in shard_of.items() if s == shard}
        for c in jsonl(shard / "calls.jsonl"):
            if c.get("task") in mine:
                have_calls = True
                tasks[c["task"]]["out"] += (c.get("usage") or {}).get("completion_tokens") or len(out_ids(c))
                tasks[c["task"]]["think"] += n_think(c)
    why = []
    if meta and meta.get("pins") is not True:
        why.append("not in the PINS=1 env")
    if meta and (meta.get("pace") or "none") != "none":
        why.append(f"paced ({meta['pace']})")
    if not meta:
        why.append("no run_meta.json")
    return {"name": run.name, "meta": meta, "tasks": tasks, "have_calls": have_calls, "excluded": "; ".join(why)}


# ---- statistics
def sign_test(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def rates(runs: list[dict]) -> dict[str, float]:
    got = defaultdict(list)
    for r in runs:
        for t, x in r["tasks"].items():
            got[t].append(x["resolved"])
    return {t: sum(v) / len(v) for t, v in got.items()}


def paired(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict:
    common = sorted(set(a) & set(b))
    d = [a[t] - b[t] for t in common]
    if not d:
        return {"common": 0}
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) for _ in range(BOOTSTRAP))
    better, worse = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return {"common": len(common), "better": better, "worse": worse, "delta": sum(d),
            "ci95": [boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP) - 1]], "p": sign_test(better, worse),
            "better_ids": [t for t, x in zip(common, d) if x > 0], "worse_ids": [t for t, x in zip(common, d) if x < 0]}


def behaviour(runs: list[dict]) -> dict | None:
    xs = [x for r in runs for x in r["tasks"].values() if x["facts"]]
    if not xs:
        return None
    calls = [c for x in xs for c in x["facts"]["calls"]]
    bad = lambda c: c["stale"] or c["malformed"]
    pairs = [(cs[i - 1], cs[i]) for x in xs for cs in (x["facts"]["calls"],) for i in range(1, len(cs))]
    after_bad = [bad(b) for a, b in pairs if bad(a)]
    after_clean = [bad(b) for a, b in pairs if not bad(a) and not a["err"]]
    edits = [x["facts"]["first_edit_s"] for x in xs if x["facts"]["first_edit_s"] is not None]
    with_calls = [x for r in runs if r["have_calls"] for x in r["tasks"].values()]
    share = lambda k, n: k / n if n else None
    return {
        "task_runs": len(xs), "calls_per_task": st.median(len(x["facts"]["calls"]) for x in xs),
        "agent_s": st.mean(x["agent_s"] for x in xs),
        "first_edit_s": st.median(edits) if edits else None, "never_edited": len(xs) - len(edits),
        "stale": share(sum(c["stale"] for c in calls), len(calls)),
        "malformed": share(sum(c["malformed"] for c in calls), len(calls)),
        "bad_after_bad": share(sum(after_bad), len(after_bad)), "bad_after_clean": share(sum(after_clean), len(after_clean)),
        "out_per_task": st.mean(x["out"] for x in with_calls) if with_calls else None,
        "think_share": share(sum(x["think"] for x in with_calls), sum(x["out"] for x in with_calls)) if with_calls else None,
    }


# ---- report
def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def num(v, fmt="{:.0f}") -> str:
    return "–" if v is None else fmt.format(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--ids", type=Path, default=ROOT / "train/rft0/dev_sound_ids.txt",
                    help="the tasks to count (default: the 70 dev-sound tasks); 'none' counts every task a run has")
    ap.add_argument("--arms", type=Path, default=HERE / "arms.tsv")
    ap.add_argument("--json", type=Path, help="also write everything as JSON here")
    a = ap.parse_args()
    ids = None if str(a.ids) == "none" or not a.ids.is_file() else set(a.ids.read_text().split())
    rng = random.Random(0)

    arms = read_arms(a.arms)
    data = {}
    for arm in arms:
        loaded = [load_run(p, ids) for p in arm_runs(arm, a.results)]
        data[arm["arm"]] = {"arm": arm, "runs": [r for r in loaded if not r["excluded"]],
                            "excluded": [r for r in loaded if r["excluded"]]}
    for d in data.values():
        d["rates"] = rates(d["runs"])
    n_ids = len(ids) if ids else None
    print(f"# OPSD baselines: {a.results}, {n_ids or 'all'} tasks{f' from {a.ids.name}' if ids else ''}\n")

    print("| arm | runs: resolved | in every run | in any run | vs ref: better / worse | Δ resolved (95% CI) | sign test p |")
    print("|---|---|---|---|---|---|---|")
    out = {}
    for name, d in data.items():
        arm, runs = d["arm"], d["runs"]
        per_run = ", ".join(f"{r['name']} {sum(x['resolved'] for x in r['tasks'].values())}/{len(r['tasks'])}" for r in runs)
        common = set.intersection(*(set(r["tasks"]) for r in runs)) if runs else set()
        every = sum(all(r["tasks"][t]["resolved"] for r in runs) for t in common)
        anyrun = sum(any(r["tasks"].get(t, {}).get("resolved") for r in runs) for t in d["rates"])
        ref = data.get(arm["ref"])
        cmp = paired(d["rates"], ref["rates"], rng) if ref and ref["runs"] and runs else None
        if cmp and cmp["common"]:
            vs = f"{arm['ref']}: {cmp['better']} / {cmp['worse']}"
            delta = f"{cmp['delta']:+.1f} ({cmp['ci95'][0]:+.1f} to {cmp['ci95'][1]:+.1f}) over {cmp['common']}"
            p = f"{cmp['p']:.3f}"
        else:
            vs, delta, p = (f"{arm['ref']}: no runs" if arm["ref"] != "-" else "–"), "–", "–"
        none = f"no comparable runs ({len(d['excluded'])} left out)" if d["excluded"] else "no runs yet"
        print(f"| {name} | {per_run or none} | {every if runs else '–'} | {anyrun if runs else '–'} | {vs} | {delta} | {p} |")
        d["paired"], d["behaviour"] = cmp, behaviour(runs)
        out[name] = {"about": arm["about"], "kaggle": arm["kaggle"],
                     "runs": [{"name": r["name"], "resolved": sum(x["resolved"] for x in r["tasks"].values()),
                               "finished": len(r["tasks"]), "zip_sha256": r["meta"].get("agent_zip_sha256"),
                               "start": r["meta"].get("start"), "calls_jsonl": r["have_calls"]} for r in runs],
                     "excluded": [{"name": r["name"], "why": r["excluded"]} for r in d["excluded"]],
                     "rates": d["rates"], "paired": cmp, "behaviour": d["behaviour"]}

    print("\n| arm | tool calls / task (median) | agent s / task | first source edit (median s; never) | stale repeat | malformed |"
          " next bad, after bad / after clean | output tokens / task | thinking share |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, d in data.items():
        b = d["behaviour"]
        if not b:
            print(f"| {name} | {'no traces' if d['runs'] else 'no runs yet'} |  |  |  |  |  |  |  |")
            continue
        print(f"| {name} | {num(b['calls_per_task'])} | {num(b['agent_s'])} | {num(b['first_edit_s'])}; {b['never_edited']} |"
              f" {pct(b['stale'])} | {pct(b['malformed'])} | {pct(b['bad_after_bad'])} / {pct(b['bad_after_clean'])} |"
              f" {num(b['out_per_task'], '{:,.0f}')} | {pct(b['think_share'])} |")

    excluded = [(name, r) for name, d in data.items() for r in d["excluded"]]
    if excluded:
        print("\nLeft out: " + "; ".join(f"{r['name']} ({name}: {r['excluded']})" for name, r in excluded))
    print("\nBehaviour pools every run's tasks that have a trace. Bad = a stale repeat or malformed arguments.")
    print("Agent time counts a task stopped by the clock as its whole budget. Output tokens and thinking need calls.jsonl.")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
