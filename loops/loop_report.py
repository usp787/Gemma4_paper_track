"""Loop metrics of agent runs (README.md, experiment 2): how often identical calls start, lock in and end, per run and
task group, and what the served chat template did to the prompts. stdlib only, as baseline/baseline_report.py.

    python loop_report.py --runs <run dir> [<run dir> ...] [--group NAME=IDS_FILE ...] [--json OUT]
    python loop_report.py --arm ctrl=<run>,<run_r2> --arm ctrl_t1=<run>,<run_r2> --pair ctrl_t1:ctrl [--group ...]

The rules are Gate 1's (gate1.py label_calls, on trigger_replay.py's): a stale repeat is the same tool and arguments as
an earlier call with no successful edit between; bad is stale or malformed; a run is a stretch of identical calls in a
row. Per run and group:
- tasks, resolved; tool calls per task (median); stale and malformed shares; P(bad | previous bad) and P(bad |
  previous clean and ok);
- copies: the share of calls that are the 2nd identical call in a row (a copy starts), and lock-in, P(the next call is
  one more identical copy | the last k calls identical) for k = 1, 2, 3+;
- runs of 3+ and of 5+ identical calls, and the tasks with one; loop tasks, with at least half their calls bad;
- how each run of 3+ ends, checked in this order: the task ends inside it (task end); the budget warning came in one of
  its results (budget warning); the next call's prompt is shorter, so a compaction came between (compaction); else own;
- from calls.jsonl, the main agent's calls by their place in the task: the share whose prompt ends with the empty
  thought <|channel>thought\\n<channel|> (the template's prefill) and the share whose completion opens a thought (the
  model writing the block itself). Without a group, tasks are grouped by repo: rich, and the other three (dev).

Arms (round 2's G3, README.md section 3): --arm NAME=RUN[,RUN...] pools an arm's passes, and --pair A:B compares arm A
with arm B task by task. Then the report adds, per group and over all tasks:
- each arm's pooled metrics, with stale repeats split into copies (the call just before, again) and returns (an earlier
  call, again: cycles), runs of 3+ by type (search, other command, read, edit, submit, malformed), and what follows
  a run of 3+ that ends before the task does: the call re-sent later (re-entry), the next call working (not bad, and
  the tool returned a result: a command's output, a read, an edit), a successful source edit later in the task;
- per pair, A − B for each metric, with a 95% interval from resampling tasks (--reps, --seed). Each resampled task
  brings all of its runs in both arms, so the passes stay paired with their task;
- per arm with 2 or more runs, run 2 − run 1 the same way: how much one pass varies;
- every run's provenance: wheelhouse, harness, vLLM build, zip, chat template, sampling, PINS, pace, budget, GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "gate1"), str(HERE.parent / "baseline")]
import gate1  # noqa: E402  # stdlib at import time: label_calls, main_agent, iter_jsonl
import dose  # noqa: E402  # stdlib at import time: is_search
from baseline_report import source_edit  # noqa: E402  # stdlib: an edit of the library's source, not a test

EMPTY_IDS = [100, 45518, 107, 101]   # <|channel>thought\n<channel|> in Gemma 4's tokenizer (gate1 Vocab.empty_thought)
CH_OPEN = 100
SKIPPED = "SKIPPED_DEADLINE"
PLACES = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 10 ** 6))   # a main-agent call's place in its task, 1-based


def place_of(n: int) -> str:
    a, b = next((a, b) for a, b in PLACES if a <= n <= b)
    return str(a) if a == b else f"{a}+" if b >= 10 ** 6 else f"{a}-{b}"


def first_ids(c: dict) -> list:
    ids = c.get("completion_ids") or []
    return ids[0] if ids and isinstance(ids[0], list) else ids


def load_run(run: Path) -> dict:
    """task -> {resolved, calls (gate1 labels), prompt_ends, opens}, from every shard; a later rerun of a task wins."""
    rows, shard_of = {}, {}
    for shard in sorted(p for p in run.glob("shard_*") if p.is_dir()):
        for r in gate1.iter_jsonl(shard / "task_results.jsonl"):
            if r.get("error") != SKIPPED:
                rows[r["instance_id"]], shard_of[r["instance_id"]] = r, shard
    tasks = {}
    for t, r in rows.items():
        f = shard_of[t] / "traces" / f"trace_{t}.json"
        if not f.is_file():
            continue
        trace = json.loads(f.read_text(encoding="utf-8"))
        tasks[t] = {"resolved": bool(r.get("resolved")), "calls": gate1.label_calls(trace, gate1.main_agent(trace)),
                    "prompt_ends": [], "opens": []}
    for shard in sorted(set(shard_of.values())):
        mine = {t for t, s in shard_of.items() if s == shard and t in tasks}
        recs = defaultdict(list)
        for c in gate1.iter_jsonl(shard / "calls.jsonl"):
            if c.get("task") in mine and c.get("n_tools") and c.get("prompt_ids"):
                ids = first_ids(c)
                recs[c["task"]].append((c.get("i", 0), c["n_tools"], c["prompt_ids"][-4:] == EMPTY_IDS,
                                        bool(ids) and ids[0] == CH_OPEN))
        if not recs:
            continue
        main = Counter(n for rs in recs.values() for _, n, _, _ in rs).most_common(1)[0][0]   # the main agent's tools
        for t, rs in recs.items():
            calls = [x for x in sorted(rs) if x[1] == main]
            tasks[t]["prompt_ends"] = [x[2] for x in calls]
            tasks[t]["opens"] = [x[3] for x in calls]
    template = (run / ".chat_template").read_text().strip() if (run / ".chat_template").is_file() else "model"
    return {"name": run.name, "tasks": tasks, "template": template, "prov": provenance(run, template)}


SAMPLING_KEYS = re.compile(r"^(temperature|top_p|top_k):[ \t]*([0-9.]+)", re.M)


def provenance(run: Path, template: str) -> dict:
    """What the run ran on: its first run_meta.json, and the sampling of every agent dir eval_gpu.slurm extracted."""
    meta = {}
    for shard in sorted(p for p in run.glob("shard_*") if p.is_dir()):
        if (shard / "run_meta.json").is_file():
            meta = json.loads((shard / "run_meta.json").read_text(encoding="utf-8"))
            break
    samplings = set()
    for f in sorted(run.glob("agent_j*/configs/sampling.yaml")):
        samplings.add(", ".join(f"{k} {v}" for k, v in SAMPLING_KEYS.findall(f.read_text(encoding="utf-8"))))
    short = lambda x: str(x)[:12] if x else "-"
    budget = meta.get("budget") or {}
    return {"wheelhouse": meta.get("wheelhouse_version") or "-", "harness": short(meta.get("harness_sig")),
            "vllm": short(meta.get("host_vllm_sha")), "zip": short(meta.get("agent_zip_sha256")), "template": template,
            "sampling": " | ".join(sorted(samplings)) or "-", "pins": meta.get("pins", "-"), "pace": meta.get("pace") or "-",
            "budget": (f"{budget.get('max_time_minutes')} min, {budget.get('max_tool_calls')} calls, "
                       f"{budget.get('max_turns')} turns") if budget else "-", "gpu": meta.get("gpu") or "-",
            "agreed": len(samplings) <= 1}


def runs_of(cs: list[dict]) -> list[tuple[int, int]]:
    """(start, end) of every stretch of identical calls in a row, end inclusive."""
    out, s = [], 0
    for i in range(1, len(cs) + 1):
        if i == len(cs) or cs[i]["key"] != cs[s]["key"]:
            out.append((s, i - 1))
            s = i
    return out


def end_of(cs: list[dict], s: int, e: int) -> str:
    if e == len(cs) - 1:
        return "task end"
    if any(isinstance(cs[j]["obs"], dict) and "budget_warning" in cs[j]["obs"] for j in range(s, e + 1)):
        return "budget warning"
    p = lambda c: ((c["step"].get("metrics") or {}).get("prompt_tokens") or 0)
    return "compaction" if p(cs[e + 1]) < p(cs[e]) else "own"


def share(k, n):
    return k / n if n else None


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def metrics(tasks: dict) -> dict:
    cs_all = [x["calls"] for x in tasks.values()]
    calls = [c for cs in cs_all for c in cs]
    pairs = [(cs[i - 1], cs[i]) for cs in cs_all for i in range(1, len(cs))]
    lock = defaultdict(lambda: [0, 0])   # k -> [identical next, all next]
    for a, b in pairs:
        k = min(a["run_k"], 3)
        lock[k][1] += 1
        lock[k][0] += b["same_as_prev"]
    stretches = [(cs, s, e) for cs in cs_all for s, e in runs_of(cs)]
    long3 = [(cs, s, e) for cs, s, e in stretches if e - s + 1 >= 3]
    ends = Counter(end_of(cs, s, e) for cs, s, e in long3)
    by_place = defaultdict(lambda: [0, 0, 0])   # place -> [prompt ends with the block, completion opens a thought, n]
    for x in tasks.values():
        for n, (pe, op) in enumerate(zip(x["prompt_ends"], x["opens"]), 1):
            b = by_place[place_of(n)]
            b[0] += pe
            b[1] += op
            b[2] += 1
    n_rec = sum(b[2] for b in by_place.values())
    return {
        "tasks": len(tasks), "resolved": sum(x["resolved"] for x in tasks.values()), "calls": len(calls),
        "calls_per_task": st.median(len(cs) for cs in cs_all) if cs_all else None,
        "stale": share(sum(c["stale"] for c in calls), len(calls)),
        "malformed": share(sum(c["malformed"] for c in calls), len(calls)),
        "bad_after_bad": share(sum(b["bad"] for a, b in pairs if a["bad"]), sum(a["bad"] for a, _ in pairs)),
        "bad_after_clean": share(sum(b["bad"] for a, b in pairs if not a["bad"] and not a["err"]),
                                 sum(1 for a, _ in pairs if not a["bad"] and not a["err"])),
        "copy_starts": share(sum(c["run_k"] == 2 for c in calls), len(calls)),
        "lock_in": {str(k) + ("+" if k == 3 else ""): share(*lock[k]) for k in (1, 2, 3)},
        "lock_in_n": {str(k) + ("+" if k == 3 else ""): lock[k][1] for k in (1, 2, 3)},
        "runs3": len(long3), "runs3_tasks": len({id(cs) for cs, _, _ in long3}),
        "runs5": sum(e - s + 1 >= 5 for _, s, e in stretches),
        "runs5_tasks": len({id(cs) for cs, s, e in stretches if e - s + 1 >= 5}),
        "loop_tasks": sum(1 for cs in cs_all if cs and sum(c["bad"] for c in cs) >= len(cs) / 2),
        "run3_ends": dict(ends),
        "records": n_rec,
        "prompt_ends_with_block": share(sum(b[0] for b in by_place.values()), n_rec),
        "opens_thought": share(sum(b[1] for b in by_place.values()), n_rec),
        "by_place": {p: {"n": b[2], "prompt_ends_with_block": share(b[0], b[2]), "opens_thought": share(b[1], b[2])}
                     for p, b in sorted(by_place.items(), key=lambda kv: int(kv[0].split("-")[0].rstrip("+")))},
    }


# ---------------------------------------------------------------- arms: passes pooled, pairs, pass to pass

PRODUCTIVE = {"ok_out", "ok", "ok_trunc"}   # gate1.result_class: something came back (a command's output, a read, an edit)
TYPES = ("search", "command", "read", "edit", "submit", "malformed", "other")
RATIOS = {   # metric -> (numerator, denominator), both summed over task-runs
    "resolved": ("resolved", "tasks"),
    "stale": ("stale", "calls"),
    "copies": ("copies", "calls"),
    "returns": ("returns", "calls"),
    "malformed": ("malformed", "calls"),
    "lock-in k=2": ("l2", "l2n"),
    "lock-in k=3+": ("l3", "l3n"),
    "runs 3+ per task": ("runs3", "tasks"),
    "runs 5+ per task": ("runs5", "tasks"),
    "run length (3+)": ("len3", "runs3"),
    "loop tasks": ("loop_task", "tasks"),
    "re-entered": ("reentered", "ended"),
    "next call works": ("productive", "ended"),
    "source edit after": ("edit_after", "ended"),
    "tasks with a source edit": ("source_edit", "tasks"),
}
SHARES = {"resolved", "stale", "copies", "returns", "malformed", "lock-in k=2", "lock-in k=3+", "loop tasks",
          "re-entered", "next call works", "source edit after", "tasks with a source edit"}
PASS_METRICS = ("resolved", "stale", "copies", "returns", "lock-in k=3+", "runs 3+ per task")


def loop_type(c: dict) -> str:
    if c["malformed"]:
        return "malformed"
    if c["fn"] == "run_command":
        return "search" if dose.is_search(str(c["args"].get("command", ""))) else "command"
    return {"read_file": "read", "edit_file": "edit", "write_file": "edit", "submit_patch": "submit"}.get(c["fn"], "other")


def is_source_edit(c: dict) -> bool:
    return c["fn"] in ("edit_file", "write_file") and c["rc"] == "ok" and source_edit(c["fn"], c["args"])


def counts(x: dict) -> Counter:
    """One task-run's counts, the numerators and denominators of RATIOS, plus runs of 3+ by type."""
    cs = x["calls"]
    out = Counter(tasks=1, resolved=int(x["resolved"]), calls=len(cs))
    for i, c in enumerate(cs):
        out["stale"] += c["stale"]
        out["malformed"] += c["malformed"]
        out["copies"] += c["stale"] and c["same_as_prev"]
        out["returns"] += c["stale"] and not c["same_as_prev"]
        if i:
            k = min(cs[i - 1]["run_k"], 3)
            if k >= 2:
                out[f"l{k}n"] += 1
                out[f"l{k}"] += c["same_as_prev"]
    for s, e in runs_of(cs):
        n = e - s + 1
        if n < 3:
            continue
        out["runs3"] += 1
        out["len3"] += n
        out["runs5"] += n >= 5
        out["type:" + loop_type(cs[s])] += 1
        if e < len(cs) - 1:
            later = cs[e + 1:]
            out["ended"] += 1
            out["reentered"] += any(c["stale"] and c["key"] == cs[s]["key"] for c in later)
            out["productive"] += not later[0]["bad"] and later[0]["rc"] in PRODUCTIVE
            out["edit_after"] += any(is_source_edit(c) for c in later)
    out["loop_task"] = int(bool(cs) and sum(c["bad"] for c in cs) >= len(cs) / 2)
    out["source_edit"] = int(any(is_source_edit(c) for c in cs))
    return out


def ratio(cnt: Counter, name: str):
    num, den = RATIOS[name]
    return cnt[num] / cnt[den] if cnt[den] else None


def by_task(runs: list[dict], ids: set) -> dict[str, list[Counter]]:
    """task -> its task-runs' counts, one per run that has the task."""
    out = defaultdict(list)
    for r in runs:
        for t, x in r["tasks"].items():
            if t in ids:
                out[t].append(r["counts"][t])
    return out


def pooled(tasks: dict[str, list[Counter]]) -> Counter:
    return sum((c for cs in tasks.values() for c in cs), Counter())


def paired(a: dict, b: dict, names, reps: int, seed: int) -> dict:
    """a − b for each metric on the tasks both have, with a 95% interval from resampling those tasks. Each draw
    brings all of a task's runs in both arms."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"common": 0}
    sa = [sum(a[t], Counter()) for t in common]
    sb = [sum(b[t], Counter()) for t in common]
    keys = sorted({k for name in names for k in RATIOS[name]})
    va = {k: [c[k] for c in sa] for k in keys}
    vb = {k: [c[k] for c in sb] for k in keys}

    def diff(w):   # w: weights per common task
        out = {}
        for name in names:
            num, den = RATIOS[name]
            da, db = sum(x * y for x, y in zip(w, va[den])), sum(x * y for x, y in zip(w, vb[den]))
            out[name] = (sum(x * y for x, y in zip(w, va[num])) / da - sum(x * y for x, y in zip(w, vb[num])) / db
                         if da and db else None)
        return out
    point = diff([1] * len(common))
    rng, boots = random.Random(seed), defaultdict(list)
    for _ in range(reps):
        w = [0] * len(common)
        for _ in common:
            w[rng.randrange(len(common))] += 1
        for name, v in diff(w).items():
            if v is not None:
                boots[name].append(v)
    out = {"common": len(common), "metrics": {}}
    for name in names:
        bs = sorted(boots[name])
        ci = [bs[int(0.025 * (len(bs) - 1))], bs[int(0.975 * (len(bs) - 1))]] if len(bs) >= 20 else None
        out["metrics"][name] = {"a": ratio(pooled({t: a[t] for t in common}), name),
                                "b": ratio(pooled({t: b[t] for t in common}), name), "diff": point[name], "ci95": ci}
    return out


def fmt_metric(name: str, v) -> str:
    if v is None:
        return "-"
    return pct(v) if name in SHARES else f"{v:.2f}"


def fmt_diff(name: str, d: dict) -> str:
    if d["diff"] is None:
        return "-"
    scale, unit, nd = (100, " pts", 1) if name in SHARES else (1, "", 2)
    ci = f" ({scale * d['ci95'][0]:+.{nd}f} to {scale * d['ci95'][1]:+.{nd}f})" if d["ci95"] else ""
    return f"{scale * d['diff']:+.{nd}f}{unit}{ci}"


def report_arms(arms: dict[str, list[dict]], pairs: list[tuple[str, str]], groups: dict[str, set], all_ids: set,
                reps: int, seed: int) -> dict:
    out = {"groups": {}, "types": {}, "pairs": {}, "pass_to_pass": {}}
    print("## Arms, passes pooled\n")
    print("Each arm pools its runs' task-runs. Copies repeat the call just before; returns repeat an earlier one. "
          "After a run of 3+ identical calls that ends before the task does: is its call re-sent later (re-entry), "
          "does the next call work (not bad, and the tool returned a result: a command's output, a read, an edit), "
          "does a source edit follow?\n")
    for gname, ids in [("all tasks", all_ids)] + list(groups.items()):
        rows = {}
        for name, runs in arms.items():
            tasks = by_task(runs, ids)
            if tasks:
                rows[name] = (len(runs), sum(len(v) for v in tasks.values()), pooled(tasks))
        if not rows:
            continue
        print(f"### {gname}\n")
        print("| arm | runs | task-runs | resolved | stale: copies / returns | malformed | lock-in k=2 / 3+ | runs 3+ / 5+ "
              "per task | run length (3+) | loop tasks | after a run of 3+: re-entered / next works / "
              "source edit after (runs) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        out["groups"][gname] = {}
        for name, (nr, ntr, c) in rows.items():
            m = {k: ratio(c, k) for k in RATIOS}
            out["groups"][gname][name] = {"runs": nr, "task_runs": ntr, **m}
            f = lambda k: fmt_metric(k, m[k])
            print(f"| {name} | {nr} | {ntr} | {f('resolved')} | {f('stale')}: {f('copies')} / {f('returns')} | "
                  f"{f('malformed')} | {f('lock-in k=2')} / {f('lock-in k=3+')} | {f('runs 3+ per task')} / "
                  f"{f('runs 5+ per task')} | {f('run length (3+)')} | {f('loop tasks')} | {f('re-entered')} / "
                  f"{f('next call works')} / {f('source edit after')} ({c['ended']}) |")
        print()
    print("### Runs of 3+ identical calls by type, all tasks\n")
    print("| arm | " + " | ".join(TYPES) + " |\n|---|" + "---|" * len(TYPES))
    for name, runs in arms.items():
        c = pooled(by_task(runs, all_ids))
        out["types"][name] = {t: c["type:" + t] for t in TYPES}
        print(f"| {name} | " + " | ".join(str(c["type:" + t]) for t in TYPES) + " |")
    print("\nsearch: a read-only grep or find (dose.py's rule); command: any other run_command; submit: submit_patch, "
          "the termination loop; malformed: the call's arguments don't fit the tool, whatever the tool.\n")
    if pairs:
        print("## Paired differences, all tasks\n")
        print(f"A − B on the tasks both arms ran; 95% intervals from {reps} resamples of tasks (seed {seed}). Shares "
              "differ in percentage points.\n")
    for a, b in pairs:
        if a not in arms or b not in arms:
            print(f"- {a} − {b}: {'no runs of ' + a if a not in arms else 'no runs of ' + b}\n")
            continue
        res = paired(by_task(arms[a], all_ids), by_task(arms[b], all_ids), list(RATIOS), reps, seed)
        out["pairs"][f"{a} - {b}"] = res
        if not res["common"]:
            print(f"- {a} − {b}: no common tasks\n")
            continue
        print(f"### {a} − {b} ({res['common']} tasks)\n")
        print(f"| metric | {a} | {b} | difference (95% interval) |\n|---|---|---|---|")
        for name, d in res["metrics"].items():
            print(f"| {name} | {fmt_metric(name, d['a'])} | {fmt_metric(name, d['b'])} | {fmt_diff(name, d)} |")
        print()
    multi = {n: r for n, r in arms.items() if len(r) >= 2}
    if multi:
        print("## Pass to pass: each arm's run 2 − run 1, all tasks\n")
        print("How much one pass of an arm varies. A difference between arms that is no larger than this is no "
              "finding.\n")
        print("| arm | " + " | ".join(PASS_METRICS) + " |\n|---|" + "---|" * len(PASS_METRICS))
        for name, runs in multi.items():
            res = paired(by_task(runs[1:2], all_ids), by_task(runs[:1], all_ids), PASS_METRICS, reps, seed)
            out["pass_to_pass"][name] = res
            cells = [fmt_diff(m, res["metrics"][m]) if res["common"] else "-" for m in PASS_METRICS]
            print(f"| {name} ({runs[1]['name']} − {runs[0]['name']}) | " + " | ".join(cells) + " |")
        print()
    return out


def print_provenance(runs: list[dict]) -> None:
    print("## Provenance\n")
    print("| run | wheelhouse | harness | vLLM build | zip sha256 | chat template | sampling | PINS | pace | budget | GPU |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        p = r["prov"]
        samp = p["sampling"] + ("" if p["agreed"] else " (agent dirs disagree!)")
        print(f"| {r['name']} | {p['wheelhouse']} | {p['harness']} | {p['vllm']} | {p['zip']} | {p['template']} | {samp} "
              f"| {p['pins']} | {p['pace']} | {p['budget']} | {p['gpu']} |")
    print("\nSampling is read from the agent dirs the eval jobs extracted (configs/sampling.yaml). Keys the file doesn't "
          "set, such as top_k, come from the model's generation config.\n")


def sign_test(a: int, b: int) -> float:
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="*", default=[])
    ap.add_argument("--group", action="append", default=[], help="NAME=IDS_FILE: report these tasks as one group")
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME=RUN[,RUN...]: an arm, its passes pooled (their runs are reported one by one too)")
    ap.add_argument("--pair", action="append", default=[], help="A:B: arm A against arm B, task by task")
    ap.add_argument("--reps", type=int, default=2000, help="task resamples for the arms' intervals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    groups = {}
    for g in a.group:
        name, _, path = g.partition("=")
        groups[name] = set(Path(path).read_text().split())
    arm_dirs = {}
    for spec in a.arm:
        name, _, dirs = spec.partition("=")
        if not name or not dirs:
            ap.error(f"--arm {spec}: use NAME=RUN[,RUN...]")
        arm_dirs[name] = [Path(x) for x in dirs.split(",") if x]
    pairs = []
    for spec in a.pair:
        x, _, y = spec.partition(":")
        if not x or not y:
            ap.error(f"--pair {spec}: use A:B")
        pairs.append((x, y))
    order = list(dict.fromkeys([*a.runs, *(p for ds in arm_dirs.values() for p in ds)]))
    if not order:
        ap.error("give --runs or --arm")
    loaded = {p: load_run(p) for p in order if p.is_dir()}
    missing = [str(p) for p in order if not p.is_dir()]
    runs = list(loaded.values())
    if not runs:
        print(f"FAIL: no run dirs: {missing}")
        return 1
    for r in runs:
        r["counts"] = {t: counts(x) for t, x in r["tasks"].items()}
    arms = {n: [loaded[p] for p in ds if p in loaded] for n, ds in arm_dirs.items()}
    arms = {n: rs for n, rs in arms.items() if rs}
    if not groups:
        every = {t for r in runs for t in r["tasks"]}
        groups = {"dev repos (fastapi, requests, httpx)": {t for t in every if not t.startswith("rich_")},
                  "rich": {t for t in every if t.startswith("rich_")}}
    all_ids = set().union(*groups.values())
    out = {"runs": {r["name"]: {"template": r["template"], "tasks": len(r["tasks"]), "provenance": r["prov"]}
                    for r in runs}, "groups": {}}
    print("# Loop metrics: " + ", ".join(f"{r['name']} (template {r['template']})" for r in runs) +
          (f"; missing: {', '.join(missing)}" if missing else "") + "\n")
    if arms:
        print("Arms: " + "; ".join(f"{n} = {', '.join(r['name'] for r in rs)}" for n, rs in arms.items()) + "\n")
        print_provenance(runs)
        out["arms"] = report_arms(arms, pairs, groups, all_ids, a.reps, a.seed)
        print("## Each run\n")
    for gname, ids in groups.items():
        per = {r["name"]: metrics({t: x for t, x in r["tasks"].items() if t in ids}) for r in runs}
        if not any(m["tasks"] for m in per.values()):
            continue
        out["groups"][gname] = per
        print(f"{'###' if arms else '##'} {gname}\n")
        print("| run | tasks | resolved | calls / task | stale | malformed | bad after bad / after clean | copies start "
              "| lock-in k=1 / 2 / 3+ | runs 3+ (tasks) | runs 5+ (tasks) | loop tasks |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, m in per.items():
            li = m["lock_in"]
            print(f"| {name} | {m['tasks']} | {m['resolved']} | {m['calls_per_task'] if m['calls_per_task'] is not None else '-'} "
                  f"| {pct(m['stale'])} | {pct(m['malformed'])} | {pct(m['bad_after_bad'])} / {pct(m['bad_after_clean'])} "
                  f"| {pct(m['copy_starts'])} | {pct(li['1'])} / {pct(li['2'])} / {pct(li['3+'])} "
                  f"| {m['runs3']} ({m['runs3_tasks']}) | {m['runs5']} ({m['runs5_tasks']}) | {m['loop_tasks']} |")
        print("\nHow runs of 3+ identical calls end:\n")
        kinds = ("task end", "budget warning", "compaction", "own")
        print("| run | " + " | ".join(kinds) + " |\n|---|" + "---|" * len(kinds))
        for name, m in per.items():
            print(f"| {name} | " + " | ".join(str(m["run3_ends"].get(k, 0)) for k in kinds) + " |")
        print("\nThe main agent's calls by place in the task: prompt ends with the empty thought / completion opens a "
              "thought (n):\n")
        places = sorted({p for m in per.values() for p in m["by_place"]}, key=lambda p: int(p.split("-")[0].rstrip("+")))
        print("| run | all | " + " | ".join(f"call {p}" for p in places) + " |\n|---|---|" + "---|" * len(places))
        for name, m in per.items():
            cells = [f"{pct(m['by_place'][p]['prompt_ends_with_block'])} / {pct(m['by_place'][p]['opens_thought'])} "
                     f"({m['by_place'][p]['n']})" if p in m["by_place"] else "-" for p in places]
            print(f"| {name} | {pct(m['prompt_ends_with_block'])} / {pct(m['opens_thought'])} ({m['records']}) | "
                  + " | ".join(cells) + " |")
        if len(runs) >= 2 and not arms:   # each run against the first, task by task (arms have their own pairs)
            ref = runs[0]
            print(f"\nResolved, task by task, against {ref['name']}:\n")
            for r in runs[1:]:
                common = sorted(set(r["tasks"]) & set(ref["tasks"]) & ids)
                better = sum(r["tasks"][t]["resolved"] and not ref["tasks"][t]["resolved"] for t in common)
                worse = sum(ref["tasks"][t]["resolved"] and not r["tasks"][t]["resolved"] for t in common)
                print(f"- {r['name']}: {better} better, {worse} worse over {len(common)} tasks, sign test "
                      f"p = {sign_test(better, worse):.3f}")
                out["groups"][gname][r["name"]]["vs_first"] = {"common": len(common), "better": better, "worse": worse,
                                                               "p": sign_test(better, worse)}
        print()
    print("Bad = a stale repeat or malformed arguments. Lock-in k: P(the next call is identical to the last one | the "
          "last k calls were identical). A compaction is read off the next call's prompt being shorter.")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
