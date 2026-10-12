"""Loop metrics of agent runs (README.md, experiment 2): how often identical calls start, lock in and end, per run and
task group, and what the served chat template did to the prompts. stdlib only, as baseline/baseline_report.py.

    python loop_report.py --runs <run dir> [<run dir> ...] [--group NAME=IDS_FILE ...] [--json OUT]

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
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gate1"))
import gate1  # noqa: E402  # stdlib at import time: label_calls, main_agent, iter_jsonl

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
    return {"name": run.name, "tasks": tasks, "template": template}


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


def sign_test(a: int, b: int) -> float:
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--group", action="append", default=[], help="NAME=IDS_FILE: report these tasks as one group")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    groups = {}
    for g in a.group:
        name, _, path = g.partition("=")
        groups[name] = set(Path(path).read_text().split())
    runs = [load_run(r) for r in a.runs if r.is_dir()]
    missing = [str(r) for r in a.runs if not r.is_dir()]
    if not runs:
        print(f"FAIL: no run dirs: {missing}")
        return 1
    if not groups:
        every = {t for r in runs for t in r["tasks"]}
        groups = {"dev repos (fastapi, requests, httpx)": {t for t in every if not t.startswith("rich_")},
                  "rich": {t for t in every if t.startswith("rich_")}}
    out = {"runs": {r["name"]: {"template": r["template"], "tasks": len(r["tasks"])} for r in runs}, "groups": {}}
    print("# Loop metrics: " + ", ".join(f"{r['name']} (template {r['template']})" for r in runs) +
          (f"; missing: {', '.join(missing)}" if missing else "") + "\n")
    for gname, ids in groups.items():
        per = {r["name"]: metrics({t: x for t, x in r["tasks"].items() if t in ids}) for r in runs}
        if not any(m["tasks"] for m in per.values()):
            continue
        out["groups"][gname] = per
        print(f"## {gname}\n")
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
        if len(runs) >= 2:   # each run against the first, task by task
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
