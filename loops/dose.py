"""The dose test (README.md section 3, round 2's G1): do added copies of a call raise the chance of one more copy, at a
fixed history? Round 1's lock-in curve compares different histories, so it can come from survivor selection. Here
every arm starts from the same recorded context.

A history is the context just after the first occurrence of an empty read-only search: a grep or find that printed
nothing (exit 1, or exit 0 behind a pipe), the call c and its result r. It is picked by the history alone, whatever
the student did next. Every arm adds blocks after it that carry the identical result r and differ only in the call:
  dose1  the history itself: c once
  dose2  c twice in a row       dose4  c four times in a row
  para2  c, then the same search written differently (quotes, flag order, long options, a dropped trailing slash...)
  para4  c, then three such paraphrases. They return exactly r, so length, position and the number of failures match
         dose's, and only the call's surface form changes.
  var2   c, then a different empty search from the same task (where one exists; its result is the same constant JSON)
Consecutive call blocks sit next to each other in the prompt, as the model's chat template renders repeated calls
(prep checks this on probe conversations).

Subcommands, in this order (loops/dose.slurm runs them, loops/submit_dose.sh queues the jobs):
  prep    CPU. Labels each run's calls with gate1.py's rules, picks the histories, builds the added blocks and checks
          them against the template. Writes states.jsonl, prompts.jsonl, prep.json and prep.md.
  score   GPU (vLLM). The probability of the next call being c, and being the last added call, after every arm's
          context: straight to the call, and after the empty thought the student often writes first (the template
          strips it from the history). Writes score.jsonl.
  sample  GPU (vLLM). --samples continuations per arm at the run's sampling, each judged: c exactly, the same search in
          any form, a stale repeat, malformed, or an escape. Writes samples.jsonl.
  report  CPU, stdlib only. Per arm, and paired between arms with 95% intervals over tasks. Writes dose.md, dose.json.
score and sample resume: a rerun skips what is already written.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shlex
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gate1"))
import gate1 as g  # noqa: E402  # stdlib at import time: labels, alignment, call blocks, sampling probabilities

ARMS = {"dose1": ("c", 0), "dose2": ("c", 1), "dose4": ("c", 3), "para2": ("para", 1), "para4": ("para", 3),
        "var2": ("var", 1)}   # arm -> (what the added blocks call, how many blocks are added)
PATHS = ("direct", "block")   # the next call straight away, or after the empty thought <|channel>thought\n<channel|>
SEARCH = re.compile(r"(^|[;&|(]\s*|\bxargs\s+)(grep|egrep|fgrep|rg|ag|ack|find|git\s+grep)\b")
WRITE = re.compile(r"(?<![0-9&])>(?!&)|\btee\b|\bsed\s+-i|\b(mv|cp|rm|mkdir|touch|chmod|patch|truncate)\b|\bpip\b"
                   r"|\bgit\s+(apply|checkout|commit|stash|reset|am|restore)\b|\bpython3?\b|\bperl\b|\bawk\b")
BOOL = set("rnilwRHsEFIh")   # grep's boolean short flags: safe to reorder, split or spell long
LONG = {"r": "--recursive", "n": "--line-number", "i": "--ignore-case", "l": "--files-with-matches",
        "w": "--word-regexp", "R": "--dereference-recursive", "H": "--with-filename", "s": "--no-messages",
        "E": "--extended-regexp", "F": "--fixed-strings"}
SHORT = {v: k for k, v in LONG.items()}
PROBE_CMD = 'grep -rn "probe_symbol" fastapi/'
PROBE_PARA = "grep -nr 'probe_symbol' fastapi"
PROBE_RES = json.dumps({"status": "error", "error_type": "CommandError", "error_message": "",
                        "details": {"stdout": "", "stderr": "", "exit_code": 1}})


# ---------------------------------------------------------------- commands: searches and their paraphrases

def command(c: dict) -> str:
    return str(c["args"].get("command", "")) if c["fn"] == "run_command" else ""


def empty_result(d) -> bool:
    """Nothing printed and nothing on stderr: exit 1 (grep found nothing) or exit 0 (behind a pipe, or find)."""
    if not isinstance(d, dict) or "budget_warning" in d:
        return False
    det = d.get("details") if d.get("status") == "error" else d
    if not isinstance(det, dict) or (d.get("error_message") or "").strip():
        return False
    return (det.get("stdout") or "") == "" and (det.get("stderr") or "") == "" and det.get("exit_code") in (0, 1)


def is_search(cmd: str) -> bool:
    return bool(SEARCH.search(cmd)) and not WRITE.search(cmd)


def first_segment(cmd: str) -> tuple[int, int]:
    """(start, end) of the first grep or find invocation, up to its pipe, ';' or '&&' (outside quotes)."""
    m = SEARCH.search(cmd)
    if not m:
        return 0, 0
    a, quote = m.start(2), None
    for i in range(a, len(cmd)):
        ch = cmd[i]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "|;" or cmd.startswith("&&", i):
            return a, i
    return a, len(cmd)


def _swap_quotes(seg: str) -> str | None:
    m = re.search(r"\"([^\"]*)\"|'([^']*)'", seg)
    if not m:
        return None
    if m.group(1) is not None:
        s = m.group(1)
        return None if re.search(r"['\\$`!\n]", s) else seg[:m.start()] + f"'{s}'" + seg[m.end():]
    s = m.group(2)
    return None if re.search(r"[\"\\$`!\n]", s) else seg[:m.start()] + f'"{s}"' + seg[m.end():]


def _cluster(seg: str):
    return re.search(r"(?<=\s)-([A-Za-z]{2,})(?=\s)", seg) if seg.startswith("grep") else None


def _reorder(seg: str) -> str | None:
    m = _cluster(seg)
    if not m or not set(m.group(1)) <= BOOL or m.group(1)[::-1] == m.group(1):
        return None
    return seg[:m.start()] + "-" + m.group(1)[::-1] + seg[m.end():]


def _split(seg: str) -> str | None:
    m = _cluster(seg)
    if not m or not set(m.group(1)) <= BOOL:
        return None
    return seg[:m.start()] + " ".join("-" + ch for ch in m.group(1)) + seg[m.end():]


def _long(seg: str) -> str | None:
    if not seg.startswith("grep"):
        return None
    m = re.search(r"(?<=\s)-([A-Za-z]+)(?=\s)", seg)
    if not m or not all(ch in LONG for ch in m.group(1)):
        return None
    return seg[:m.start()] + " ".join(LONG[ch] for ch in m.group(1)) + seg[m.end():]


def _slash(seg: str) -> str | None:
    """A directory written with a trailing slash loses it, or '.' becomes './' (outside quotes). A grep whose pattern
    isn't quoted is left alone: the pattern itself could end with the slash."""
    if seg.startswith("grep") and not re.search(r"\s(\"[^\"]*\"|'[^']*')", seg):
        return None
    parts = re.split(r"(\"[^\"]*\"|'[^']*')", seg)
    for k in range(0, len(parts), 2):
        m = re.search(r"(?<=\s)([\w.][\w./-]*?)/(?=\s|$)", parts[k])
        if m:
            parts[k] = parts[k][:m.end(1)] + parts[k][m.end(1) + 1:]
            return "".join(parts)
    for k in range(0, len(parts), 2):
        m = re.search(r"(?<=\s)\.(?=\s|$)", parts[k])
        if m:
            parts[k] = parts[k][:m.end()] + "/" + parts[k][m.end():]
            return "".join(parts)
    return None


def _dash_e(seg: str) -> str | None:
    if not seg.startswith("grep") or re.search(r"\s-e\s", seg):
        return None
    m = re.search(r"(?<=\s)(\"[^\"]*\"|'[^']*')", seg)
    if not m or m.group(1)[1:2] == "-":
        return None
    return seg[:m.start()] + "-e " + seg[m.start():]


RULES = (("quotes", _swap_quotes), ("flag order", _reorder), ("long flags", _long), ("split flags", _split),
         ("slash", _slash), ("-e", _dash_e))


def _head(cmd: str) -> str | None:
    m = re.search(r"\bhead -n (\d+)\b", cmd) or re.search(r"\bhead -(\d+)\b", cmd)
    if not m:
        return None
    new = f"head -{m.group(1)}" if m.group(0).startswith("head -n") else f"head -n {m.group(1)}"
    return cmd[:m.start()] + new + cmd[m.end():]


def paraphrases(cmd: str, taken: set[str], n: int = 3) -> list[tuple[str, str]]:
    """Up to n commands that run the same search as cmd and print the same nothing: (command, rules applied). Each
    rule keeps grep's or find's output and exit code, and none is in taken (the commands already in the context)."""
    a, b = first_segment(cmd)
    if a == b:
        return []
    seg = cmd[a:b]
    trailing = seg[len(seg.rstrip()):]
    seg = seg.rstrip()

    def whole(s: str) -> str:
        return cmd[:a] + s + trailing + cmd[b:]
    out, seen = [], {cmd} | taken
    singles = [(name, f(seg)) for name, f in RULES]
    singles = [(name, s) for name, s in singles if s and s != seg]
    cands = [(name, whole(s)) for name, s in singles]
    h = _head(cmd)
    if h:
        cands.append(("head", h))
    for i, (n1, s1) in enumerate(singles):   # then two rules at once
        for n2, f2 in RULES:
            s2 = f2(s1)
            if s2 and s2 != s1 and n2 != n1:
                cands.append((f"{n1} + {n2}", whole(s2)))
    for name, c in cands:
        if c not in seen and norm(c) == norm(cmd):
            out.append((c, name))
            seen.add(c)
        if len(out) == n:
            break
    return out


def norm(cmd: str):
    """A command's search, ignoring how it is written: per pipeline segment, the program, its single-letter flags as
    a set, and its other arguments in order (a trailing slash dropped, './' as '.', 'head -N' as 'head -n N')."""
    try:
        toks = shlex.split(cmd.strip())
    except ValueError:
        return ("raw", cmd.strip())
    segs, cur = [], []
    for t in toks + ["|"]:
        if t in ("|", ";", "&&", "||"):
            if cur:
                prog, flags, rest = cur[0], set(), []
                skip_e = False
                for x in cur[1:]:
                    if x == "-e":
                        skip_e = True
                        continue
                    if x in SHORT:
                        flags.add(SHORT[x])
                    elif re.fullmatch(r"-[A-Za-z]+", x) and (prog != "find" or len(x) == 2) and set(x[1:]) <= BOOL:
                        flags |= set(x[1:])
                    elif prog == "head" and re.fullmatch(r"-\d+", x):   # head -N is head -n N
                        flags.add("n")
                        rest.append(x[1:])
                    elif x == "./":
                        rest.append(".")
                    elif len(x) > 1 and x.endswith("/") and not skip_e:
                        rest.append(x.rstrip("/"))
                    else:
                        rest.append(x)
                    skip_e = False
                segs.append((prog, tuple(sorted(flags)), tuple(rest)))
            cur = []
        else:
            cur.append(t)
    return tuple(segs)


def target(cmd: str):
    """The search's pattern and places, for telling a different search from a paraphrase (heuristic)."""
    segs = norm(cmd)
    if not segs or segs[0] == "raw":
        return segs
    prog, _, rest = segs[0]
    return prog, tuple(x for x in rest if not x.startswith("-"))


# ---------------------------------------------------------------- blocks

def call_ids(v: g.Vocab, cmd: str) -> list[int]:
    """A run_command call as the template renders it: <|tool_call>call:run_command{command:<|"|>...<|"|>}<tool_call|>."""
    return v.enc(f"<|tool_call>call:run_command{{command:{g.Q}{cmd}{g.Q}}}<tool_call|>")


def context(pr: dict, arm: str) -> tuple[list[int], list[int]] | None:
    """(the arm's context, the last added call's tokens), or None where the arm doesn't apply."""
    kind, n = ARMS[arm]
    pool = [pr["call_ids"]] * n if kind == "c" else pr[f"{kind}_ids"][:n]
    if len(pool) < n:
        return None
    return pr["prompt_ids"] + [t for ids in pool for t in ids + pr["resp_ids"]], (pool[-1] if pool else pr["call_ids"])


def probe_checks(tok, v: g.Vocab) -> list[tuple[str, bool, str]]:
    """The added blocks against the model's own template: k identical calls render as the first one plus k-1 copies of
    its block, and a paraphrase renders as its call tokens followed by the same response."""
    def conv(cmds):
        msgs = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix the bug in a.py."}]
        for k, c in enumerate(cmds):
            msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{k}", "type": "function",
                      "function": {"name": "run_command", "arguments": {"command": c}}}]},
                     {"role": "tool", "tool_call_id": f"c{k}", "name": "run_command", "content": PROBE_RES}]
        text = tok.apply_chat_template(msgs, tools=g.PROBE_TOOLS, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        return v.enc(text)
    one = conv([PROBE_CMD])
    bl = g.call_blocks(v, one)
    out = [("thinking off leaves the turn open after a tool response", bool(one) and one[-1] == v.resp_close,
            repr(v.dec(one[-12:])))]
    if not bl or bl[-1][1] != len(one):
        return out + [("the probe ends with its call block", False, repr(v.dec(one[-40:])))]
    s, e, call = bl[-1]
    block = one[s:e]
    out.append(("the call's tokens are the template's", call_ids(v, PROBE_CMD) == list(call), v.dec(call)))
    for k in (2, 4):
        out.append((f"{k} identical calls render as the first one plus {k - 1} copies of its block",
                    conv([PROBE_CMD] * k) == one + block * (k - 1), ""))
    para = conv([PROBE_CMD, PROBE_PARA])
    out.append(("a paraphrase renders as its call tokens and the same response",
                para == one + call_ids(v, PROBE_PARA) + one[s + len(call):e], ""))
    d = g.dedup(v, one + block * 3)
    out.append(("dedup of 4 identical calls gives back the history", d is not None and d[0] == one and d[1] == 3, ""))
    rules = paraphrases(PROBE_CMD, set())
    out.append(("the probe command has 3 paraphrases", len(rules) == 3, "; ".join(f"{c!r} ({r})" for c, r in rules)))
    return out


# ---------------------------------------------------------------- prep

def cmd_prep(a) -> int:
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    v = g.Vocab(tok)
    a.work.mkdir(parents=True, exist_ok=True)
    checks = probe_checks(tok, v)
    counts, cands = Counter(), []
    for run in a.runs:
        last = {}   # task -> (shard, result): a later rerun of a task wins, as loop_report.py reads a run
        for shard in sorted(p for p in run.glob("shard_*") if p.is_dir()):
            for r in g.iter_jsonl(shard / "task_results.jsonl"):
                if r.get("error") != "SKIPPED_DEADLINE":
                    last[r["instance_id"]] = (shard, r)
        by_shard = defaultdict(list)
        for tid, (shard, r) in last.items():
            by_shard[shard].append((tid, r))
        for shard in sorted(by_shard):
            recs = g.load_records(shard)
            for tid, res in sorted(by_shard[shard], key=lambda x: x[0]):
                f = shard / "traces" / f"trace_{tid}.json"
                if not f.is_file():
                    counts["tasks without a trace (left out)"] += 1
                    continue
                counts["tasks read"] += 1
                trace = json.loads(f.read_text(encoding="utf-8"))
                agent = g.main_agent(trace)
                cs = g.label_calls(trace, agent)
                steps = [s for s in g.agent_steps(trace, agent) if s.get("metrics")]
                matched = dict(zip(map(id, steps), g.align(steps, recs.get(tid, []))))
                for i, c in enumerate(cs):
                    if not (empty_result(c["obs"]) and is_search(command(c))):
                        continue
                    counts["empty read-only searches"] += 1
                    if c["prev_same"] is not None:
                        counts["  repeats of an earlier call (left out)"] += 1
                        continue
                    if norm(command(c)) in {norm(command(y)) for y in cs[:i] if y["fn"] == "run_command"}:
                        counts["  the same search written differently earlier (left out)"] += 1
                        continue
                    if i + 1 >= len(cs):
                        counts["  the task's last call (left out)"] += 1
                        continue
                    rec = matched.get(id(cs[i + 1]["step"]))
                    if rec is None:
                        counts["  next call without a calls.jsonl record (left out)"] += 1
                        continue
                    counts["  candidates"] += 1
                    cands.append({"run": run.name, "shard": shard, "task": tid, "i": i, "cs": cs, "line": rec["line"],
                                  "resolved": bool(res.get("resolved"))})
    rng = random.Random(a.seed)
    by_task = defaultdict(list)
    for x in cands:
        by_task[x["task"]].append(x)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    for t in tasks:
        rng.shuffle(by_task[t])
    order = [by_task[t][k] for k in range(a.per_task) for t in tasks if len(by_task[t]) > k]   # round-robin
    wanted = defaultdict(dict)
    for x in order:
        x["key"] = f"{x['run']}/{x['shard'].name}/{x['task']}/{x['i']}"
        wanted[x["shard"]][x["line"]] = x["key"]
    prompts = {}
    for shard, lines in wanted.items():
        prompts.update(g.fetch_prompts(shard, lines))
    states, rows, built = [], [], Counter()
    for x in order:
        if len(states) >= a.target:
            break
        cs, i = x["cs"], x["i"]
        c, nxt = cs[i], cs[i + 1]
        pr = prompts.get(x["key"])
        p = pr["prompt_ids"] if pr else []
        bl = g.call_blocks(v, p)
        if not bl or bl[-1][1] != len(p) or p[-1] != v.resp_close:
            built["the next prompt doesn't end with the call's block, e.g. after a compaction (left out)"] += 1
            continue
        s, e, call = bl[-1]
        parsed = g.parse_call(v.dec(call))
        if not parsed or g.call_key(*parsed) != c["key"]:
            built["the prompt's last block is another call (left out)"] += 1
            continue
        cmd = command(c)
        if call_ids(v, cmd) != list(call):
            built["the call doesn't re-encode to its own tokens (left out)"] += 1
            continue
        resp = p[s + len(call):e]
        content = (c["step"].get("observation") or {}).get("content")
        taken = {command(y) for y in cs[:i + 1]}
        paras = paraphrases(cmd, taken)
        para_ids = [call_ids(v, q) for q, _ in paras]
        if any(g.parse_call(v.dec(ids)) != ("run_command", {"command": q}) for ids, (q, _) in zip(para_ids, paras)):
            built["a paraphrase doesn't parse back (left out)"] += 1
            continue
        seen_t = {target(command(y)) for y in cs[:i + 1]}
        variants = []
        for y in cs[i + 1:]:   # its result is the same constant JSON, recorded later in the task
            q = command(y)
            if (y["prev_same"] is None and is_search(q) and empty_result(y["obs"]) and q not in taken
                    and (y["step"].get("observation") or {}).get("content") == content and target(q) not in seen_t
                    and g.parse_call(v.dec(call_ids(v, q))) == ("run_command", {"command": q})):
                variants.append({"command": q, "i": y["i"]})
                seen_t.add(target(q))
        variants = variants[:3]
        var_ids = [call_ids(v, w["command"]) for w in variants]
        long_block = max([len(call)] + [len(ids) for ids in para_ids + var_ids])
        if len(p) + 3 * (long_block + len(resp)) + a.call_tokens > a.max_len:
            built["too long for --max-len with 3 added blocks (left out)"] += 1
            continue
        d = g.dedup(v, p + p[s:e])
        if d is None or d != (p, 1):
            built["dedup doesn't read an added copy as a copy (left out)"] += 1
            continue
        built["picked"] += 1
        built["with para4"] += len(paras) >= 3
        built["with var2"] += len(variants) >= 1
        states.append({
            "key": x["key"], "run": x["run"], "shard": x["shard"].name, "task": x["task"], "resolved": x["resolved"],
            "i": c["i"], "n_calls": len(cs), "command": cmd, "call_key": c["key"], "rc": c["rc"],
            "paraphrases": [{"command": q, "rules": r} for q, r in paras], "variants": variants,
            "stale_keys": nxt["stale_keys"], "next_cls": nxt["cls"], "next_copies": nxt["same_as_prev"],
            "n_prompt": len(p), "call_len": len(call), "resp_len": len(resp),
            "para_lens": [len(ids) for ids in para_ids], "var_lens": [len(ids) for ids in var_ids]})
        rows.append({"key": x["key"], "prompt_ids": p, "call_ids": list(call), "resp_ids": resp, "para_ids": para_ids,
                     "var_ids": var_ids})
    g.write_jsonl(a.work / "states.jsonl", states)
    g.write_jsonl(a.work / "prompts.jsonl", rows)
    ok = all(ch[1] for ch in checks)
    summary = {"runs": [str(r) for r in a.runs], "counts": dict(counts), "built": dict(built),
               "tasks": len({s["task"] for s in states}), "candidate_tasks": len(by_task),
               "checks": [{"check": n, "pass": p, "detail": d} for n, p, d in checks], "secs": round(time.time() - t0, 1)}
    (a.work / "prep.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [f"# Dose test prep: {a.work.name}", "", f"Runs: {', '.join(r.name for r in a.runs)}. {summary['secs']} s.",
             "", "## Template checks", ""]
    lines += [f"- {'PASS' if p else 'FAIL'} {n}{': ' + d if d else ''}" for n, p, d in checks]
    lines += ["", "## Histories", ""] + [f"- {k}: {n}" for k, n in counts.items()]
    lines += [f"- tasks with a candidate: {len(by_task)}; picked round-robin, at most {a.per_task} per task, "
              f"{a.target} at most:"] + [f"  - {k}: {n}" for k, n in built.items()]
    lines += [f"- picked: {len(states)} histories in {summary['tasks']} tasks; result "
              + ", ".join(f"{k} {n}" for k, n in Counter(s["rc"] for s in states).most_common())
              + "; the student's next call: "
              + ", ".join(f"{k} {n}" for k, n in Counter(s["next_cls"] for s in states).most_common()), "",
              "## Examples", ""]
    for s in random.Random(a.seed).sample(states, min(3, len(states))):
        lines += [f"- {s['task']}, call {s['i']} of {s['n_calls']} ({s['rc']}): `{g.trim(s['command'], 140)}`"]
        lines += [f"  - para: `{g.trim(q['command'], 140)}` ({q['rules']})" for q in s["paraphrases"]]
        lines += [f"  - var: `{g.trim(w['command'], 140)}` (call {w['i']})" for w in s["variants"][:1]]
    (a.work / "prep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not ok:
        print("FAIL: a template check failed; the GPU steps would score contexts the template never renders")
        return 1
    if not states:
        print("FAIL: no histories picked")
        return 1
    return 0


# ---------------------------------------------------------------- score and sample (vLLM)

def load_work(work: Path):
    states = list(g.iter_jsonl(work / "states.jsonl"))
    prompts = {r["key"]: r for r in g.iter_jsonl(work / "prompts.jsonl")}
    return states, prompts


def vocab(a) -> g.Vocab:
    from transformers import AutoTokenizer
    return g.Vocab(AutoTokenizer.from_pretrained(str(a.model), local_files_only=True))


def cmd_score(a) -> int:
    """Prompt logprobs of the next call: c, and the last added call where that is another one, on both paths."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    v = vocab(a)
    states, prompts = load_work(a.work)
    path = a.work / "score.jsonl"
    done = {r["key"] for r in g.iter_jsonl(path)}
    jobs, skipped = [], Counter()
    for s in states:
        pr = prompts[s["key"]]
        for arm in a.arms:
            built = context(pr, arm)
            if built is None:
                skipped[f"{arm} doesn't apply"] += 1
                continue
            ctx, last = built
            targets = [("c", pr["call_ids"])] + ([("last", last)] if ARMS[arm][0] != "c" else [])
            for name, ids in targets:
                for way in PATHS:
                    key = f"{s['key']}|{arm}|{name}|{way}"
                    if key in done:
                        continue
                    full = ctx + (v.empty_thought if way == "block" else []) + ids
                    if len(full) + 1 > a.max_len:
                        skipped["longer than --max-len"] += 1
                        continue
                    jobs.append((key, s, arm, name, way, full, len(ctx)))
    print(f"score: {len(jobs)} to do, {len(done)} done, skipped {dict(skipped)}", flush=True)
    if not jobs:
        return 0
    llm = g.make_llm(a, prefix_caching=False)   # as gate1's score: prompt logprobs need every position computed
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=a.topk, detokenize=False)
    t0, n_tok = time.time(), 0
    for k in range(0, len(jobs), a.batch):
        part = jobs[k:k + a.batch]
        outs = llm.generate([TokensPrompt(prompt_token_ids=j[5]) for j in part], sp, use_tqdm=False)
        rows = []
        for (key, s, arm, name, way, full, start), o in zip(part, outs):
            n_tok += len(full)
            lp = {"raw": 0.0, "run": 0.0, "t1": 0.0}
            for t in range(start, len(full)):
                d = o.prompt_logprobs[t]
                top = sorted(((i, e) for i, e in d.items() if e.rank is not None and e.rank <= a.topk),
                             key=lambda ie: ie[1].rank)[:a.topk]
                tids, tlps = [i for i, _ in top], [e.logprob for _, e in top]
                lp["raw"] += d[full[t]].logprob
                lp["run"] += g.sampled_logprob(tids, tlps, full[t], a.temperature, a.top_p, a.top_k)
                lp["t1"] += g.sampled_logprob(tids, tlps, full[t], 1.0, a.top_p, a.top_k)
            rows.append({"key": key, "state": s["key"], "arm": arm, "target": name, "path": way,
                         "n_ctx": start, "n": len(full) - start,
                         **{f"logp_{m}": round(x, 4) if math.isfinite(x) else None for m, x in lp.items()}})
        g.write_jsonl(path, rows, "a")
        el = time.time() - t0
        print(f"  {k + len(part)}/{len(jobs)}, {el:.0f} s, {n_tok / max(el, 1e-9):.0f} prompt tok/s", flush=True)
    return 0


def judge(v: g.Vocab, s: dict, arm: str, ids: list[int]) -> dict:
    """One sampled continuation after an arm's context."""
    opened = ids[:len(v.empty_thought)] == v.empty_thought
    parsed = g.parse_call(v.dec(ids))
    kind, n = ARMS[arm]
    added = [s["command"]] * n if kind == "c" else [q["command"] for q in (
        s["paraphrases"] if kind == "para" else s["variants"])[:n]]
    stale = set(s["stale_keys"]) | {g.call_key("run_command", {"command": q}) for q in added}
    key = g.call_key(*parsed) if parsed else None
    cmd = str(parsed[1].get("command", "")) if parsed and parsed[0] == "run_command" else None
    last = added[-1] if added else s["command"]
    bad = parsed is not None and g.malformed(*parsed)
    return {"has_call": parsed is not None, "opened_block": opened, "exact_c": key == s["call_key"],
            "exact_last": cmd == last, "same_search": cmd is not None and norm(cmd) == norm(s["command"]),
            "stale": key in stale, "malformed": bad, "escape": parsed is not None and key not in stale and not bad,
            "call_text": g.trim(v.dec(ids), 300)}


def cmd_sample(a) -> int:
    """--samples continuations per arm, one request each. Every arm of a history uses the history's seed, so the arms
    draw the same random numbers (as gate1's own does for block and C-none)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    v = vocab(a)
    states, prompts = load_work(a.work)
    path = a.work / "samples.jsonl"
    done = {r["key"] for r in g.iter_jsonl(path)}
    todo, skipped = [], Counter()
    for s in sorted(states, key=lambda s: s["key"]):
        for arm in a.arms:   # the arms of a history share its prefix, for vLLM's prefix cache
            key = f"{s['key']}|{arm}"
            if key in done:
                continue
            built = context(prompts[s["key"]], arm)
            if built is None:
                skipped[f"{arm} doesn't apply"] += 1
                continue
            if len(built[0]) + a.call_tokens > a.max_len:
                skipped["longer than --max-len"] += 1
                continue
            todo.append((key, s, arm, built[0]))
    print(f"sample: {len(todo)} to do, {len(done)} done, skipped {dict(skipped)}; {a.samples} each, "
          f"T={a.temperature}, top_p {a.top_p}, top_k {a.top_k}", flush=True)
    if not todo:
        return 0
    llm = g.make_llm(a, prefix_caching=True)
    stop = [v.call_close]
    t0 = time.time()
    for k in range(0, len(todo), a.chunk):
        part = todo[k:k + a.chunk]
        sps = [SamplingParams(n=a.samples, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                              max_tokens=a.call_tokens, stop_token_ids=stop, seed=g.seed_of(s["key"]),
                              detokenize=False) for _, s, _, _ in part]
        outs = llm.generate([TokensPrompt(prompt_token_ids=ctx) for *_, ctx in part], sps, use_tqdm=False)
        rows = []
        for (key, s, arm, ctx), o in zip(part, outs):
            rows.append({"key": key, "state": s["key"], "arm": arm, "n_ctx": len(ctx),
                         "samples": [dict(judge(v, s, arm, g.ended_with(x, stop)), finish=x.finish_reason)
                                     for x in o.outputs]})
        g.write_jsonl(path, rows, "a")
        print(f"  {k + len(part)}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------- report (stdlib)

def paired(diffs: dict[str, list[float]], reps: int, seed: int):
    """Mean of paired differences over histories, and its 95% interval from resampling tasks with replacement.
    diffs: task -> the differences of its histories."""
    tasks = sorted(diffs)
    flat = [x for t in tasks for x in diffs[t]]
    if not flat:
        return None, None, None
    rng, boot = random.Random(seed), []
    for _ in range(reps):
        pick = [diffs[rng.choice(tasks)] for _ in tasks]
        n = sum(len(d) for d in pick)
        boot.append(sum(sum(d) for d in pick) / n)
    boot.sort()
    return sum(flat) / len(flat), boot[int(0.025 * (reps - 1))], boot[int(0.975 * (reps - 1))]


SAMPLE_FIELDS = ("exact_c", "same_search", "stale", "malformed", "escape", "has_call", "opened_block")


def cmd_report(a) -> int:
    states = {s["key"]: s for s in g.iter_jsonl(a.work / "states.jsonl")}
    sc = defaultdict(dict)   # (state, arm) -> {(target, path): row}
    for r in g.iter_jsonl(a.work / "score.jsonl"):
        sc[r["state"], r["arm"]][r["target"], r["path"]] = r
    smp = {(r["state"], r["arm"]): r["samples"] for r in g.iter_jsonl(a.work / "samples.jsonl")}

    def p(st, arm, tgt="c", m="run"):
        d = sc.get((st, arm), {})
        rows = [d.get((tgt, w)) for w in PATHS]
        if any(r is None for r in rows):
            return None
        return sum(math.exp(r[f"logp_{m}"]) if r[f"logp_{m}"] is not None else 0.0 for r in rows)

    def share(st, arm, f):
        xs = smp.get((st, arm))
        return sum(x[f] for x in xs) / len(xs) if xs else None
    metrics = {"p(c), run's sampling": lambda st, arm: p(st, arm),
               "p(c), T=1": lambda st, arm: p(st, arm, m="t1"),
               "p(last added call)": lambda st, arm: p(st, arm, "c" if ARMS[arm][0] == "c" else "last"),
               **{f"samples: {f}": (lambda f: lambda st, arm: share(st, arm, f))(f) for f in SAMPLE_FIELDS}}
    arms = [arm for arm in ARMS if any(k[1] == arm for k in list(sc) + list(smp))]
    out = {"states": len(states), "tasks": len({s["task"] for s in states.values()}), "arms": {}, "paired": {}}
    meta = json.loads((a.work / "prep.json").read_text(encoding="utf-8")) if (a.work / "prep.json").is_file() else {}
    n_smp = max((len(x) for x in smp.values()), default=0)
    lines = [f"# The dose test: {a.work.name}", "",
             f"{len(states)} histories in {out['tasks']} tasks, from {', '.join(Path(r).name for r in meta.get('runs', []))}. "
             "Each history ends just after the first occurrence of an empty read-only search, c (README.md section 3, "
             "G1). Every arm adds blocks with the identical result, and they differ only in the call: dose repeats c, "
             "para writes the same search differently, and var is a different search from the same task.", "",
             "## By arm: the next call's probability", "",
             "p(c) adds the two ways the next call can start: straight away, or after the empty thought the student "
             "writes itself. It is rebuilt from the top-20 logprobs at the run's sampling (T=0.2, top_p 0.95, top_k "
             "64), and at T=1 with the same top_p and top_k.", "",
             "| arm | histories | added tokens | p(c), run's sampling | p(c), T=1 | p(last added call) |",
             "|---|---|---|---|---|---|"]
    for arm in arms:
        keys = [k for k in states if context_applies(states[k], arm)]
        vals = {name: [f(k, arm) for k in keys] for name, f in metrics.items()}
        vals = {name: [x for x in xs if x is not None] for name, xs in vals.items()}
        added = [added_tokens(states[k], arm) for k in keys]
        row = {name: (sum(xs) / len(xs) if xs else None) for name, xs in vals.items()}
        row.update(histories=len(keys), added_tokens=sum(added) / len(added) if added else 0)
        out["arms"][arm] = row
        lines.append(f"| {arm} | {len(keys)} | {row['added_tokens']:.0f} | " + " | ".join(
            g.fmt(row[name], 3) for name in ("p(c), run's sampling", "p(c), T=1", "p(last added call)")) + " |")
    lines += ["", "## By arm: sampled continuations", "",
              f"{n_smp} per arm and history at the run's sampling, with the history's seed in every arm. Shares of "
              "samples: c exactly; the same search in any form; a stale repeat in that context; malformed; an escape "
              "(a call that is neither); no call; opened with the empty thought.", "",
              "| arm | exact c | same search | stale | malformed | escape | no call | empty thought first |",
              "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        r = out["arms"][arm]
        nc = None if r["samples: has_call"] is None else 1 - r["samples: has_call"]
        lines.append(f"| {arm} | " + " | ".join(g.fmt(x, 3) for x in (
            r["samples: exact_c"], r["samples: same_search"], r["samples: stale"], r["samples: malformed"],
            r["samples: escape"], nc, r["samples: opened_block"])) + " |")
    pairs = [("dose2", "dose1"), ("dose4", "dose1"), ("dose4", "dose2"), ("para2", "dose1"), ("para4", "dose1"),
             ("dose2", "para2"), ("dose4", "para4"), ("var2", "dose1"), ("dose2", "var2")]
    lines += ["", "## Paired, on the histories where both arms apply", "",
              f"Mean difference over histories; 95% interval from {a.reps} resamples of tasks (seed {a.seed}).", "",
              "| arms | histories | tasks | " + " | ".join(("p(c), run's sampling", "samples: exact_c",
                                                          "samples: same_search", "samples: escape")) + " |",
              "|---|---|---|---|---|---|---|"]
    for x, y in pairs:
        if x not in arms or y not in arms:
            continue
        cells, rec = [], {}
        keys = [k for k in states if context_applies(states[k], x) and context_applies(states[k], y)]
        for name in ("p(c), run's sampling", "samples: exact_c", "samples: same_search", "samples: escape"):
            f = metrics[name]
            diffs = defaultdict(list)
            for k in keys:
                fx, fy = f(k, x), f(k, y)
                if fx is not None and fy is not None:
                    diffs[states[k]["task"]].append(fx - fy)
            point, lo, hi = paired(diffs, a.reps, a.seed)
            rec[name] = {"diff": point, "ci": [lo, hi], "n": sum(len(d) for d in diffs.values())}
            cells.append("-" if point is None else f"{point:+.3f} ({lo:+.3f} to {hi:+.3f})")
        out["paired"][f"{x}-{y}"] = rec
        lines.append(f"| {x} − {y} | {len(keys)} | {len({states[k]['task'] for k in keys})} | " + " | ".join(cells) + " |")
    lines += ["", "## By the search's result and by what the student did next (p(c), run's sampling)", "",
              "| group | histories | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    groups = [(f"result {rc}", lambda s, rc=rc: s["rc"] == rc) for rc in sorted({s["rc"] for s in states.values()})]
    groups += [("student copied next", lambda s: s["next_copies"]), ("student did not", lambda s: not s["next_copies"])]
    for name, keep in groups:
        keys = [k for k, s in states.items() if keep(s)]
        cells = []
        for arm in arms:
            xs = [p(k, arm) for k in keys if context_applies(states[k], arm)]
            xs = [x for x in xs if x is not None]
            cells.append(g.fmt(sum(xs) / len(xs), 3) if xs else "-")
        lines.append(f"| {name} | {len(keys)} | " + " | ".join(cells) + " |")
    lines += ["", "Reading it: a dose effect (dose2 and dose4 above dose1) that para and var don't match says copies "
              "of the identical call raise the next copy at a fixed history. If para matches dose, the same search "
              "repeated in any form does it; if neither moves p(c), the lock-in curve's rise is selection. "
              "same_search judges a sampled command by its search, ignoring quotes, flag order and spelling "
              "(a heuristic). An escape is a call that is neither a stale repeat in that context nor malformed.", ""]
    (a.work / "dose.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (a.work / "dose.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def context_applies(s: dict, arm: str) -> bool:
    kind, n = ARMS[arm]
    return kind == "c" or len(s["paraphrases" if kind == "para" else "variants"]) >= n


def added_tokens(s: dict, arm: str) -> int:
    """Tokens the arm adds after the history."""
    kind, n = ARMS[arm]
    lens = [s["call_len"]] * n if kind == "c" else s[f"{kind}_lens"][:n]
    return sum(x + s["resp_len"] for x in lens)


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="run dirs: shard_*/{calls.jsonl,traces/,task_results.jsonl}")
    p.add_argument("--target", type=int, default=60, help="histories at most")
    p.add_argument("--per-task", type=int, default=2, help="histories per task at most")
    p.add_argument("--seed", type=int, default=0)
    for name, chunk in (("score", 2048), ("sample", 8192)):
        p = sub.add_parser(name)
        p.add_argument("--arms", type=lambda s: [x for x in s.split(",") if x], default=list(ARMS))
        p.add_argument("--gpu-mem", type=float, default=0.85)
        p.add_argument("--chunk-tokens", type=int, default=chunk,
                       help="vLLM's max_num_batched_tokens: score keeps it small, as gate1's score does")
        p.add_argument("--max-seqs", type=int, default=32)
        p.add_argument("--topk", type=int, default=20, help="top logprobs per position (vLLM's max_logprobs)")
    sub.choices["score"].add_argument("--batch", type=int, default=8, help="prompts per vLLM call (prompt logprobs are big)")
    sub.choices["sample"].add_argument("--samples", type=int, default=3, help="continuations per arm and history")
    sub.choices["sample"].add_argument("--chunk", type=int, default=64, help="requests per batch written")
    p = sub.add_parser("report")
    p.add_argument("--reps", type=int, default=10000, help="bootstrap resamples of tasks")
    p.add_argument("--seed", type=int, default=0)
    for name, p in sub.choices.items():
        p.add_argument("--work", type=Path, required=True, help="the work dir every step reads and writes")
        if name != "report":
            p.add_argument("--model", type=Path, required=True, help="the model dir (tokenizer and chat template)")
            p.add_argument("--max-len", type=int, default=32768)
            p.add_argument("--call-tokens", type=int, default=1024, help="tokens for a sampled call")
            p.add_argument("--temperature", type=float, default=0.2, help="the run's sampling (off_args: 0.2)")
            p.add_argument("--top-p", type=float, default=0.95)
            p.add_argument("--top-k", type=int, default=64, help="vLLM's default from the model's generation_config.json")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    return {"prep": cmd_prep, "score": cmd_score, "sample": cmd_sample, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
