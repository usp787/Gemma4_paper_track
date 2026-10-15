"""The feedback factorial (README.md section 3, round 2's G2): is the copy cued by the repeated result's surface form,
or by what the result means? G1 (dose.py) re-spelled the call and kept the result; this keeps every call and changes
how the results are rendered, at fixed histories.

A state is the context just after an empty read-only search c that repeats an earlier identical call and got the
identical result r, with no write since c first ran (a repeat state). It is picked by the history alone, whatever the
student did next. First failures, where c ran once, are the controls, as in G1. The relevant results R are every
result in the context identical to r whose command's emptiness means "no match" (faithful_miss); an identical result
of another command, such as a write that printed nothing, stays as recorded.

Arms, each a rendering of the same recorded prompt:
  orig       the prompt as recorded                         dedup  without c's earlier copies (gate1's dedup)
  placebo    R gets "note": NEUTRAL, "id": one ID            semantic  "note": SEMANTIC, "id": one ID
  break      "note": NEUTRAL, "id": an ID of its own each    both      "note": SEMANTIC, "id": its own each
  block      orig, then the empty thought (always-block's prefill)
  think_all  the empty thought where each past model output starts, and at the end
The 2 x 2 is (one ID | varying IDs) x (neutral | semantic note). Both keys go at the end of each result's JSON, the note
first, so the cells add the same tokens at the same places; an ID is 8 digits, one token each. In placebo the ID is
the last result's ID in break, so those two end with identical tokens and differ only in whether the earlier results
match the last one. think_all against block asks whether the block works by breaking the match between "r, then c"
in the history and the end of the context: restoring the thought in the history restores the match.

Subcommands, in this order (loops/feedback.slurm runs them, loops/submit_feedback.sh queues the jobs):
  prep    CPU. Labels each run's calls with gate1.py's rules, picks the states, rebuilds every arm and checks the edits
          against the model's chat template. Writes states.jsonl, prompts.jsonl, prep.json and prep.md.
  score   GPU (vLLM). The probability of the next call being c after each arm: straight to the call, and after the empty
          thought the student often writes first; after block and think_all, which end with it, only the call. Writes
          score.jsonl.
  sample  GPU (vLLM). --samples continuations per arm at the run's sampling, each judged: c exactly, the same search in
          any form, stale, malformed, an escape, and for an escape whether it changes approach. Writes samples.jsonl.
  report  CPU, stdlib only. Per arm; the factorial's effects and the anchors, paired over states with 95% intervals
          over tasks. Writes feedback.md, feedback.json.
score and sample resume: a rerun skips what is already written.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / "gate1")]
import gate1 as g  # noqa: E402  # stdlib at import time: labels, alignment, call blocks, sampling probabilities
import dose as d  # noqa: E402  # the empty-search rules, the search normaliser, call tokens, the paired bootstrap

TEXT = {   # the same number of tokens (prep checks); neither tells the model what to do
    "neutral": "The tool server returned this response to the agent in the current session.",
    "semantic": "The search completed and found zero matches for this pattern in these paths.",
}
CELLS = {"placebo": ("one", "neutral"), "semantic": ("one", "semantic"), "break": ("own", "neutral"),
         "both": ("own", "semantic")}   # arm -> (the results' IDs, their note)
ARMS = ("orig", "dedup") + tuple(CELLS) + ("block", "think_all")
THOUGHT_ARMS = ("block", "think_all")   # their contexts end with the empty thought
PATHS = ("direct", "block")   # the next call straight away, or after the empty thought <|channel>thought\n<channel|>
NOTE, ID = "note", "id"
ID_DIGITS = 8
GREPS = {"grep", "egrep", "fgrep", "rg", "ag", "ack"}
QUOTED = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'[^']*'")
VALUED = {"-A", "-B", "-C", "-m", "-f", "-d", "-D", "-g", "-t", "-T", "--max-count", "--file", "--context",
          "--after-context", "--before-context", "--glob", "--type"}   # grep and rg options that take a value
PROBE_OTHER = 'grep -rn "other_symbol" fastapi/'


# ---------------------------------------------------------------- what an empty result means

def faithful_miss(cmd: str, rc: str) -> bool:
    """Whether "found zero matches for this pattern" is a faithful reading of this command's empty result. Not for an
    inverted search (-v, -L), which finds nothing when every line matched; for an exit-0 result, not for a quiet grep
    (-q exits 0 on a match), xargs, or a find that acts on what it finds."""
    if not d.is_search(cmd):
        return False
    for seg in re.split(r"\|\||&&|[|;]", QUOTED.sub("Q", cmd)):
        toks = seg.split()
        if not toks:
            continue
        prog = toks[0]
        if prog == "git" and toks[1:2] == ["grep"]:
            prog, toks = "grep", toks[1:]
        if prog in GREPS:
            for t in toks[1:]:
                short = set(t[1:]) if re.fullmatch(r"-[A-Za-z]+", t) else set()
                if t in ("--invert-match", "--files-without-match") or short & {"v", "L"}:
                    return False
                if rc == "ok_empty" and (t in ("--quiet", "--silent") or "q" in short):
                    return False
        if prog == "xargs" and rc == "ok_empty":
            return False
        if prog == "find" and any(t in ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-quit") for t in toks):
            return False
    return True


def pattern_of(cmd: str) -> str | None:
    """The first search's pattern, lowercased: grep's first argument, find's -name (heuristic)."""
    segs = d.norm(cmd)
    if not segs or segs[0] == "raw":
        return None
    for prog, _, rest in segs:
        if prog == "git" and rest[:1] == ("grep",):
            prog, rest = "grep", rest[1:]
        if prog in GREPS:
            skip = False
            for x in rest:
                if skip:
                    skip = False
                elif x in VALUED:
                    skip = True
                elif not x.startswith("-"):
                    return x.lower().strip("^$")
        if prog == "find":
            for k, x in enumerate(rest[:-1]):
                if x in ("-name", "-iname", "-path", "-ipath", "-regex", "-iregex"):
                    return rest[k + 1].lower().strip("*")
    return None


def related(a: str | None, b: str | None) -> bool:
    """The same pattern, or one inside the other: a variant of the search rather than a new one."""
    return bool(a and b) and (a == b or (len(a) >= 3 and a in b) or (len(b) >= 3 and b in a))


# ---------------------------------------------------------------- token-level renderings

def edit_response(v: g.Vocab, resp: list[int], note: str, rid: str) -> list[int] | None:
    """A tool response, response:NAME{value:<|"|>{...}<|"|>}<tool_response|>, with "note" and then "id" added at the
    end of its JSON. The JSON is encoded again whole; None if it doesn't re-encode to its own tokens, isn't an object,
    or has either key already."""
    a, b = g.find(resp, [v.quote]), g.rfind(resp, v.quote)
    if a is None or b is None or b <= a:
        return None
    body = v.dec(resp[a + 1:b])
    if v.enc(body) != resp[a + 1:b] or not body.endswith("}"):
        return None
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or NOTE in obj or ID in obj:
        return None
    new = body[:-1] + f", {json.dumps(NOTE)}: {json.dumps(note)}, {json.dumps(ID)}: {json.dumps(rid)}}}"
    return resp[:a + 1] + v.enc(new) + resp[b:]


def render(v: g.Vocab, p: list[int], blocks: list[tuple], note: str, ids: list[str]) -> list[int] | None:
    """p with the responses of blocks (start, end, call; in order) edited, block k getting note and ids[k]."""
    out, prev = [], 0
    for (s, e, call), rid in zip(blocks, ids):
        r0 = s + len(call)
        new = edit_response(v, p[r0:e], note, rid)
        if new is None:
            return None
        out += p[prev:r0] + new
        prev = e
    return out + p[prev:]


def output_starts(v: g.Vocab, p: list[int]) -> list[int]:
    """Where the model's past outputs start: after each model-turn header, and after each tool response the model
    continued. Thinking off strips the empty thought from these places in the history."""
    head, at, out = v.enc("<|turn>model\n"), 0, set()
    while True:
        h = g.find(p, head, at)
        if h is None:
            break
        at = h + len(head)
        out.add(at)
    out |= {e for _, e, _ in g.call_blocks(v, p) if e < len(p) and p[e] != v.turn_close}
    return sorted(x for x in out if x < len(p) and p[x:x + len(v.empty_thought)] != v.empty_thought)


def with_thoughts(v: g.Vocab, p: list[int]) -> list[int]:
    out, prev = [], 0
    for at in output_starts(v, p):
        out += p[prev:at] + v.empty_thought
        prev = at
    return out + p[prev:]


def arm_context(v: g.Vocab, s: dict, p: list[int], arm: str) -> list[int] | None:
    """The arm's context: a rendering of the state's prompt p. None where the arm doesn't apply (dedup without an
    earlier copy; break and both with fewer than 2 relevant results, where one ID can't differ from another)."""
    if arm == "orig":
        return p
    if arm == "dedup":
        dd = g.dedup(v, p)
        return dd[0] if dd else None
    if arm == "block":
        return p + v.empty_thought
    if arm == "think_all":
        return with_thoughts(v, p) + v.empty_thought
    ids, note = CELLS[arm]
    if ids == "own" and len(s["R"]) < 2:
        return None
    bl = g.call_blocks(v, p)
    rid = s["ids"] if ids == "own" else [s["ids"][-1]] * len(s["R"])
    return render(v, p, [bl[k] for k in s["R"]], TEXT[note], rid)


def paths_of(arm: str) -> tuple[str, ...]:
    return ("after",) if arm in THOUGHT_ARMS else PATHS


def make_ids(key: str, n: int) -> list[str]:
    """n different 8-digit IDs, from the state's key."""
    rng, out = random.Random(g.seed_of(key + "|ids")), []
    while len(out) < n:
        x = str(rng.randrange(10 ** (ID_DIGITS - 1), 10 ** ID_DIGITS))
        if x not in out:
            out.append(x)
    return out


def probe_checks(tok, v: g.Vocab) -> list[tuple[str, bool, str]]:
    """The renderings against the model's own template, on a probe: c, c, another empty search, c, all with the same
    result. The template must render the edited results as the token-level edit has them."""
    cmds = [d.PROBE_CMD, d.PROBE_CMD, PROBE_OTHER, d.PROBE_CMD]

    def conv(contents):
        msgs = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix the bug in a.py."}]
        for k, (c, content) in enumerate(zip(cmds, contents)):
            msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": f"c{k}", "type": "function",
                      "function": {"name": "run_command", "arguments": {"command": c}}}]},
                     {"role": "tool", "tool_call_id": f"c{k}", "name": "run_command", "content": content}]
        return v.enc(tok.apply_chat_template(msgs, tools=g.PROBE_TOOLS, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False))
    base = conv([d.PROBE_RES] * 4)
    bl = g.call_blocks(v, base)
    out = [("thinking off leaves the turn open after a tool response", base[-1] == v.resp_close, repr(v.dec(base[-12:]))),
           ("the probe renders 4 call blocks, the last at the end", len(bl) == 4 and bl[-1][1] == len(base), "")]
    if len(bl) != 4:
        return out
    s = {"R": [0, 1, 2, 3], "ids": make_ids("probe", 4)}
    lens = {}
    for arm, (ids, note) in CELLS.items():
        rid = s["ids"] if ids == "own" else [s["ids"][-1]] * 4
        want = conv([d.PROBE_RES[:-1] + f", {json.dumps(NOTE)}: {json.dumps(TEXT[note])}, {json.dumps(ID)}: "
                     f"{json.dumps(r)}}}" for r in rid])
        got = arm_context(v, s, base, arm)
        out.append((f"{arm}: the token-level edit is the template's rendering", got == want,
                    "" if got == want else repr(v.dec((got or [])[-80:]))))
        if got:
            nb = g.call_blocks(v, got)
            lens[arm] = [e - st for st, e, _ in nb]
            objs = []
            for st, e, call in nb:
                r = got[st + len(call):e]
                objs.append(json.loads(v.dec(r[g.find(r, [v.quote]) + 1:g.rfind(r, v.quote)])))
            ok = all(list(o)[-2:] == [NOTE, ID] and {k: o[k] for k in o if k not in (NOTE, ID)} == json.loads(d.PROBE_RES)
                     and o[NOTE] == TEXT[note] and o[ID] == r for o, r in zip(objs, rid))
            out.append((f"{arm}: each result keeps its JSON and ends with note, then id", ok, ""))
    out.append(("the four cells add the same tokens to every result",
                len({tuple(x) for x in lens.values()}) == 1 and len(lens) == 4, json.dumps(lens)))
    out.append(("the neutral and the semantic note are as long", len(v.enc(TEXT["neutral"])) == len(v.enc(TEXT["semantic"])),
                f"{len(v.enc(TEXT['neutral']))} and {len(v.enc(TEXT['semantic']))} tokens"))
    pl, br = arm_context(v, s, base, "placebo"), arm_context(v, s, base, "break")
    b3 = g.call_blocks(v, pl)[3]
    out.append(("placebo and break end with the same last result", pl[b3[0]:] == br[b3[0]:] and pl != br, ""))
    et = v.empty_thought
    has_et = any(base[i:i + len(et)] == et for i in range(len(base) - len(et) + 1))
    out.append(("the template's history has no empty thought before past calls", not has_et, ""))
    ta = with_thoughts(v, base)
    starts = output_starts(v, base)
    back = list(ta)
    for at in reversed([a + k * len(et) for k, a in enumerate(starts)]):
        del back[at:at + len(et)]
    out.append(("think_all puts the empty thought before each of the 4 calls", len(starts) == 4 and back == base
                and all(ta[st - len(et):st] == et for st, _, _ in g.call_blocks(v, ta)), f"{len(starts)} places"))
    dd = g.dedup(v, base)
    out.append(("dedup drops c's 2 earlier copies and keeps the other search",
                dd is not None and dd[1] == 2 and len(g.call_blocks(v, dd[0])) == 2, ""))
    cases = {('grep -rn "x" .', "err_silent"): True, ('grep -rv "x" .', "err_silent"): False,
             ('grep -L "x" a.py', "err_silent"): False, ('grep -q "x" a.py', "err_silent"): True,
             ('grep -q "x" a.py', "ok_empty"): False, ('grep -rn "x" . | head -5', "ok_empty"): True,
             ("find . -name '*x*'", "ok_empty"): True, ("find . -name x -delete", "ok_empty"): False,
             ('grep -rn "-v" .', "err_silent"): True, ("ls tests | xargs grep -l x", "ok_empty"): False}
    bad = [k for k, want in cases.items() if faithful_miss(*k) != want]
    out.append(("what counts as a faithful miss, on 10 cases", not bad, str(bad)))
    return out


# ---------------------------------------------------------------- prep

def candidates(runs: list[Path], counts: Counter) -> list[dict]:
    """Every empty read-only search with a next call on record, as a repeat state or a control (dose.py's reading of
    runs: a task's rerun in a later shard wins, unless it was skipped at the deadline)."""
    cands = []
    for run in runs:
        last = {}
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
                    if not (d.empty_result(c["obs"]) and d.is_search(d.command(c))):
                        continue
                    counts["empty read-only searches"] += 1
                    if i + 1 >= len(cs):
                        counts["  the task's last call (left out)"] += 1
                        continue
                    rec = matched.get(id(cs[i + 1]["step"]))
                    if rec is None:
                        counts["  next call without a calls.jsonl record (left out)"] += 1
                        continue
                    occ = [j for j in range(i + 1) if cs[j]["key"] == c["key"]]
                    content = (c["step"].get("observation") or {}).get("content")
                    if len(occ) == 1:
                        if d.norm(d.command(c)) in {d.norm(d.command(y)) for y in cs[:i] if y["fn"] == "run_command"}:
                            counts["  first runs of a search written differently earlier (left out)"] += 1
                            continue
                        kind = "control"
                    elif any((cs[j]["step"].get("observation") or {}).get("content") != content for j in occ):
                        counts["  repeats whose result changed (left out)"] += 1
                        continue
                    elif any((y["fn"] in ("edit_file", "write_file") and isinstance(y["obs"], dict)
                              and y["obs"].get("status") == "ok") or (y["fn"] == "run_command" and d.WRITE.search(d.command(y)))
                             for y in cs[occ[0]:i]):
                        counts["  repeats with a write since c first ran (left out)"] += 1
                        continue
                    else:
                        kind = "k2" if len(occ) == 2 else "k3+"
                    counts[f"  candidates: {kind}"] += 1
                    cands.append({"run": run.name, "shard": shard, "task": tid, "i": i, "cs": cs, "kind": kind,
                                  "line": rec["line"], "resolved": bool(res.get("resolved"))})
    return cands


def pick_order(cands: list[dict], per_kind: int, seed: int) -> tuple[list[dict], list[dict]]:
    """(repeat states, controls), each in the order to try. Round-robin over task runs: per round, one k2 and one k3+
    state from each, up to per_kind of each kind (different calls first); controls one per task run."""
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    for x in cands:
        groups[x["run"], x["task"]][x["kind"]].append(x)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        for kind, xs in groups[k].items():
            rng.shuffle(xs)
            seen, first, rest = set(), [], []
            for x in xs:   # one state per call before a second of the same call
                (rest if x["cs"][x["i"]]["key"] in seen else first).append(x)
                seen.add(x["cs"][x["i"]]["key"])
            groups[k][kind] = first + rest
    reps = [groups[k][kind][r] for r in range(per_kind) for k in keys for kind in ("k2", "k3+")
            if len(groups[k][kind]) > r]
    ctrl = [groups[k]["control"][0] for k in keys if groups[k]["control"]]
    return reps, ctrl


def build_state(v: g.Vocab, x: dict, p: list[int], a, built: Counter) -> dict | None:
    """The state's record, or None (counted in built) if a check on its prompt fails."""
    cs, i = x["cs"], x["i"]
    c, nxt = cs[i], cs[i + 1]
    bl = g.call_blocks(v, p)
    if not bl or bl[-1][1] != len(p) or p[-1] != v.resp_close:
        built["the next prompt doesn't end with c's block, e.g. after a compaction (left out)"] += 1
        return None
    s0, e0, call = bl[-1]
    parsed = g.parse_call(v.dec(call))
    cmd = d.command(c)
    if not parsed or g.call_key(*parsed) != c["key"] or d.call_ids(v, cmd) != list(call):
        built["the prompt's last block isn't c's tokens (left out)"] += 1
        return None
    resp = p[s0 + len(call):e0]
    mine = [k for k, (_, _, cl) in enumerate(bl) if cl == call]
    if x["kind"] != "control" and len(mine) < 2:
        built["c's earlier copies aren't in the context, e.g. compacted (left out)"] += 1
        return None
    if not faithful_miss(cmd, c["rc"]):
        built["c's empty result doesn't mean no match, e.g. grep -v or -q (left out)"] += 1
        return None
    same = [k for k, (st, e, cl) in enumerate(bl) if p[st + len(cl):e] == resp]
    if not set(mine) <= set(same):
        built["an earlier copy of c shows another result (left out)"] += 1
        return None

    def miss(k):
        q = g.parse_call(v.dec(bl[k][2]))
        return bool(q) and q[0] == "run_command" and faithful_miss(str(q[1].get("command", "")), c["rc"])
    R = [k for k in same if miss(k)]   # an identical result from a write (sed -i, cat > f) stays as recorded
    key = f"{x['run']}/{x['shard'].name}/{x['task']}/{x['i']}"
    s = {"key": key, "R": R, "ids": make_ids(key, len(R))}
    ctx = {arm: arm_context(v, s, p, arm) for arm in ARMS}
    if any(ctx[arm] is None for arm in CELLS if CELLS[arm][0] == "one" or len(R) >= 2):
        built["a relevant result's JSON can't take the note (left out)"] += 1
        return None
    cells = [ctx[arm] for arm in CELLS if ctx[arm] is not None]
    per_block = {tuple(e - st for st, e, _ in g.call_blocks(v, q)) for q in cells}
    if len({len(q) for q in cells}) != 1 or len(per_block) != 1:
        built["the cells' added tokens differ (left out)"] += 1
        return None
    longest = max(len(q) for q in ctx.values() if q is not None)
    if longest + len(v.empty_thought) + max(len(call), a.call_tokens) + 1 > a.max_len:
        built["too long for --max-len (left out)"] += 1
        return None
    if x["kind"] != "control" and ctx["dedup"] is None:
        built["dedup doesn't apply (left out)"] += 1
        return None
    run_len = 0
    for _, _, cl in reversed(bl):
        if cl != call:
            break
        run_len += 1
    built[f"picked: {x['kind']}"] += 1
    return {"key": key, "run": x["run"], "shard": x["shard"].name, "task": x["task"], "resolved": x["resolved"],
            "i": c["i"], "n_calls": len(cs), "command": cmd, "call_key": c["key"], "rc": c["rc"], "kind": x["kind"],
            "k": len(mine), "run_len": run_len, "R": R, "n_R": len(R), "n_R_other": len(R) - len(mine),
            "n_same_unedited": len(same) - len(R),
            "ids": s["ids"], "n_blocks": len(bl), "n_thoughts": len(output_starts(v, p)) + 1,
            "added": len(cells[0]) - len(p), "stale_keys": nxt["stale_keys"], "next_cls": nxt["cls"],
            "next_copies": nxt["same_as_prev"], "n_prompt": len(p), "call_len": len(call)}


def cmd_prep(a) -> int:
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    v = g.Vocab(tok)
    a.work.mkdir(parents=True, exist_ok=True)
    checks = probe_checks(tok, v)
    counts = Counter()
    cands = candidates(a.runs, counts)
    reps, ctrl = pick_order(cands, a.per_kind, a.seed)
    want = reps[:2 * a.target] + ctrl[:2 * a.controls]   # spares, for states that fail a check
    lines_by_shard = defaultdict(dict)
    for x in want:
        lines_by_shard[x["shard"]][x["line"]] = f"{x['run']}/{x['shard'].name}/{x['task']}/{x['i']}"
    prompts = {}
    for shard, lines in lines_by_shard.items():
        prompts.update(g.fetch_prompts(shard, lines))
    states, rows, built = [], [], Counter()
    for pool, n in ((reps, a.target), (ctrl, a.controls)):
        got = 0
        for x in pool[:2 * n]:
            if got >= n:
                break
            pr = prompts.get(f"{x['run']}/{x['shard'].name}/{x['task']}/{x['i']}")
            s = build_state(v, x, pr["prompt_ids"], a, built) if pr else None
            if s is None:
                continue
            states.append(s)
            rows.append({"key": s["key"], "prompt_ids": pr["prompt_ids"]})
            got += 1
    g.write_jsonl(a.work / "states.jsonl", states)
    g.write_jsonl(a.work / "prompts.jsonl", rows)
    ok = all(ch[1] for ch in checks)
    rep = [s for s in states if s["kind"] != "control"]
    summary = {"runs": [str(r) for r in a.runs], "counts": dict(counts), "built": dict(built),
               "states": len(states), "tasks": len({s["task"] for s in states}),
               "repeat_tasks": len({s["task"] for s in rep}), "task_runs": len({(s["run"], s["task"]) for s in rep}),
               "texts": TEXT, "checks": [{"check": n, "pass": p, "detail": dt} for n, p, dt in checks],
               "secs": round(time.time() - t0, 1)}
    (a.work / "prep.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    cnt = lambda f, xs: ", ".join(f"{k} {n}" for k, n in sorted(Counter(map(f, xs)).items(), key=lambda kv: str(kv[0])))
    lines = [f"# Feedback factorial prep: {a.work.name}", "", f"Runs: {', '.join(r.name for r in a.runs)}. "
             f"{summary['secs']} s.", "", "## Template checks", ""]
    lines += [f"- {'PASS' if p else 'FAIL'} {n}{': ' + dt if dt else ''}" for n, p, dt in checks]
    lines += ["", "## States", ""] + [f"- {k}: {n}" for k, n in counts.items()]
    lines += [f"- tried in round-robin order over task runs (at most {a.per_kind} of each kind per task run; "
              f"targets {a.target} repeat states and {a.controls} controls):"] + [f"  - {k}: {n}" for k, n in built.items()]
    lines += [f"- repeat states: {len(rep)} in {summary['repeat_tasks']} tasks ({summary['task_runs']} task runs); "
              f"controls: {len(states) - len(rep)}",
              f"- repeat states by kind (times c ran): {cnt(lambda s: s['kind'], rep)}; by copies of c in the context: "
              f"{cnt(lambda s: min(s['k'], 5), rep)} (5 = 5 or more); by result: {cnt(lambda s: s['rc'], rep)}",
              f"- copies (c also the call before) {sum(s['run_len'] >= 2 for s in rep)}, returns "
              f"{sum(s['run_len'] < 2 for s in rep)}; with another search's identical result in R "
              f"{sum(s['n_R_other'] > 0 for s in rep)}; with an identical result left as recorded (not a search) "
              f"{sum(s['n_same_unedited'] > 0 for s in states)}",
              f"- relevant results per state: {cnt(lambda s: min(s['n_R'], 10), states)} (10 = 10 or more)",
              f"- the student's next call: {cnt(lambda s: s['next_cls'], rep)}",
              f"- tokens added by each cell: mean {sum(s['added'] for s in states) / max(1, len(states)):.0f}", "",
              "## Notes", "", f"- neutral: \"{TEXT['neutral']}\"", f"- semantic: \"{TEXT['semantic']}\"", "",
              "## An example: the last result in each cell", ""]
    ex = next((s for s in random.Random(a.seed).sample(rep, len(rep)) if s["n_R"] >= 2), None)
    if ex:
        p = next(r["prompt_ids"] for r in rows if r["key"] == ex["key"])
        lines += [f"{ex['task']}, call {ex['i']} of {ex['n_calls']}: `{g.trim(ex['command'], 140)}`, "
                  f"{ex['k']} copies in the context, {ex['n_R']} relevant results", ""]
        for arm in CELLS:
            q = arm_context(v, ex, p, arm)
            st, e, cl = g.call_blocks(v, q)[-1]
            lines.append(f"- {arm}: `{g.trim(v.dec(q[st + len(cl):e]), 400)}`")
    (a.work / "prep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not ok:
        print("FAIL: a template check failed; the GPU steps would score renderings the template never makes")
        return 1
    if not rep:
        print("FAIL: no repeat states picked")
        return 1
    return 0


# ---------------------------------------------------------------- score and sample (vLLM)

def load_work(work: Path):
    states = list(g.iter_jsonl(work / "states.jsonl"))
    prompts = {r["key"]: r["prompt_ids"] for r in g.iter_jsonl(work / "prompts.jsonl")}
    return states, prompts


def score_specs(v: g.Vocab, s: dict, p: list[int], arms, done: set, a, skipped: Counter) -> list[list[tuple]]:
    """A state's requests, each a list of (row key, arm, path, where the scored part starts). orig's path through the
    empty thought is block's context followed by c: one request scores both."""
    jobs = {}
    for arm in arms:
        ctx = arm_context(v, s, p, arm)
        if ctx is None:
            skipped[f"{arm} doesn't apply"] += 1
            continue
        for way in paths_of(arm):
            key = f"{s['key']}|{arm}|{way}"
            if key in done:
                continue
            if len(ctx) + (len(v.empty_thought) if way == "block" else 0) + s["call_len"] + 1 > a.max_len:
                skipped["longer than --max-len"] += 1
                continue
            tag = "orig+thought" if (arm, way) in (("orig", "block"), ("block", "after")) else key
            jobs.setdefault(tag, []).append((key, arm, way, len(ctx)))
    return list(jobs.values())


def scored_sequence(v: g.Vocab, s: dict, p: list[int], spec: tuple) -> list[int]:
    """The request's tokens: the arm's context, the empty thought on the block path, then c."""
    _, arm, way, _ = spec
    return arm_context(v, s, p, arm) + (v.empty_thought if way == "block" else []) + d.call_ids(v, s["command"])


def cmd_score(a) -> int:
    """Prompt logprobs of c after every arm's context, on each of its paths."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    v = d.vocab(a)
    states, prompts = load_work(a.work)
    path = a.work / "score.jsonl"
    done = {r["key"] for r in g.iter_jsonl(path)}
    skipped, todo = Counter(), []
    for s in states:
        todo += [(s, specs) for specs in score_specs(v, s, prompts[s["key"]], a.arms, done, a, skipped)]
    print(f"score: {len(todo)} requests for {sum(len(x[1]) for x in todo)} rows to do, {len(done)} done, "
          f"skipped {dict(skipped)}", flush=True)
    if not todo:
        return 0
    llm = g.make_llm(a, prefix_caching=False)   # as gate1's score: prompt logprobs need every position computed
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=a.topk, detokenize=False)
    t0, n_tok, n_done = time.time(), 0, 0
    for k in range(0, len(todo), a.batch):
        part = [(s, scored_sequence(v, s, prompts[s["key"]], specs[0]), specs)   # built per batch: memory stays small
                for s, specs in todo[k:k + a.batch]]
        outs = llm.generate([TokensPrompt(prompt_token_ids=f) for _, f, _ in part], sp, use_tqdm=False)
        rows = []
        for (s, full, specs), o in zip(part, outs):
            n_tok += len(full)
            for key, arm, way, start in specs:
                lp = {"raw": 0.0, "run": 0.0, "t1": 0.0}
                for t in range(start, len(full)):
                    dd = o.prompt_logprobs[t]
                    top = sorted(((i, e) for i, e in dd.items() if e.rank is not None and e.rank <= a.topk),
                                 key=lambda ie: ie[1].rank)[:a.topk]
                    tids, tlps = [i for i, _ in top], [e.logprob for _, e in top]
                    lp["raw"] += dd[full[t]].logprob
                    lp["run"] += g.sampled_logprob(tids, tlps, full[t], a.temperature, a.top_p, a.top_k)
                    lp["t1"] += g.sampled_logprob(tids, tlps, full[t], 1.0, a.top_p, a.top_k)
                rows.append({"key": key, "state": s["key"], "arm": arm, "path": way, "n_ctx": start,
                             "n": len(full) - start,
                             **{f"logp_{m}": round(x, 4) if math.isfinite(x) else None for m, x in lp.items()}})
        g.write_jsonl(path, rows, "a")
        n_done += len(part)
        el = time.time() - t0
        print(f"  {n_done}/{len(todo)}, {el:.0f} s, {n_tok / max(el, 1e-9):.0f} prompt tok/s", flush=True)
    return 0


def judge(v: g.Vocab, s: dict, ids: list[int]) -> dict:
    """One sampled continuation: c exactly, the same search in any form, stale, malformed, an escape; and an escape's
    approach: a variant of c's search, a new search, or another tool (the last two change approach)."""
    opened = ids[:len(v.empty_thought)] == v.empty_thought
    parsed = g.parse_call(v.dec(ids))
    key = g.call_key(*parsed) if parsed else None
    cmd = str(parsed[1].get("command", "")) if parsed and parsed[0] == "run_command" else None
    bad = parsed is not None and g.malformed(*parsed)
    stale = key in set(s["stale_keys"])
    escape = parsed is not None and not stale and not bad
    approach = None
    if escape:
        if cmd is not None and d.is_search(cmd):
            approach = "variant" if related(pattern_of(cmd), pattern_of(s["command"])) else "new_search"
        else:
            approach = "other_tool"
    return {"has_call": parsed is not None, "opened_block": opened, "exact_c": key == s["call_key"],
            "same_search": cmd is not None and d.norm(cmd) == d.norm(s["command"]), "stale": stale, "malformed": bad,
            "escape": escape, "approach": approach, "change": approach in ("new_search", "other_tool"),
            "call_text": g.trim(v.dec(ids), 300)}


def cmd_sample(a) -> int:
    """--samples continuations per arm, one request each. Every arm of a state uses the state's seed."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    v = d.vocab(a)
    states, prompts = load_work(a.work)
    path = a.work / "samples.jsonl"
    done = {r["key"] for r in g.iter_jsonl(path)}
    todo, skipped = [], Counter()
    for s in sorted(states, key=lambda s: s["key"]):
        for arm in a.arms:
            key = f"{s['key']}|{arm}"
            if key in done:
                continue
            ctx = arm_context(v, s, prompts[s["key"]], arm)
            if ctx is None:
                skipped[f"{arm} doesn't apply"] += 1
                continue
            if len(ctx) + a.call_tokens > a.max_len:
                skipped["longer than --max-len"] += 1
                continue
            todo.append((key, s, arm))
    print(f"sample: {len(todo)} to do, {len(done)} done, skipped {dict(skipped)}; {a.samples} each, "
          f"T={a.temperature}, top_p {a.top_p}, top_k {a.top_k}", flush=True)
    if not todo:
        return 0
    llm = g.make_llm(a, prefix_caching=True)
    stop = [v.call_close]
    t0 = time.time()
    for k in range(0, len(todo), a.chunk):
        part = [(key, s, arm, arm_context(v, s, prompts[s["key"]], arm)) for key, s, arm in todo[k:k + a.chunk]]
        sps = [SamplingParams(n=a.samples, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                              max_tokens=a.call_tokens, stop_token_ids=stop, seed=g.seed_of(s["key"]),
                              detokenize=False) for _, s, _, _ in part]
        outs = llm.generate([TokensPrompt(prompt_token_ids=ctx) for *_, ctx in part], sps, use_tqdm=False)
        rows = [{"key": key, "state": s["key"], "arm": arm, "n_ctx": len(ctx),
                 "samples": [dict(judge(v, s, g.ended_with(x, stop)), finish=x.finish_reason) for x in o.outputs]}
                for (key, s, arm, ctx), o in zip(part, outs)]
        g.write_jsonl(path, rows, "a")
        print(f"  {k + len(part)}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------- report (stdlib)

SAMPLE_FIELDS = ("exact_c", "same_search", "stale", "malformed", "escape", "change", "has_call", "opened_block")
CONTRASTS = {   # name -> {arm: weight}: a state's value is the weighted sum of its arms' values
    "surface: varying IDs (main effect)": {"break": .5, "both": .5, "placebo": -.5, "semantic": -.5},
    "meaning: the semantic note (main effect)": {"semantic": .5, "both": .5, "placebo": -.5, "break": -.5},
    "interaction: (both − semantic) − (break − placebo)": {"both": 1, "semantic": -1, "break": -1, "placebo": 1},
    "break − placebo": {"break": 1, "placebo": -1},
    "semantic − placebo": {"semantic": 1, "placebo": -1},
    "both − placebo": {"both": 1, "placebo": -1},
    "placebo − orig (the added tokens themselves)": {"placebo": 1, "orig": -1},
    "dedup − orig": {"dedup": 1, "orig": -1},
    "both − dedup": {"both": 1, "dedup": -1},
    "block − orig": {"block": 1, "orig": -1},
    "think_all − block": {"think_all": 1, "block": -1},
    "think_all − orig": {"think_all": 1, "orig": -1},
}
PAIRED_METRICS = ("p(c), run's sampling", "p(c), exact", "samples: exact_c", "samples: escape", "samples: change")


def cmd_report(a) -> int:
    states = {s["key"]: s for s in g.iter_jsonl(a.work / "states.jsonl")}
    sc = defaultdict(dict)   # (state, arm) -> {path: row}
    for r in g.iter_jsonl(a.work / "score.jsonl"):
        sc[r["state"], r["arm"]][r["path"]] = r
    smp = {(r["state"], r["arm"]): r["samples"] for r in g.iter_jsonl(a.work / "samples.jsonl")}

    def p(st, arm, m="run"):
        rows = [sc.get((st, arm), {}).get(w) for w in paths_of(arm)]
        if any(r is None for r in rows):
            return None
        return sum(math.exp(r[f"logp_{m}"]) if r[f"logp_{m}"] is not None else 0.0 for r in rows)

    def share(st, arm, f):
        xs = smp.get((st, arm))
        return sum(bool(x[f]) for x in xs) / len(xs) if xs else None
    metrics = {"p(c), run's sampling": lambda st, arm: p(st, arm), "p(c), exact": lambda st, arm: p(st, arm, "raw"),
               "p(c), T=1": lambda st, arm: p(st, arm, "t1"),
               **{f"samples: {f}": (lambda f: lambda st, arm: share(st, arm, f))(f) for f in SAMPLE_FIELDS}}
    arms = [arm for arm in ARMS if any(k[1] == arm for k in list(sc) + list(smp))]
    rep = [k for k, s in states.items() if s["kind"] != "control"]
    ctrl = [k for k, s in states.items() if s["kind"] == "control"]
    meta = json.loads((a.work / "prep.json").read_text(encoding="utf-8")) if (a.work / "prep.json").is_file() else {}
    out = {"states": len(states), "repeat_states": len(rep), "controls": len(ctrl),
           "tasks": len({states[k]["task"] for k in rep}), "arms": {}, "paired": {}}
    n_smp = max((len(x) for x in smp.values()), default=0)
    lines = [f"# The feedback factorial: {a.work.name}", "",
             f"{len(rep)} repeat states in {out['tasks']} tasks, and {len(ctrl)} first failures as controls, from "
             f"{', '.join(Path(r).name for r in meta.get('runs', []))}. A repeat state ends just after an empty "
             "read-only search c that repeats an earlier identical call with the identical result (README.md section "
             "3, G2). The cells edit every result identical to the last one: one ID or an ID of its own each, and a "
             "neutral or a semantic note. p(c) is the next call being c once more.", "",
             "p(c) adds the next call's two starts (straight away, or after the empty thought the student writes), "
             "except after block and think_all, which end with it. Rebuilt from the top-20 logprobs at the run's "
             "sampling (T=0.2, top_p 0.95, top_k 64); exact is the model's own probability (T=1, no truncation). "
             f"Samples: {n_smp} per arm and state at the run's sampling, the state's seed in every arm.", ""]
    for title, keys in (("Repeat states", rep), ("Controls (first failures)", ctrl)):
        if not keys:
            continue
        lines += [f"## {title}, by arm", "",
                  "| arm | states | p(c), run's sampling | p(c), exact | exact c | same search | stale | malformed | "
                  "escape | changes approach | no call |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        out["arms"][title] = {}
        for arm in arms:
            vals = {name: [f(k, arm) for k in keys] for name, f in metrics.items()}
            vals = {name: [x for x in xs if x is not None] for name, xs in vals.items()}
            row = {name: (sum(xs) / len(xs) if xs else None) for name, xs in vals.items()}
            row["states"] = len(vals["p(c), run's sampling"]) or len(vals["samples: exact_c"])
            out["arms"][title][arm] = row
            nc = None if row["samples: has_call"] is None else 1 - row["samples: has_call"]
            lines.append(f"| {arm} | {row['states']} | " + " | ".join(g.fmt(x, 3) for x in (
                row["p(c), run's sampling"], row["p(c), exact"], row["samples: exact_c"], row["samples: same_search"],
                row["samples: stale"], row["samples: malformed"], row["samples: escape"], row["samples: change"], nc))
                + " |")
        lines.append("")

    def contrast(keys, weights, f):
        diffs = defaultdict(list)
        for k in keys:
            xs = {arm: f(k, arm) for arm in weights}
            if all(x is not None for x in xs.values()):
                diffs[states[k]["task"]].append(sum(w * xs[arm] for arm, w in weights.items()))
        return d.paired(diffs, a.reps, a.seed), sum(len(x) for x in diffs.values()), len(diffs)
    lines += ["## The factorial and the anchors, on repeat states", "",
              f"Mean over states of each contrast; 95% interval from {a.reps} resamples of tasks (seed {a.seed}). "
              "A state counts where all of a contrast's arms apply; break and both need at least 2 relevant results.", "",
              "| contrast | states | tasks | " + " | ".join(PAIRED_METRICS) + " |", "|---|---|---|" + "---|" * len(PAIRED_METRICS)]
    for name, w in CONTRASTS.items():
        if not set(w) <= set(arms):
            continue
        cells, rec, n, nt = [], {}, 0, 0
        for m in PAIRED_METRICS:
            (point, lo, hi), n, nt = contrast(rep, w, metrics[m])
            rec[m] = {"diff": point, "ci": [lo, hi], "n": n}
            cells.append("-" if point is None else f"{point:+.3f} ({lo:+.3f} to {hi:+.3f})")
        out["paired"][name] = rec
        lines.append(f"| {name} | {n} | {nt} | " + " | ".join(cells) + " |")
    groups = [("repeat: k = 2 copies in the context", lambda s: s["kind"] != "control" and s["k"] == 2),
              ("repeat: k ≥ 3", lambda s: s["kind"] != "control" and s["k"] >= 3),
              ("repeat: a copy (c also the call before)", lambda s: s["kind"] != "control" and s["run_len"] >= 2),
              ("repeat: a return", lambda s: s["kind"] != "control" and s["run_len"] < 2),
              ("repeat: err_silent (exit 1)", lambda s: s["kind"] != "control" and s["rc"] == "err_silent"),
              ("repeat: ok_empty (exit 0)", lambda s: s["kind"] != "control" and s["rc"] == "ok_empty"),
              ("repeat: student copied next", lambda s: s["kind"] != "control" and s["next_copies"]),
              ("repeat: student did not", lambda s: s["kind"] != "control" and not s["next_copies"]),
              ("controls", lambda s: s["kind"] == "control"),
              ("controls with another identical result", lambda s: s["kind"] == "control" and s["n_R"] >= 2)]
    lines += ["", "## By group (p(c), run's sampling)", "", "| group | states | tasks | " + " | ".join(arms) + " |",
              "|---|---|---|" + "---|" * len(arms)]
    for name, keep in groups:
        keys = [k for k, s in states.items() if keep(s)]
        if not keys:
            continue
        cells = []
        for arm in arms:
            xs = [x for x in (p(k, arm) for k in keys) if x is not None]
            cells.append(g.fmt(sum(xs) / len(xs), 3) if xs else "-")
        lines.append(f"| {name} | {len(keys)} | {len({states[k]['task'] for k in keys})} | " + " | ".join(cells) + " |")
    groups_sub = [(n, f) for n, f in groups if n.startswith("repeat: k") or n.startswith("repeat: err") or n.startswith("repeat: ok")]
    lines += ["", "## The two main effects by group (p(c), exact)", "",
              "The exact probability, because the run's sampling pushes p(c) towards 0 or 1, which bends additive "
              "contrasts and makes interactions out of floors and ceilings.", "",
              "| group | states | tasks | surface | meaning | interaction |", "|---|---|---|---|---|---|"]
    for name, keep in groups_sub:
        keys = [k for k in rep if keep(states[k])]
        cells, n, nt = [], 0, 0
        for cname in list(CONTRASTS)[:3]:
            (point, lo, hi), n, nt = contrast(keys, CONTRASTS[cname], metrics["p(c), exact"])
            cells.append("-" if point is None else f"{point:+.3f} ({lo:+.3f} to {hi:+.3f})")
        lines.append(f"| {name} | {n} | {nt} | " + " | ".join(cells) + " |")
    lines += ["", "Reading it: a negative surface effect means the identical surface form of repeated results helps "
              "cue the copy; a negative meaning effect beyond the placebo means the faithful reading helps the model off "
              "it. Report an interaction rather than one cause, and a null as a narrower claim, not a reason to sweep "
              "wordings. think_all near orig and above block says the block works by breaking the match between the "
              "history's \"r, then c\" and the end of the context; think_all near block says it works through the "
              "thinking mode. An escape changes approach when it is a new search or another tool, not a variant of c's "
              "search (a heuristic on the search's pattern). Intervals are over about as many tasks as the table says, "
              "not over states.", ""]
    (a.work / "feedback.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (a.work / "feedback.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="run dirs: shard_*/{calls.jsonl,traces/,task_results.jsonl}")
    p.add_argument("--target", type=int, default=128, help="repeat states at most")
    p.add_argument("--controls", type=int, default=32, help="first failures at most, one per task run")
    p.add_argument("--per-kind", type=int, default=2, help="repeat states of each kind (k2, k3+) per task run at most")
    p.add_argument("--seed", type=int, default=0)
    for name, chunk in (("score", 2048), ("sample", 8192)):
        p = sub.add_parser(name)
        p.add_argument("--arms", type=lambda s: [x for x in s.split(",") if x], default=list(ARMS))
        p.add_argument("--gpu-mem", type=float, default=0.85)
        p.add_argument("--chunk-tokens", type=int, default=chunk,
                       help="vLLM's max_num_batched_tokens: score keeps it small, as gate1's score does")
        p.add_argument("--max-seqs", type=int, default=32)
        p.add_argument("--topk", type=int, default=20, help="top logprobs per position (vLLM's max_logprobs)")
    sub.choices["score"].add_argument("--batch", type=int, default=8, help="requests per vLLM call (prompt logprobs are big)")
    sub.choices["sample"].add_argument("--samples", type=int, default=3, help="continuations per arm and state")
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
    if getattr(a, "arms", None):
        bad = [x for x in a.arms if x not in ARMS]
        if bad:
            ap.error(f"unknown arms {bad}; the arms are {', '.join(ARMS)}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    return {"prep": cmd_prep, "score": cmd_score, "sample": cmd_sample, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
