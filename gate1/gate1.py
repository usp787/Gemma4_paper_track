"""Gate 1 of the OPSD study (../opsd_assessment.md §1.4, change 1): before any training, where does each candidate
teacher disagree with the thinking-off student, and what breaks the student's stuck loops?

It scores recorded trajectories of the student: the calls.jsonl that baseline/token_proxy.py writes in the main-track
repo, and the traces. No new rollouts, no training. README.md (next to this file) defines the call classes, the token
roles, the contexts and the gate.

Subcommands, in this order (gate1.slurm runs them, submit_gate1.sh queues the jobs):
  prep      CPU. Aligns each run's calls.jsonl with its traces. Labels every main-agent call with trigger_replay.py's
            rules plus the stuck-loop classes, and every completion token with its role in the tool call. Picks the
            states to score, and checks the token-level prompt edits against the model's own chat template.
            Writes states.jsonl, prompts.jsonl, priv.jsonl (with --data), prep.md and prep.json.
  generate  GPU (vLLM). T-think at every picked state: the same weights with thinking on write a thought of at most
            --budget tokens, then their own next call. Writes gen.jsonl. With --context T-open, the thought is forced
            open instead: the thought channel is prefilled and can't close before --min-thought tokens, so the model
            thinks at every state, not only where it chooses to. Writes gen_T-open.jsonl.
  score     GPU (vLLM). Prompt logprobs of the student's own completion under each context, S (the student itself)
            first. Per token: the log-ratio and the top-k forward KL against S. Per call: the probability of the
            student's call at T=1 and at the run's sampling. Writes score_<context>.jsonl.
  own       GPU (vLLM). The model's own next call after a context that ends in a forced empty thought (block,
            C-none): one sample at the run's sampling, checked as T-think's own call is. Writes own_<context>.jsonl.
  report    CPU, stdlib only. Tables by call class and token role, the replay, the gate, the forced empty thought, and
            the forced thought. Writes gate1.md and gate1.json.
generate, score and own resume: a rerun skips the states already written.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics as st
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adaptive_inference"))
import trigger_replay as tr  # noqa: E402  # the shared rules: stale repeat, malformed, bad

SERVED = "gemma-4-31b-it-qat-w4a16-ct"
EMPTY = {"err_silent", "ok_empty"}   # what a run_command that found nothing returns (prep's result_class)
# Call classes, by the call the student made. A call gets the first that fits (call_class).
STUCK = ("copy_2nd", "copy_3rd_plus", "return_to_earlier", "stale_other", "malformed")
CLASSES = STUCK + ("after_miss", "after_bad", "after_error", "first", "routine")
# The default pick, per class: (at most per task, at most in all). About 650 states.
CAPS = {"copy_2nd": (99, 80), "copy_3rd_plus": (3, 80), "return_to_earlier": (4, 110), "stale_other": (2, 50),
        "malformed": (99, 40), "after_miss": (3, 100), "after_bad": (2, 50), "after_error": (2, 40),
        "first": (1, 35), "routine": (2, 120)}
CALL_ROLES = ("tool", "argname", "command", "path", "code", "value")   # the content of a call; struct and think are not
CONTEXTS = ("S", "C-none", "T-think", "T-hint", "note", "dedup", "block", "T-open", "T-patch", "T-patch-on", "T-loc",
            "C-wrong")
THINKING_ON = {"C-none", "T-think", "T-open", "T-patch-on"}
OWN = ("block", "C-none")   # the contexts own samples after: the empty thought forced with thinking off, and on
# The contexts with a written thought, and the file generate writes it to. T-think thinks where it chooses to; T-open
# is forced to (the forced thought, report section 4).
GEN = {"T-think": "gen.jsonl", "T-open": "gen_T-open.jsonl"}
MIN_THOUGHT = 32   # T-open's default --min-thought: below 98% of T-think's own thoughts on the first run (min 28, p05 57)
PATCHED = {"T-patch": "patch", "T-patch-on": "patch", "T-loc": "loc", "C-wrong": "wrong"}
# Gemma 4's chat-template tokens (the model's tokenizer_config.json names them all)
SPECIAL = {"turn_open": "<|turn>", "turn_close": "<turn|>", "think": "<|think|>", "ch_open": "<|channel>",
           "ch_close": "<channel|>", "tool_open": "<|tool>", "tool_close": "<tool|>", "call_open": "<|tool_call>",
           "call_close": "<tool_call|>", "resp_open": "<|tool_response>", "resp_close": "<tool_response|>",
           "quote": '<|"|>'}
Q = SPECIAL["quote"]
# Notes a controller could add before the call, from what is already in the context (the "note" context). Like the
# harness's own budget_warning, a note is one more key at the end of the last tool response.
MISS_NOTE = ("This command printed nothing: what it searched for is not in those files. Do not run it again; "
             "search for something else, or read the most likely file.")
REPEAT_NOTE = ("This call was identical to the one before it and returned the same result. Repeating it again will "
               "not change the result: try a different approach.")
PRIV_TEXT = {   # what each patched teacher sees at the end of the system prompt
    "patch": "Reference fix for this task, which only you can see:\n```diff\n{text}\n```",
    "loc": "Where the reference fix for this task goes, which only you can see: {text}",
    "wrong": "Reference fix for this task, which only you can see:\n```diff\n{text}\n```",
}
PATCH_CHARS = 8000   # longer gold patches are cut, to keep every variant inside the 32k context


# ---------------------------------------------------------------- shared helpers

def iter_jsonl(path: Path):
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def write_jsonl(path: Path, rows, mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def find(seq: list[int], sub: list[int], start: int = 0, end: int | None = None) -> int | None:
    end = len(seq) if end is None else min(end, len(seq))
    n = len(sub)
    for i in range(start, end - n + 1):
        if seq[i:i + n] == sub:
            return i
    return None


def rfind(seq: list[int], tok: int, start: int = 0, end: int | None = None) -> int | None:
    end = len(seq) if end is None else end
    for i in range(end - 1, start - 1, -1):
        if seq[i] == tok:
            return i
    return None


class Vocab:
    """The template's special tokens and fixed token sequences, looked up in the model's own tokenizer."""

    def __init__(self, tok, edit_mode: str = "tokens"):
        self.tok = tok
        self.edit_mode = edit_mode   # "tokens", or "text" if prep found the token-level thinking-on edit off the template
        for name, s in SPECIAL.items():
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit(f"{s!r} is not one token in this tokenizer: {ids}")
            setattr(self, name, ids[0])
        self.empty_thought = self.enc("<|channel>thought\n<channel|>")   # what thinking off prefills, or writes
        self.open_thought = self.enc("<|channel>thought\n")              # T-open's prefill: a thought, left open
        if self.empty_thought != self.open_thought + [self.ch_close]:
            raise SystemExit(f"the open thought {self.open_thought} is no prefix of the empty one {self.empty_thought}")
        self.think_insert = self.enc("<|think|>\n")                     # what thinking on adds to the system turn
        self.sys_header = self.enc("<|turn>system\n")
        self.close_thought = self.enc("\n<channel|>")                   # ends a thought cut at --budget
        eos = tok.eos_token_id
        self.structural = {getattr(self, n) for n in SPECIAL} | set(eos if isinstance(eos, list) else [eos]) \
            | {tok.bos_token_id}

    def enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def dec(self, ids) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def pieces(self, ids) -> list[str]:
        """Each token's own text, for the role parser: SentencePiece's '▁' as a space, byte tokens as their byte."""
        out = []
        for p in self.tok.convert_ids_to_tokens(list(ids)):
            p = p or ""
            m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", p)
            out.append(chr(int(m.group(1), 16)) if m else p.replace("▁", " "))
        return out


# ---------------------------------------------------------------- token-level edits of a recorded prompt

def to_thinking_on(v: Vocab, p: list[int]) -> list[int]:
    """The thinking-on rendering of a thinking-off prompt. The template adds '<|think|>\\n' right after the system
    turn's header, and with thinking off it prefills an empty thought after a user turn (not after a tool response).
    edit_mode "text" makes the same edit on the decoded prompt and encodes it again, for a tokenizer that merges the
    newline with what follows it (prep decides, on probes rendered by the template)."""
    if v.edit_mode == "text":
        text = v.dec(p)
        head = "<|turn>system\n"
        at = text.find(head, 0, 64)
        if at < 0:
            raise ValueError("the prompt does not start with the system turn")
        text = text[:at + len(head)] + "<|think|>\n" + text[at + len(head):]
        tail = "<|channel>thought\n<channel|>"
        return v.enc(text[:-len(tail)] if text.endswith(tail) else text)
    h = find(p, v.sys_header, 0, len(v.sys_header) + 2)
    if h is None:
        raise ValueError("the prompt does not start with the system turn")
    at = h + len(v.sys_header)
    q = p[:at] + v.think_insert + p[at:]
    if q[-len(v.empty_thought):] == v.empty_thought:
        q = q[:-len(v.empty_thought)]
    return q


def with_note(v: Vocab, p: list[int], note: str) -> list[int] | None:
    """The prompt with one more key, "note", at the end of the last tool response's JSON, where the harness puts its
    budget_warning. That JSON is a string, response:NAME{value:<|"|>{...}<|"|>}<tool_response|>; only the token that
    holds its closing brace is re-encoded. None if the last turn is no such tool response."""
    r = rfind(p, v.resp_close)
    if r is None or r < len(p) - 3:
        return None
    q = rfind(p, v.quote, 0, r)
    if q is None or q < 1:
        return None
    last = v.dec(p[q - 1:q])
    t = last.rstrip()
    if not t.endswith("}"):
        return None
    new = v.enc(t[:-1] + ", " + json.dumps("note") + ": " + json.dumps(note) + "}")
    return p[:q - 1] + new + p[q:]


def call_blocks(v: Vocab, p: list[int]) -> list[tuple[int, int, tuple]]:
    """(start, end, call tokens) of every <|tool_call>...<tool_call|><|tool_response>...<tool_response|> block."""
    out, i = [], 0
    while True:
        a = find(p, [v.call_open], i)
        if a is None:
            return out
        b = find(p, [v.call_close], a)
        if b is None:
            return out
        if b + 1 < len(p) and p[b + 1] == v.resp_open:
            c = find(p, [v.resp_close], b)
            if c is None:
                return out
            out.append((a, c + 1, tuple(p[a:b + 1])))
            i = c + 1
        else:
            i = b + 1


def dedup(v: Vocab, p: list[int]) -> tuple[list[int], int] | None:
    """The prompt without the earlier copies of its last call and their results: one copy stays, the latest.
    None if the last call has no earlier copy in the context."""
    bl = call_blocks(v, p)
    if len(bl) < 2:
        return None
    last = bl[-1][2]
    drop = [(s, e) for s, e, call in bl[:-1] if call == last]
    if not drop:
        return None
    q, prev = [], 0
    for s, e in drop:
        q += p[prev:s]
        prev = e
    return q + p[prev:], len(drop)


def with_system_text(v: Vocab, p: list[int], text: str) -> list[int] | None:
    """The prompt with text at the end of the system prompt, just before the first tool declaration."""
    a = find(p, [v.tool_open], 0)
    return None if a is None else p[:a] + v.enc("\n\n" + text) + p[a:]


# ---------------------------------------------------------------- a completion's tool call, token by token

def value_role(key: str) -> str:
    k = key.strip().strip('"')
    if k in ("filepath", "file_path", "path"):
        return "path"
    if k in ("old_string", "new_string", "content"):
        return "code"
    return "command" if k == "command" else "value"


def token_roles(v: Vocab, ids: list[int]) -> list[str]:
    """One role per completion token, from Gemma 4's call syntax
    <|tool_call>call:NAME{key:<|"|>string<|"|>,key:bare}<tool_call|>: think (a thought channel, never scored),
    struct (syntax and turn ends), tool (the name), argname, a value's role (command, path, code, value), text."""
    roles, state, key, depth, colon = [], "out", "", 0, False
    for tid, piece in zip(ids, v.pieces(ids)):
        if state == "think":
            roles.append("think")
            state = "out" if tid == v.ch_close else state
            continue
        if tid == v.ch_open:
            roles.append("think")
            state = "think"
            continue
        if tid == v.call_open:
            roles.append("struct")
            state, colon = "head", False
            continue
        if tid == v.call_close:
            roles.append("struct")
            state = "out"
            continue
        if tid == v.quote:
            roles.append("struct")
            state = {"value": "str", "str": "after"}.get(state, state)
            continue
        if state == "out" or tid in v.structural:
            roles.append("text" if state == "out" and piece.strip() and tid not in v.structural else "struct")
            continue
        role = None
        for ch in piece:
            if state == "head":
                if not colon:
                    r, colon = "struct", ch == ":"
                elif ch == "{":
                    r, state, key = "struct", "key", ""
                else:
                    r = "tool"
            elif state == "key":
                if ch == ":":
                    r, state, depth = "struct", "value", 0
                elif ch == "}":
                    r, state = "struct", "end"
                elif ch in ", ":
                    r = "struct"
                else:
                    r, key = "argname", key + ch
            elif state == "value":   # a bare value: a number, true/false, or code a malformed call left outside <|"|>
                if ch in ",}" and depth == 0:
                    r = "struct"
                    state, key = ("key", "") if ch == "," else ("end", key)
                else:
                    depth += (ch in "{[") - (ch in "}]")
                    r = value_role(key)
            elif state == "str":
                r = value_role(key)
            elif state == "after":
                if ch == ",":
                    r, state, key = "struct", "key", ""
                else:
                    r, state = "struct", ("end" if ch == "}" else state)
            else:
                r = "struct"
            if role is None and not (r == "struct" and ch.isspace()):
                role = r
        roles.append(role or "struct")
    return roles


def parse_call(text: str) -> tuple[str, dict] | None:
    """(name, arguments) of the first tool call in a decoded completion, as vLLM's gemma4 parser reads it; None if
    there is no call."""
    m = text.find("<|tool_call>call:")
    if m < 0:
        return None
    s = text[m + len("<|tool_call>call:"):]
    e = s.find("<tool_call|>")
    s = s if e < 0 else s[:e]
    b = s.find("{")
    if b < 0:
        return s.strip(), {}
    name, body, args, i = s[:b].strip(), s[b + 1:], {}, 0
    while i < len(body):
        if body[i] in " ,":
            i += 1
            continue
        if body[i] == "}":
            break
        j = body.find(":", i)
        if j < 0:
            break
        k, i = body[i:j].strip(), j + 1
        if body.startswith(Q, i):
            z = body.find(Q, i + len(Q))
            val, i = (body[i + len(Q):], len(body)) if z < 0 else (body[i + len(Q):z], z + len(Q))
        else:
            z, depth = i, 0
            while z < len(body):
                ch = body[z]
                if ch in "{[":
                    depth += 1
                elif ch in "}]":
                    if depth == 0:
                        break
                    depth -= 1
                elif ch == "," and depth == 0:
                    break
                z += 1
            raw, i = body[i:z].strip(), z
            try:
                val = json.loads(raw)
            except ValueError:
                val = raw
        args[k] = val
    return name, args


def call_key(fn, args) -> str:
    return json.dumps([fn, args], sort_keys=True, default=str)


def malformed(fn, args) -> bool:
    req, opt = tr.TOOL_ARGS.get(fn, (set(), set()))
    return fn in tr.TOOL_ARGS and bool((set(args) - req - opt) or (req - set(args)))


def call_span(v: Vocab, tail: list[int]) -> tuple[int, int] | None:
    a = find(tail, [v.call_open])
    if a is None:
        return None
    b = find(tail, [v.call_close], a)
    return a, (len(tail) - 1 if b is None else b)


# ---------------------------------------------------------------- traces: labels and alignment

def result_class(fn, d) -> str:
    """What the model saw: for run_command, ok_out, ok_empty, err_silent (a grep without a match: exit 1, nothing
    printed), err_out; for the others ok, ok_trunc (a cut read_file) or the error type."""
    if not isinstance(d, dict):
        return "nonjson"
    if "error" in d and "status" not in d:
        return "adk_arg_error"   # ADK's own argument check: a required argument missing
    stt = d.get("status")
    if fn == "run_command":
        if stt == "ok":
            out, err = (d.get("stdout") or "").strip(), (d.get("stderr") or "").strip()
            return "ok_out" if out else ("ok_stderr_only" if err else "ok_empty")
        if d.get("error_type") == "CommandError":
            det = d.get("details") or {}
            msg, out = (d.get("error_message") or "").strip(), (det.get("stdout") or "").strip()
            return "err_silent" if not msg and not out else "err_out"
        return f"err_{d.get('error_type')}"
    if stt == "ok":
        return "ok_trunc" if fn == "read_file" and d.get("is_truncated") else "ok"
    return f"err_{d.get('error_type')}"


def main_agent(trace: dict) -> str:
    authors = Counter((s.get("extra") or {}).get("author") for s in trace["steps"] if s.get("source") == "agent")
    return "swe_agent" if authors.get("swe_agent") else "swe_baseline_agent"   # budgetfit's agent is swe_agent


def agent_steps(trace: dict, agent: str) -> list[dict]:
    return [s for s in trace["steps"] if s.get("source") == "agent" and (s.get("extra") or {}).get("author") in (None, agent)]


def call_class(cs: list[dict], i: int) -> str:
    c = cs[i]
    if i == 0:
        return "first"
    if c["malformed"]:
        return "malformed"
    if c["stale"]:
        if c["same_as_prev"]:
            return "copy_2nd" if c["run_k"] == 2 else "copy_3rd_plus"
        return "stale_other" if cs[i - 1]["bad"] else "return_to_earlier"
    p = cs[i - 1]
    if p["fn"] == "run_command" and p["rc"] in EMPTY:
        return "after_miss"
    if p["bad"]:
        return "after_bad"
    return "after_error" if p["err"] else "routine"


def label_calls(trace: dict, agent: str) -> list[dict]:
    """trigger_replay.trace_calls' calls, each with what Gate 1 adds: its key, what it returned, its place in a run of
    identical calls, its class, and the keys that would be stale repeats at this point."""
    tr.MAIN_AGENT = agent
    base = tr.trace_calls(trace)
    steps = [s for s in agent_steps(trace, agent) if s.get("tool_calls")]
    assert len(steps) == len(base), (len(steps), len(base))
    out, last_seen, n_edits = [], {}, 0
    for i, (s, c) in enumerate(zip(steps, base)):
        tc = s["tool_calls"][-1]
        fn, args = tc.get("function_name"), tc.get("arguments") or {}
        key = call_key(fn, args)
        try:
            d = json.loads((s.get("observation") or {}).get("content") or "")
        except ValueError:
            d = None
        c = dict(c, step=s, i=i, fn=fn, args=args, key=key, rc=result_class(fn, d), obs=d,
                 stale_keys=sorted(k for k, n in last_seen.items() if n == n_edits))
        c["bad"] = tr.bad(c)
        last_seen[key] = n_edits
        if fn in ("edit_file", "write_file") and isinstance(d, dict) and d.get("status") == "ok":
            n_edits += 1
        out.append(c)
    for i, c in enumerate(out):
        c["same_as_prev"] = i > 0 and c["key"] == out[i - 1]["key"]
        c["run_k"] = out[i - 1]["run_k"] + 1 if c["same_as_prev"] else 1
        c["prev_same"] = next((k for k in range(i - 1, -1, -1) if out[k]["key"] == c["key"]), None)
    for i, c in enumerate(out):
        c["cls"] = call_class(out, i)
    return out


def usage(x: dict) -> tuple:
    m = x.get("metrics") or x.get("usage") or {}
    return m.get("prompt_tokens"), m.get("completion_tokens")


def align(steps: list[dict], recs: list[dict]) -> list[dict | None]:
    """For each trace step in order, the calls.jsonl record with the same token counts (None if none follows). A
    task's records also hold its sub-agents' calls and ADK's compaction calls, which match no step."""
    out, j = [], 0
    for s in steps:
        u, k = usage(s), j
        while k < len(recs) and usage(recs[k]) != u:
            k += 1
        if k < len(recs) and u != (None, None):
            out.append(recs[k])
            j = k + 1
        else:
            out.append(None)
    return out


def load_records(shard: Path) -> dict[str, list[dict]]:
    """calls.jsonl's base-model calls that offered tools, per task, without their prompts (fetched later by line)."""
    recs = defaultdict(list)
    path = shard / "calls.jsonl"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            c = json.loads(line)
            if not c.get("n_tools") or c.get("model") != SERVED or not c.get("prompt_ids"):
                continue
            recs[c["task"]].append({"i": c["i"], "usage": c.get("usage") or {}, "finish": c.get("finish"),
                                    "completion_ids": c.get("completion_ids") or [], "line": n})
    return {t: sorted(cs, key=lambda c: c["i"]) for t, cs in recs.items()}


def fetch_prompts(shard: Path, lines: dict[int, str]) -> dict[str, dict]:
    """key -> {prompt_ids, completion_ids} for the wanted calls.jsonl lines {line: key}."""
    out = {}
    with (shard / "calls.jsonl").open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n in lines:
                c = json.loads(line)
                out[lines[n]] = {"key": lines[n], "prompt_ids": c["prompt_ids"], "completion_ids": c["completion_ids"]}
    return out


def trim(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[:n] + "..."


def result_summary(obs) -> str:
    if not isinstance(obs, dict):
        return "an unreadable result"
    if obs.get("status") == "ok":
        out = obs.get("stdout") or obs.get("content") or obs.get("diff") or ""
        return f"ok: {trim(out, 160)!r}" if str(out).strip() else "ok, with no output"
    err = obs.get("error_message") or obs.get("error") or (obs.get("details") or {}).get("stderr") or ""
    return f"an error: {trim(err, 160)!r}" if str(err).strip() else "an error, with no output"


def hint_for(cs: list[dict], i: int) -> str | None:
    """T-hint's hindsight note for a flagged call: what the call repeats and what that returned, or the argument
    names it should have used (the opsd_assessment's mechanical diagnosis). None for a clean call."""
    c = cs[i]
    if c["stale"] and c["prev_same"] is not None:
        what = (f"run `{trim(c['args'].get('command', ''), 200)}`" if c["fn"] == "run_command"
                else f"call {c['fn']} with the same arguments")
        return (f"Do not {what} again: it already ran and returned {result_summary(cs[c['prev_same']]['obs'])}. "
                "Running it again returns the same.")
    if c["malformed"]:
        req, opt = tr.TOOL_ARGS.get(c["fn"], (set(), set()))
        return f"Call {c['fn']} with exactly these argument names: {', '.join(sorted(req) + sorted(opt))}."
    return None


def note_for(cs: list[dict], i: int) -> str | None:
    """The "note" context: what a controller could say before the call, from the context alone."""
    if i == 0:
        return None
    p = cs[i - 1]
    if p["fn"] == "run_command" and p["rc"] in EMPTY:
        return MISS_NOTE
    return REPEAT_NOTE if p["same_as_prev"] else None


def patch_loc(patch: str) -> str:
    """Files and functions of a diff, from its file headers and hunk headers (T-loc's hint)."""
    files: dict[str, list[str]] = {}
    cur = None
    for line in patch.splitlines():
        m = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if m:
            cur = m.group(2)
            files.setdefault(cur, [])
            continue
        m = re.match(r"^@@ [^@]* @@\s*(.*)$", line)
        if m and cur is not None:
            f = re.search(r"\b(?:def|class)\s+(\w+)", m.group(1))
            if f and f.group(1) not in files[cur]:
                files[cur].append(f.group(1))
    return "; ".join(f"{p}" + (f" (in {', '.join(fs)})" if fs else "") for p, fs in files.items()) or "unknown"


def pick(cands: dict[str, list[dict]], caps: dict, seed: int) -> list[dict]:
    """Per class, round-robin over tasks: at most caps[cls][0] states per task and caps[cls][1] in all."""
    rng = random.Random(seed)
    out = []
    for cls in CLASSES:
        per_task, total = caps[cls]
        by_task = defaultdict(list)
        for c in cands.get(cls, []):
            by_task[c["task"]].append(c)
        tasks = sorted(by_task)
        rng.shuffle(tasks)
        for t in tasks:
            rng.shuffle(by_task[t])
        taken, k = [], 0
        while len(taken) < total and any(len(by_task[t]) > k for t in tasks) and k < per_task:
            for t in tasks:
                if len(by_task[t]) > k and len(taken) < total:
                    taken.append(by_task[t][k])
            k += 1
        out += taken
    return out


# ---------------------------------------------------------------- prep

PROBE_TOOLS = [{"type": "function", "function": {
    "name": "run_command", "description": "Run a shell command in the repository.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command."}},
                   "required": ["command"]}}}]


def template_checks(tok, v: Vocab) -> list[tuple[str, bool, str]]:
    """The token-level edits against the model's own template, on probe conversations."""
    call = {"id": "c1", "type": "function", "function": {"name": "run_command", "arguments": {"command": "grep -rn foo ."}}}
    res = json.dumps({"status": "ok", "stdout": "a.py:1:foo\n", "stderr": "", "exit_code": 0})
    turn = [{"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "name": "run_command", "content": res}]
    base = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix the bug in a.py."}]
    probes = {"after the task prompt": base, "after a tool response": base + turn, "after two identical calls": base + turn + turn}
    r = {}
    for name, msgs in probes.items():
        for think in (False, True):
            text = tok.apply_chat_template(msgs, tools=PROBE_TOOLS, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=think)
            r[name, think] = v.enc(text)
    out, mode, detail = [], None, []
    for m in ("tokens", "text"):   # token-level if it reproduces the template; else the edit on the decoded text
        v.edit_mode = m
        detail = []
        for name in probes:
            off, on = r[name, False], r[name, True]
            try:
                got = to_thinking_on(v, off)
            except ValueError as e:
                got, why = None, str(e)
            else:
                why = f"edit {v.dec(got[:12])!r}...{v.dec(got[-12:])!r}, template {v.dec(on[:12])!r}...{v.dec(on[-12:])!r}"
            if got != on:
                detail.append(f"{name}: {why}")
        if not detail:
            mode = m
            break
    v.edit_mode = mode or "tokens"
    out.append(("thinking-on edit reproduces the template on all probes", mode is not None,
                f"edit mode {mode}" if mode else "; ".join(detail)))
    first = r["after the task prompt", False]
    out.append(("thinking off prefills an empty thought after the task prompt", first[-len(v.empty_thought):] == v.empty_thought,
                repr(v.dec(first[-12:]))))
    after = r["after a tool response", False]
    out.append(("thinking off leaves the turn open after a tool response", after[-1] == v.resp_close, repr(v.dec(after[-12:]))))
    q = with_note(v, after, "probe note")
    ok = q is not None and json.dumps("note") + ": " + json.dumps("probe note") in v.dec(q) and q[-3:] == after[-3:]
    out.append(("note edit", ok, repr(v.dec(q[-60:])) if q else "not applied"))
    d = dedup(v, r["after two identical calls", False])
    ok = d is not None and d[0] == after and d[1] == 1
    out.append(("dedup edit: two identical calls become one", ok, "" if ok else f"{d and v.dec(d[0][-60:])!r}"))
    s = with_system_text(v, after, "PRIVATE")
    out.append(("system-text edit", s is not None and "You fix bugs.\n\nPRIVATE<|tool>" in v.dec(s), ""))
    roles = token_roles(v, v.enc('<|tool_call>call:run_command{command:<|"|>grep -rn foo .<|"|>}<tool_call|>'))
    out.append(("token roles of a probe call", "tool" in roles and "argname" in roles and "command" in roles,
                " ".join(roles)))
    return out


def onrun_checks(v: Vocab, runs: list[Path], n: int = 300) -> Counter:
    """Facts from a thinking-on run's own recorded calls (base_v30_l40s): is '<|think|>\\n' there right after the
    system header, token for token, and does a reply after a tool response open a thought?"""
    seen = Counter()
    for run in runs:
        for shard in sorted(run.glob("shard_*")):
            recs = [c for c in iter_jsonl(shard / "calls.jsonl")
                    if c.get("n_tools") and c.get("model") == SERVED and c.get("prompt_ids")]
            if not recs:
                continue
            main = Counter(c["n_tools"] for c in recs).most_common(1)[0][0]   # the main agent makes most calls
            for c in recs:
                if c["n_tools"] != main:
                    continue   # a sub-agent's call (the analyzer has its own prompt and tools)
                p, g = c["prompt_ids"], c.get("completion_ids") or []
                seen["thinking-on prompts read"] += 1
                if not v.dec(p[:8]).startswith(("<bos><|turn>system", "<|turn>system")):
                    seen["of those, without a system turn at the start (left out)"] += 1
                    continue
                h = find(p, v.sys_header, 0, len(v.sys_header) + 2)
                at = None if h is None else h + len(v.sys_header)
                if at is not None and p[at:at + len(v.think_insert)] == v.think_insert:
                    seen["with <|think|>\\n right after the system header, as the token-level edit puts it"] += 1
                if p[-1] == v.resp_close:
                    seen["replies after a tool response"] += 1
                    seen["of those, replies that open a thought"] += bool(g) and g[0] == v.ch_open
                if seen["thinking-on prompts read"] >= n:
                    return seen
    return seen


def cmd_prep(a) -> int:
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    v = Vocab(tok)
    a.work.mkdir(parents=True, exist_ok=True)
    checks = template_checks(tok, v)   # also sets v.edit_mode, which generate and score read back from prep.json
    onrun = onrun_checks(v, a.on_runs) if a.on_runs else Counter()
    n_on = onrun["thinking-on prompts read"] - onrun["of those, without a system turn at the start (left out)"]
    if n_on:
        k = onrun["with <|think|>\\n right after the system header, as the token-level edit puts it"]
        checks.append(("a thinking-on run's own prompts have <|think|>\\n where the token-level edit puts it",
                       k == n_on or v.edit_mode == "text", f"{k} of {n_on}"))
    counts, cands, parse_ok, parse_n = Counter(), defaultdict(list), 0, 0
    calls_by_task = {}
    for run in a.runs:
        for shard in sorted(run.glob("shard_*")):
            results = {r["instance_id"]: r for r in iter_jsonl(shard / "task_results.jsonl")}
            recs = load_records(shard)
            for f in sorted((shard / "traces").glob("trace_*.json")):
                tid = f.stem[len("trace_"):]
                trace = json.loads(f.read_text(encoding="utf-8"))
                agent = main_agent(trace)
                cs = label_calls(trace, agent)
                steps = [s for s in agent_steps(trace, agent) if s.get("metrics")]
                matched = dict(zip(map(id, steps), align(steps, recs.get(tid, []))))
                calls_by_task[run.name, shard.name, tid] = cs   # a rerun of a task sits in another shard
                for c in cs:
                    counts["calls"] += 1
                    rec = matched.get(id(c["step"]))
                    if rec is None:
                        counts["calls without a calls.jsonl record"] += 1
                        continue
                    if "length" in (rec.get("finish") or []):
                        counts["calls cut at max_tokens (left out)"] += 1
                        continue
                    comp = rec["completion_ids"]
                    off = 0
                    if comp and comp[0] == v.ch_open:   # the student's own (empty) thought block
                        e = find(comp, [v.ch_close])
                        off = len(comp) if e is None else e + 1
                    tail = comp[off:]
                    span = call_span(v, tail)
                    parsed = parse_call(v.dec(tail))
                    parse_n += 1
                    if parsed and call_key(*parsed) == c["key"]:
                        parse_ok += 1
                    if span is None:
                        counts["calls with no tool call in the completion (left out)"] += 1
                        continue
                    counts["aligned"] += 1
                    cands[c["cls"]].append({
                        "c": c, "task": tid, "run": run.name, "shard": shard, "line": rec["line"], "tail_offset": off,
                        "tail": tail, "span": span, "resolved": bool((results.get(tid) or {}).get("resolved"))})
    caps = dict(CAPS)
    if a.scale != 1.0:
        caps = {k: (p, max(1, round(n * a.scale))) for k, (p, n) in CAPS.items()}
    chosen = pick(cands, caps, a.seed)
    # prompts of the chosen calls, one pass per shard
    wanted = defaultdict(dict)
    for x in chosen:
        x["key"] = f"{x['run']}/{x['shard'].name}/{x['task']}/{x['c']['i']}"
        wanted[x["shard"]][x["line"]] = x["key"]
    prompts = {}
    for shard, lines in wanted.items():
        prompts.update(fetch_prompts(shard, lines))
    priv = {}
    if a.data and (a.data / "tasks.jsonl").is_file():
        tasks = {r["instance_id"]: r for r in iter_jsonl(a.data / "tasks.jsonl")}
        mine = sorted({x["task"] for x in chosen})
        for t in mine:
            patch = (tasks.get(t) or {}).get("patch") or ""
            repo = (tasks.get(t) or {}).get("repo")
            same = [u for u in mine if u != t and (tasks.get(u) or {}).get("repo") == repo and (tasks.get(u) or {}).get("patch")]
            others = same or [u for u in mine if u != t and (tasks.get(u) or {}).get("patch")]
            wrong = next((u for u in others if u > t), others[0] if others else None)
            if patch:
                priv[t] = {"task": t, "patch": patch[:PATCH_CHARS], "loc": patch_loc(patch),
                           "wrong_task": wrong, "wrong": ((tasks.get(wrong) or {}).get("patch") or "")[:PATCH_CHARS]}
    states, edit_counts, roundtrip = [], Counter(), Counter()
    for x in chosen:
        c, cs = x["c"], calls_by_task[x["run"], x["shard"].name, x["task"]]
        pr = prompts.get(x["key"])
        if pr is None:
            edit_counts["prompt not found again in calls.jsonl"] += 1
            continue
        p = pr["prompt_ids"]
        try:
            to_thinking_on(v, p)
            edit_counts["thinking-on edit applies"] += 1
        except ValueError:
            edit_counts["thinking-on edit fails (state left out)"] += 1
            continue
        hint, note = hint_for(cs, c["i"]), note_for(cs, c["i"])
        if hint and with_note(v, p, hint) is None:
            hint = None
            edit_counts["T-hint: no tool response to hold the note"] += 1
        if note and with_note(v, p, note) is None:
            note = None
            edit_counts["note: no tool response to hold the note"] += 1
        dd = dedup(v, p) if c["i"] > 0 and cs[c["i"] - 1]["same_as_prev"] else None
        edit_counts["T-hint applies"] += hint is not None
        edit_counts["note applies"] += note is not None
        edit_counts["dedup applies"] += dd is not None
        if sum(roundtrip.values()) < 60:
            roundtrip["re-encodes to the same ids" if v.enc(v.dec(p)) == p else "re-encodes differently"] += 1
        prev = cs[c["i"] - 1] if c["i"] > 0 else None
        states.append({
            "key": x["key"], "run": x["run"], "shard": x["shard"].name, "task": x["task"], "resolved": x["resolved"],
            "i": c["i"], "n_calls": len(cs), "cls": c["cls"], "fn": c["fn"], "call_key": c["key"],
            "stale": c["stale"], "malformed": c["malformed"], "run_k": c["run_k"], "rc": c["rc"],
            "prev_rc": prev["rc"] if prev else None, "prev_fn": prev["fn"] if prev else None,
            "tail_offset": x["tail_offset"], "roles": token_roles(v, x["tail"]), "span": list(x["span"]),
            "call_text": v.dec(x["tail"][x["span"][0]:x["span"][1] + 1]), "stale_keys": c["stale_keys"],
            "hint": hint, "note": note, "dedup_removed": dd[1] if dd else 0, "n_prompt": len(p)})
        prompts[x["key"]]["kept"] = True
    write_jsonl(a.work / "states.jsonl", states)
    write_jsonl(a.work / "prompts.jsonl", ({k: r[k] for k in ("key", "prompt_ids", "completion_ids")}
                                           for r in prompts.values() if r.get("kept")))
    write_jsonl(a.work / "priv.jsonl", priv.values())
    all_counts = {cls: len(cands.get(cls, [])) for cls in CLASSES}
    kept_counts = Counter(s["cls"] for s in states)
    ok = all(c[1] for c in checks)
    summary = {"runs": [str(r) for r in a.runs], "edit_mode": v.edit_mode, "on_run": dict(onrun),
               "counts": dict(counts), "candidates": all_counts,
               "picked": dict(kept_counts), "tasks": len({(s["run"], s["task"]) for s in states}),
               "parser_agrees_with_trace": [parse_ok, parse_n], "edits": dict(edit_counts), "roundtrip": dict(roundtrip),
               "priv_tasks": len(priv), "checks": [{"check": n, "pass": p, "detail": d} for n, p, d in checks],
               "secs": round(time.time() - t0, 1)}
    (a.work / "prep.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    lines = [f"# Gate 1 prep: {a.work.name}", "", f"Runs: {', '.join(r.name for r in a.runs)}. {summary['secs']} s.", "",
             "## Template checks (the token-level edits against the model's chat template)", ""]
    lines += [f"- {'PASS' if p else 'FAIL'} {n}{': ' + d if d else ''}" for n, p, d in checks]
    lines += [f"- the edits work on the {'decoded text' if v.edit_mode == 'text' else 'token ids'} (edit mode {v.edit_mode})"]
    if onrun:
        lines += ["", "A thinking-on run's own calls (" + ", ".join(r.name for r in a.on_runs) + "):"]
        lines += [f"- {k}: {n}" for k, n in onrun.items()]
    lines += ["", "## Calls", ""] + [f"- {k}: {n}" for k, n in counts.items()]
    lines += [f"- parser agrees with the trace's call: {parse_ok} of {parse_n}", "",
              "| class | calls | picked |", "|---|---|---|"]
    lines += [f"| {cls} | {all_counts[cls]} | {kept_counts.get(cls, 0)} |" for cls in CLASSES]
    lines += ["", f"States: {len(states)} in {summary['tasks']} tasks; patched teachers: {len(priv)} tasks with a gold patch.",
              "", "## Edits on the picked prompts", ""] + [f"- {k}: {n}" for k, n in edit_counts.items()]
    lines += [f"- decode then encode, on {sum(roundtrip.values())} prompts: {dict(roundtrip)}", "", "## Examples", ""]
    rng = random.Random(a.seed)
    for cls in ("copy_3rd_plus", "after_miss", "routine"):
        ex = [s for s in states if s["cls"] == cls]
        if ex:
            s = rng.choice(ex)
            lines += [f"- {cls}, call {s['i']} of {s['n_calls']}: `{trim(s['call_text'], 160)}`",
                      f"  - roles: {' '.join(f'{r}x{n}' for r, n in Counter(s['roles']).most_common())}",
                      f"  - note: {s['note'] or '-'}; hint: {trim(s['hint'] or '-', 120)}; dedup removes {s['dedup_removed']}"]
    (a.work / "prep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not ok:
        print("FAIL: a template check failed; the GPU steps would score wrong prompts")
        return 1
    if not states:
        print("FAIL: no states picked")
        return 1
    return 0


# ---------------------------------------------------------------- generate (vLLM)

def seed_of(key: str) -> int:
    """One seed per state, from its whole key. Until 2026-10-12 it came from the key's last 8 bytes modulo 2^31, which
    keeps only 4 of them: the first run's 581 states had 75 seeds, so T-think's and own's samples there share seeds
    within a task."""
    return zlib.crc32(key.encode()) % (2 ** 31)


def load_work(work: Path):
    states = list(iter_jsonl(work / "states.jsonl"))
    prompts = {r["key"]: r for r in iter_jsonl(work / "prompts.jsonl")}
    return states, prompts


def vocab_for(a) -> Vocab:
    """The tokenizer, with the edit mode prep chose (prep.json)."""
    from transformers import AutoTokenizer
    meta = json.loads((a.work / "prep.json").read_text(encoding="utf-8")) if (a.work / "prep.json").is_file() else {}
    return Vocab(AutoTokenizer.from_pretrained(str(a.model), local_files_only=True), meta.get("edit_mode", "tokens"))


def make_llm(a, prefix_caching: bool):
    from vllm import LLM
    # Eager and text-only, as the main track's baseline/loss_check.py runs vLLM offline. For prompt logprobs, the
    # prefill chunk bounds the log-softmax over the 262k vocabulary that vLLM's memory profile doesn't budget for.
    return LLM(model=str(a.model), max_model_len=a.max_len, gpu_memory_utilization=a.gpu_mem, enforce_eager=True,
               enable_prefix_caching=prefix_caching, max_num_batched_tokens=a.chunk_tokens, max_num_seqs=a.max_seqs,
               max_logprobs=a.topk, seed=0, limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0})


def ended_with(o, stop: list[int]) -> list[int]:
    ids = list(o.token_ids)
    if o.finish_reason == "stop" and o.stop_reason in stop and (not ids or ids[-1] != o.stop_reason):
        ids.append(o.stop_reason)   # vLLM may leave the stop token out of token_ids
    return ids


def judge_call(v: Vocab, s: dict, call: list[int]) -> dict:
    """A sampled call at a state: the student's call exactly, a stale repeat in that context, malformed."""
    parsed = parse_call(v.dec(call))
    span = call_span(v, call)
    mine = v.dec(call[span[0]:span[1] + 1]) if span else ""
    return {"call_text": mine, "has_call": parsed is not None, "same_as_student": bool(mine) and mine == s["call_text"],
            "stale": parsed is not None and call_key(*parsed) in set(s["stale_keys"]),
            "malformed": parsed is not None and malformed(*parsed),
            "same_tool": parsed is not None and parsed[0] == s["fn"]}


def cmd_generate(a) -> int:
    """A thought at every state, then the model's own next call after it. T-think: the thinking-on prompt, and the
    model decides whether to think. T-open: the same prompt with the thought channel already open, and the model can't
    close it before --min-thought tokens. vLLM's min_tokens holds back every stop token, <channel|> among them."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    v = vocab_for(a)
    states, prompts = load_work(a.work)
    forced = a.context == "T-open"
    least = a.min_thought if forced else 0
    pre = v.open_thought if forced else []   # what the thought starts with before the model writes anything
    out_path = a.work / GEN[a.context]
    done = {r["key"] for r in iter_jsonl(out_path)}
    todo = sorted((s for s in states if s["key"] not in done), key=lambda s: (s["run"], s["task"], s["i"]))
    print(f"generate {a.context}: {len(todo)} states to do, {len(done)} done; budget {a.budget}"
          f"{f', a thought of at least {least} tokens' if forced else ''}, T={a.temperature}", flush=True)
    if not todo:
        return 0
    llm = make_llm(a, prefix_caching=True)
    t0 = time.time()
    for k in range(0, len(todo), a.chunk):
        part = todo[k:k + a.chunk]
        base = [to_thinking_on(v, prompts[s["key"]]["prompt_ids"]) for s in part]
        sp = lambda s, n, stop, least=0: SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                                        max_tokens=n, min_tokens=least, stop_token_ids=stop,
                                                        seed=seed_of(s["key"]), detokenize=False)
        stop1 = [v.ch_close, v.call_close]
        outs = llm.generate([TokensPrompt(prompt_token_ids=b + pre) for b in base],
                            [sp(s, a.budget - len(pre), stop1, least) for s in part], use_tqdm=False)
        rows, second = [], []
        for s, b, o in zip(part, base, outs):
            g = o.outputs[0]
            ids = pre + ended_with(g, stop1)
            row = {"key": s["key"], "thought": [], "truncated": False, "call": [], "finish": g.finish_reason}
            if ids and ids[0] == v.ch_open:
                # how the thought ended: closed by the model, at the budget, or at a call written inside it
                row["thought_end"] = ("closed" if ids[-1] == v.ch_close else "call" if ids[-1] == v.call_close
                                      else "budget")
                if ids[-1] == v.ch_close:
                    row["thought"] = ids
                else:   # cut inside the thought: close it, as vLLM's thinking_token_budget does
                    row["thought"], row["truncated"] = ids + v.close_thought, True
                if forced:
                    row.update(min_thought=least, budget=a.budget, thought_text=v.dec(row["thought"]))
                second.append((s, b + row["thought"], row))
            else:
                row["call"] = ids   # no thought: the reply went straight to the call
            rows.append(row)
        if second:
            stop2 = [v.call_close]
            outs2 = llm.generate([TokensPrompt(prompt_token_ids=c) for _, c, _ in second],
                                 [sp(s, a.call_tokens, stop2) for s, _, _ in second], use_tqdm=False)
            for (_, _, row), o in zip(second, outs2):
                row["call"], row["call_finish"] = ended_with(o.outputs[0], stop2), o.outputs[0].finish_reason
        for s, row in zip(part, rows):
            row.update({"thought_len": len(row["thought"]), **judge_call(v, s, row["call"])})
        write_jsonl(out_path, rows, "a")
        print(f"  {k + len(part)}/{len(todo)} states, {time.time() - t0:.0f} s", flush=True)
    return 0


def cmd_own(a) -> int:
    """The model's own call after each OWN context's prompt, where the context applies. The seed is the state's, as
    for T-think's call, so block and C-none draw the same random numbers at a state."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    v = vocab_for(a)
    states, prompts = load_work(a.work)
    bad = [c for c in a.contexts if c not in OWN]
    if bad:
        raise SystemExit(f"own samples after {', '.join(OWN)} only, not {bad}")
    llm = None
    for name in a.contexts:
        out_path = a.work / f"own_{name}.jsonl"
        done = {r["key"] for r in iter_jsonl(out_path)}
        todo, skipped = [], Counter()
        for s in sorted(states, key=lambda s: (s["run"], s["task"], s["i"])):   # neighbours share prefixes
            if s["key"] in done:
                continue
            built = build_variant(v, name, s, prompts[s["key"]], None, None)
            if built is None:
                skipped["does not apply"] += 1
                continue
            if built[1] + a.call_tokens > a.max_len:
                skipped["longer than --max-len"] += 1
                continue
            todo.append((s, built[0][:built[1]]))
        print(f"own {name}: {len(todo)} to do, {len(done)} done, skipped {dict(skipped)}; T={a.temperature}",
              flush=True)
        if not todo:
            continue
        if llm is None:
            llm = make_llm(a, prefix_caching=True)
        t0 = time.time()
        stop = [v.call_close]
        for k in range(0, len(todo), a.chunk):
            part = todo[k:k + a.chunk]
            sps = [SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                  max_tokens=a.call_tokens, stop_token_ids=stop, seed=seed_of(s["key"]),
                                  detokenize=False) for s, _ in part]
            outs = llm.generate([TokensPrompt(prompt_token_ids=c) for _, c in part], sps, use_tqdm=False)
            rows = []
            for (s, _), o in zip(part, outs):
                g = o.outputs[0]
                call = ended_with(g, stop)
                rows.append({"key": s["key"], "call": call, "finish": g.finish_reason, **judge_call(v, s, call)})
            write_jsonl(out_path, rows, "a")
            print(f"  {name}: {k + len(part)}/{len(todo)} states, {time.time() - t0:.0f} s", flush=True)
    return 0


# ---------------------------------------------------------------- score (vLLM)

def build_variant(v: Vocab, name: str, s: dict, pr: dict, gen: dict | None, priv: dict | None):
    """(full token sequence, index where the student's tail starts) of one context, or None where it doesn't apply.
    The tail is the student's completion without its own empty thought; thinking-off contexts keep that block where
    the student wrote it, thinking-on ones put an empty thought (or T-think's) in its place, and block forces one
    where the student wrote none."""
    p, comp = pr["prompt_ids"], pr["completion_ids"]
    head, tail = comp[:s["tail_offset"]], comp[s["tail_offset"]:]
    if name == "S":
        ctx = p + head
    elif name in ("T-hint", "note"):
        text = s["hint"] if name == "T-hint" else s["note"]
        q = with_note(v, p, text) if text else None
        if q is None:
            return None
        ctx = q + head
    elif name == "dedup":
        d = dedup(v, p) if s["dedup_removed"] else None
        if d is None:
            return None
        ctx = d[0] + head
    elif name == "block":
        if head or p[-len(v.empty_thought):] == v.empty_thought:
            return None   # the student wrote its own, or the template prefilled one (after the task prompt): S itself
        ctx = p + v.empty_thought
    elif name in PATCHED:
        field = PATCHED[name]
        if not priv or not priv.get(field):
            return None
        text = PRIV_TEXT[field].format(text=priv[field])
        base = to_thinking_on(v, p) if name in THINKING_ON else p
        q = with_system_text(v, base, text)
        if q is None:
            return None
        ctx = q + (v.empty_thought if name in THINKING_ON else head)
    elif name == "C-none":
        ctx = to_thinking_on(v, p) + v.empty_thought
    elif name in GEN:   # T-think, T-open: the thought that context wrote (none: T-think went straight to the call)
        if gen is None:
            return None
        ctx = to_thinking_on(v, p) + gen["thought"]
    else:
        raise ValueError(name)
    return ctx + tail, len(ctx)


def sampled_logprob(top_ids, top_lps, actual, temperature, top_p, top_k) -> float:
    """log p of the actual token under temperature, then top-k and top-p, from the top logprobs (vLLM's order:
    temperature first, then top-k/top-p). Tokens outside the top logprobs count as 0: at T=0.2 they hardly ever are
    sampled; at T=1 this slightly overstates the kept tokens' share."""
    ids, lps = top_ids[:top_k], top_lps[:top_k]
    if actual not in ids:
        return -math.inf
    x = [l / temperature for l in lps]
    m = max(x)
    w = [math.exp(z - m) for z in x]
    total = sum(w)
    q = sorted(((wi / total, i) for i, wi in zip(ids, w)), reverse=True)
    keep, cum = {}, 0.0
    for qi, i in q:
        keep[i] = qi
        cum += qi
        if cum >= top_p:
            break
    return math.log(keep[actual] / sum(keep.values())) if actual in keep else -math.inf


def kl_topk(t_ids, t_lps, s_ids, s_lps, s_actual_id, s_actual_lp) -> float:
    """Forward KL(teacher || student) over the teacher's top tokens. A token outside the student's top list gets
    the student's lowest listed logprob, which understates the KL."""
    sm = dict(zip(s_ids, s_lps))
    sm.setdefault(s_actual_id, s_actual_lp)
    floor = min(sm.values())
    return sum(math.exp(l) * (l - sm.get(i, floor)) for i, l in zip(t_ids, t_lps))


def cmd_score(a) -> int:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    v = vocab_for(a)
    states, prompts = load_work(a.work)
    gens = {n: {r["key"]: r for r in iter_jsonl(a.work / f)} for n, f in GEN.items()}
    priv = {r["task"]: r for r in iter_jsonl(a.work / "priv.jsonl")}
    names = ["S"] + [c for c in a.contexts if c != "S"]
    bad = [c for c in names if c not in CONTEXTS]
    if bad:
        raise SystemExit(f"unknown contexts {bad}; known: {', '.join(CONTEXTS)}")
    llm = None
    sp = None
    sdata = {}
    for name in names:
        path = a.work / f"score_{name}.jsonl"
        done = {r["key"] for r in iter_jsonl(path)}
        if name == "S":
            sdata = {r["key"]: r for r in iter_jsonl(path)}
        jobs, skipped = [], Counter()
        for s in states:
            if s["key"] in done:
                continue
            if name in GEN and s["key"] not in gens[name]:
                skipped[f"no {name} thought yet (run generate --context {name})"] += 1
                continue
            if name != "S" and s["key"] not in sdata:
                skipped["no S score"] += 1
                continue
            built = build_variant(v, name, s, prompts[s["key"]], gens[name].get(s["key"]) if name in GEN else None,
                                  priv.get(s["task"]))
            if built is None:
                skipped["does not apply"] += 1
                continue
            if len(built[0]) + 1 > a.max_len:
                skipped["longer than --max-len"] += 1
                continue
            jobs.append((s, built[0], built[1]))
        print(f"score {name}: {len(jobs)} to do, {len(done)} done, skipped {dict(skipped)}", flush=True)
        if not jobs:
            continue
        if llm is None:
            llm = make_llm(a, prefix_caching=False)
            sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=a.topk, detokenize=False)
        t0, n_tok = time.time(), 0
        for k in range(0, len(jobs), a.batch):
            part = jobs[k:k + a.batch]
            outs = llm.generate([TokensPrompt(prompt_token_ids=f) for _, f, _ in part], sp, use_tqdm=False)
            rows = []
            for (s, full, start), o in zip(part, outs):
                n_tok += len(full)
                lp, tids, tlps = [], [], []
                for t in range(start, len(full)):
                    d = o.prompt_logprobs[t]
                    lp.append(d[full[t]].logprob)
                    top = sorted(((i, e) for i, e in d.items() if e.rank is not None and e.rank <= a.topk),
                                 key=lambda ie: ie[1].rank)[:a.topk]
                    tids.append([i for i, _ in top])
                    tlps.append([e.logprob for _, e in top])
                tail = full[start:]
                sa, sb = s["span"]
                row = {"key": s["key"], "n": len(tail),
                       "logp_call": sum(lp[sa:sb + 1]),
                       "logp_call_run": sum(sampled_logprob(tids[j], tlps[j], tail[j], a.temperature, a.top_p, a.top_k)
                                            for j in range(sa, sb + 1)),
                       "logp_call_t1": sum(sampled_logprob(tids[j], tlps[j], tail[j], 1.0, a.top_p, a.top_k)
                                           for j in range(sa, sb + 1))}
                if name == "S":
                    row.update({"lp": [round(x, 4) for x in lp], "top_ids": tids,
                                "top_lp": [[round(x, 4) for x in r] for r in tlps]})
                    sdata[s["key"]] = row
                else:
                    sr = sdata[s["key"]]
                    row["lp"] = [round(x, 4) for x in lp]
                    row["lr"] = [round(x - y, 4) for x, y in zip(lp, sr["lp"])]
                    row["kl"] = [round(kl_topk(tids[j], tlps[j], sr["top_ids"][j], sr["top_lp"][j], tail[j], sr["lp"][j]), 4)
                                 for j in range(len(tail))]
                for f in ("logp_call", "logp_call_run", "logp_call_t1"):
                    row[f] = round(row[f], 4) if math.isfinite(row[f]) else None
                rows.append(row)
            write_jsonl(path, rows, "a")
            el = time.time() - t0
            print(f"  {name}: {k + len(part)}/{len(jobs)}, {el:.0f} s, {n_tok / max(el, 1e-9):.0f} prompt tok/s", flush=True)
    return 0


# ---------------------------------------------------------------- report (stdlib)

def mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def pexp(x) -> float:
    return 0.0 if x is None else math.exp(x)


def bootstrap_ratio(per_task: dict, reps: int, seed: int):
    """Ratio of per-token means, failure over routine, and its 95% CI over tasks resampled with replacement.
    per_task: task -> (failure sum, failure n, routine sum, routine n)."""
    def ratio(ts):
        fs = sum(per_task[t][0] for t in ts)
        fn = sum(per_task[t][1] for t in ts)
        rs = sum(per_task[t][2] for t in ts)
        rn = sum(per_task[t][3] for t in ts)
        return (fs / fn) / (rs / rn) if fn and rn and rs > 0 else None
    tasks = sorted(per_task)
    point = ratio(tasks)
    rng = random.Random(seed)
    boot = sorted(r for r in (ratio([rng.choice(tasks) for _ in tasks]) for _ in range(reps)) if r is not None)
    if not boot or point is None:
        return point, None, None
    return point, boot[int(0.025 * (len(boot) - 1))], boot[int(0.975 * (len(boot) - 1))]


def kl_per_task(states: dict, rows: dict, keys, per_call: bool = False) -> dict:
    """The gate's sums over the given states: task -> [failure KL, failure n, routine KL, routine n], over the call
    tokens; with per_call, over calls instead, each at its largest token KL (a per-token mean favours long calls)."""
    per_task = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for k in keys:
        s = states[k]
        kl = [x for role, x in zip(s["roles"], rows[k]["kl"]) if role in CALL_ROLES]
        at = 0 if s["cls"] in STUCK else 2 if s["cls"] == "routine" else None
        if at is None or not kl:
            continue
        kl = [max(kl)] if per_call else kl
        per_task[s["task"]][at] += sum(kl)
        per_task[s["task"]][at + 1] += len(kl)
    return dict(per_task)


def own_shares(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    d = {f: sum(r[f] for r in rows) / len(rows) for f in ("same_as_student", "stale", "malformed")}
    return dict(d, no_call=sum(not r["has_call"] for r in rows) / len(rows), n=len(rows))


def report_block(a, states: dict, scores: dict, gen: dict) -> tuple[list[str], dict]:
    """Section 3: the empty thought forced with thinking off (block) and on (C-none), where block applies; then the
    same states with thinking on and no thought (T-think where it went straight to the call)."""
    own = {n: {r["key"]: r for r in iter_jsonl(a.work / f"own_{n}.jsonl") if r["key"] in states} for n in OWN}
    forced = [k for k in scores["block"] if k in scores["S"] and k in scores["C-none"]]
    pc = lambda n, k: pexp(scores[n][k]["logp_call_run"])
    out = {"states": len(forced), "by_class": {}, "t_think_no_thought": {}, "kl": {}}
    lines = ["## 3. The forced empty thought: thinking off and on", "",
             "block is the recorded prompt (thinking off) with the empty thought <|channel>thought\\n<channel|> forced "
             "where the student went straight to the call. Where the student wrote that block itself, or the prompt "
             "ends with the template's own (after the task prompt), block is the recorded prompt, so it isn't scored "
             "there. C-none forces the same block after the thinking-on prompt. p(call) at the run's sampling, as in "
             "section 2, on the states block applies to. Own call: one sample after the context, with the state's seed "
             "in both contexts; same / stale / malformed as for T-think's.", "",
             "| class | states | of the class | S | block | C-none | block's own call: same / stale / malformed "
             "| C-none's own call: same / stale / malformed |", "|---|---|---|---|---|---|---|---|"]
    for cls in CLASSES:
        n_cls = sum(1 for k, s in states.items() if s["cls"] == cls and k in scores["S"])
        if not n_cls:
            continue
        keys = [k for k in forced if states[k]["cls"] == cls]
        row = {"states": len(keys), "of_class": len(keys) / n_cls}
        row.update({n: mean(pc(n, k) for k in keys) for n in ("S", "block", "C-none")})
        cells = [fmt(row[n], 2) for n in ("S", "block", "C-none")]
        for n in OWN:
            d = row[f"own_{n}"] = own_shares([own[n][k] for k in keys if k in own[n]])
            cells.append("-" if d is None else
                         f"{d['same_as_student']:.0%} / {d['stale']:.0%} / {d['malformed']:.0%} ({d['n']})")
        out["by_class"][cls] = row
        lines.append(f"| {cls} | {len(keys)} | {row['of_class']:.0%} | " + " | ".join(cells) + " |")
    if "T-think" in scores:
        four = ("S", "block", "T-think", "C-none")
        lines += ["", "The same, on the states where T-think also went straight to the call (with thinking on, no "
                  "thought is the model's own choice there): thinking off or on, without or with the forced block.", "",
                  "| class | states | off: S | off + block: block | on: T-think, no thought | on + block: C-none |",
                  "|---|---|---|---|---|---|"]
        for cls in CLASSES:
            keys = [k for k in forced if states[k]["cls"] == cls and k in gen and not gen[k]["thought"]
                    and k in scores["T-think"]]
            if not keys:
                continue
            row = {"states": len(keys), **{n: mean(pc(n, k) for k in keys) for n in four}}
            out["t_think_no_thought"][cls] = row
            lines.append(f"| {cls} | {len(keys)} | " + " | ".join(fmt(row[n], 2) for n in four) + " |")
    lines += ["", "The gate's KL on the states block applies to: failure classes against routine, with a 95% CI over "
              "tasks, per call token as in the gate, and per call at its largest token KL.", ""]
    for n in ("block", "C-none"):
        out["kl"][n], cells = {}, []
        for unit in ("per token", "per call"):
            per_task = kl_per_task(states, scores[n], forced, per_call=unit == "per call")
            fk = sum(x[0] for x in per_task.values()) / max(sum(x[1] for x in per_task.values()), 1)
            rk = sum(x[2] for x in per_task.values()) / max(sum(x[3] for x in per_task.values()), 1)
            point, lo, hi = bootstrap_ratio(per_task, a.reps, a.seed)
            out["kl"][n][unit] = {"failure_kl": fk, "routine_kl": rk, "ratio": point, "ci": [lo, hi]}
            ci = f" ({fmt(lo, 2)}-{fmt(hi, 2)})" if lo is not None else ""
            cells.append(f"{unit}, failure {fmt(fk)} against routine {fmt(rk)}: ratio {fmt(point, 2)}{ci}")
        lines.append(f"- {n}: " + "; ".join(cells))
    lines.append("")
    return lines, out


LOOPING = ("copy_2nd", "copy_3rd_plus", "return_to_earlier", "stale_other")   # the stuck classes of stale repeats
EMPTY_LEN = 4   # tokens of the empty thought <|channel>thought\n<channel|> (Vocab.empty_thought); longer is a real thought


def escaped(r: dict) -> bool:
    """A sampled call that leaves the loop: a call, neither a stale repeat nor malformed."""
    return r["has_call"] and not r["stale"] and not r["malformed"]


def report_open(a, states: dict, scores: dict, gens: dict) -> tuple[list[str], dict]:
    """Section 4: the forced thought (T-open) against T-think, which chooses where to think, by class; then where
    T-think chose not to think; then on block's states, against the forced empty thought."""
    tt, to = gens["T-think"], gens["T-open"]
    own = {n: {r["key"]: r for r in iter_jsonl(a.work / f"own_{n}.jsonl")} for n in OWN}
    pc = lambda n, k: pexp(scores[n][k]["logp_call_run"]) if k in scores.get(n, {}) else None
    real = lambda k: k in tt and tt[k]["thought_len"] > EMPTY_LEN
    keys_of = lambda cls: [k for k, s in states.items() if s["cls"] in cls and k in to and k in tt]

    def shares(rows: list[dict]) -> str:
        if not rows:
            return "-"
        n = len(rows)
        return (f"{sum(r['same_as_student'] for r in rows) / n:.0%} / {sum(r['stale'] for r in rows) / n:.0%} / "
                f"{sum(r['malformed'] for r in rows) / n:.0%}")

    def esc(rows: list[dict]) -> str:
        return f"{sum(map(escaped, rows)) / len(rows):.0%}" if rows else "-"

    def pmean(n: str, keys) -> str:
        xs = [x for x in (pc(n, k) for k in keys) if x is not None]
        return fmt(mean(xs), 2) if xs else "-"

    lens = [r["thought_len"] for r in to.values()]
    ends = Counter(r.get("thought_end", "closed") for r in to.values())
    least = next((r.get("min_thought") for r in to.values()), None)
    budget = next((r.get("budget") for r in to.values()), None)
    out = {"states": len(to), "min_thought": least, "thought_len_median": st.median(lens) if lens else None,
           "thought_end": dict(ends), "no_call": sum(not r["has_call"] for r in to.values()),
           "by_class": {}, "t_think_skipped": {}, "t_think_thought": {}, "block_states": {}}
    lines = ["## 4. The forced thought: thinking on, with the thought forced open", "",
             f"T-open is T-think with the thought channel already open: the thinking-on prompt, then "
             f"<|channel>thought\\n, which the model can't close before {least} tokens (vLLM's min_tokens); at most "
             f"{budget} tokens in all, at the run's sampling. Then its own call, as "
             "T-think's. T-think decides itself whether to think; T-open thinks at every state. A real thought is "
             f"longer than the empty one ({EMPTY_LEN} tokens). Escape: the sampled call is a call, neither a stale "
             "repeat nor malformed.", "",
             f"T-open's thoughts: {len(to)}, median {fmt(out['thought_len_median'], 0)} tokens; ended "
             + ", ".join(f"{k} {n}" for k, n in ends.most_common()) + f"; no call after the thought: {out['no_call']}.",
             "", "| class | states | T-think thinks | p(call): S / T-think / T-open | T-think's own call: same / stale / "
             "malformed | T-open's own call: same / stale / malformed | escape: T-think / T-open |",
             "|---|---|---|---|---|---|---|"]
    for cls in CLASSES:
        keys = keys_of((cls,))
        if not keys:
            continue
        row = {"states": len(keys), "t_think_thinks": sum(map(real, keys)) / len(keys)}
        lines.append(f"| {cls} | {len(keys)} | {row['t_think_thinks']:.0%} | "
                     f"{pmean('S', keys)} / {pmean('T-think', keys)} / {pmean('T-open', keys)} | "
                     f"{shares([tt[k] for k in keys])} | {shares([to[k] for k in keys])} | "
                     f"{esc([tt[k] for k in keys])} / {esc([to[k] for k in keys])} |")
        out["by_class"][cls] = dict(row, escape_t_think=mean(map(escaped, (tt[k] for k in keys))),
                                    escape_t_open=mean(map(escaped, (to[k] for k in keys))))
    lines += ["", "Where T-think chose: the looping classes (" + ", ".join(LOOPING) + ") and malformed, split by "
              "whether T-think wrote a real thought there. The left half is the self-selection: the states where the "
              "model, free to think, went straight to the call (or wrote the empty thought).", "",
              "| class | T-think skipped: states | T-think: p(call), stale | T-open: p(call), stale, escape "
              "| T-think thought: states | T-think: stale, escape | T-open: stale, escape |", "|---|---|---|---|---|---|---|"]
    for label, cls in [(c, (c,)) for c in LOOPING] + [("all four looping", LOOPING), ("malformed", ("malformed",))]:
        keys = keys_of(cls)
        if not keys:
            continue
        skip, think = [k for k in keys if not real(k)], [k for k in keys if real(k)]
        stale = lambda g, ks: f"{sum(g[k]['stale'] for k in ks) / len(ks):.0%}" if ks else "-"
        lines.append(f"| {label} | {len(skip)} | {pmean('T-think', skip)}, {stale(tt, skip)} | "
                     f"{pmean('T-open', skip)}, {stale(to, skip)}, {esc([to[k] for k in skip])} | {len(think)} | "
                     f"{stale(tt, think)}, {esc([tt[k] for k in think])} | "
                     f"{stale(to, think)}, {esc([to[k] for k in think])} |")
        for part, ks in (("t_think_skipped", skip), ("t_think_thought", think)):
            out[part][label] = {"states": len(ks), "tasks": len({states[k]["task"] for k in ks}),
                                "t_think_stale": mean(tt[k]["stale"] for k in ks),
                                "t_open_stale": mean(to[k]["stale"] for k in ks),
                                "t_think_escape": mean(escaped(tt[k]) for k in ks),
                                "t_open_escape": mean(escaped(to[k]) for k in ks)}
    blk = [k for k in keys_of(LOOPING) if k in own["block"] and k in own["C-none"]]
    if blk:
        lines += ["", f"On the {len(blk)} looping states where block applies (section 3), the share of sampled calls "
                  "that escape:", "",
                  "| context | escape | stale | malformed |", "|---|---|---|---|"]
        for name, rows in (("block (empty thought, thinking off)", [own["block"][k] for k in blk]),
                           ("C-none (empty thought, thinking on)", [own["C-none"][k] for k in blk]),
                           ("T-think (its own choice)", [tt[k] for k in blk]),
                           ("T-open (a forced thought)", [to[k] for k in blk])):
            n = len(rows)
            lines.append(f"| {name} | {esc(rows)} | {sum(r['stale'] for r in rows) / n:.0%} | "
                         f"{sum(r['malformed'] for r in rows) / n:.0%} |")
            out["block_states"][name.split(" ")[0]] = {"states": n, "escape": mean(map(escaped, rows))}
    rng = random.Random(a.seed)
    ex = []
    for cls in ("copy_3rd_plus", "return_to_earlier"):
        pool = [k for k in keys_of((cls,)) if to[k].get("thought_text")]
        for k in rng.sample(pool, min(2, len(pool))):
            r = to[k]
            ex += [f"- {cls}, call {states[k]['i']} of {states[k]['n_calls']}, {'escaped' if escaped(r) else 'stale' if r['stale'] else 'not escaped'}:"
                   f" the student's call `{trim(states[k]['call_text'], 120)}`",
                   f"  - T-open's thought ({r['thought_len']} tokens): {trim(r['thought_text'], 500)!r}",
                   f"  - its call: `{trim(r['call_text'], 160)}`"]
    if ex:
        lines += ["", "Examples (T-open's thought, then its call):", ""] + ex
    lines.append("")
    return lines, out


def cmd_report(a) -> int:
    states = {s["key"]: s for s in iter_jsonl(a.work / "states.jsonl")}
    gens = {n: {r["key"]: r for r in iter_jsonl(a.work / f) if r["key"] in states} for n, f in GEN.items()}
    gen = gens["T-think"]
    scores = {}
    for name in CONTEXTS:
        rows = {r["key"]: r for r in iter_jsonl(a.work / f"score_{name}.jsonl") if r["key"] in states}
        if rows:
            scores[name] = rows
    if "S" not in scores:
        print("FAIL: no score_S.jsonl yet; run score first")
        return 1
    out = {"work": str(a.work), "states": len(states), "contexts": {k: len(v) for k, v in scores.items()},
           "teachers": {}, "replay": {}}
    lines = [f"# Gate 1: {a.work.name}", "",
             f"{len(states)} states in {len({(s['run'], s['task']) for s in states.values()})} tasks. "
             f"Scored: {', '.join(f'{k} {len(v)}' for k, v in scores.items())}.", ""]

    # 1. the gate: does a teacher's KL sit on the failure tokens?
    lines += ["## 1. The gate: KL on the failure tokens against routine tokens", "",
              "Per token of the student's call (the roles " + ", ".join(CALL_ROLES) + "): forward KL over the teacher's "
              f"top-{a.topk} (nats), and the log-ratio log p_T - log p_S of the student's own token. Failure = the "
              "classes " + ", ".join(STUCK) + "; routine = the class routine. Ratio = failure KL / routine KL, with a 95% "
              f"CI from {a.reps} resamples of tasks. Proposed rule: a teacher passes if the ratio is at least "
              f"{a.gate_ratio} and the CI's lower end is above {a.gate_low}.", "",
              "| teacher | failure KL | routine KL | ratio (95% CI) | failure log-ratio | routine log-ratio | verdict |",
              "|---|---|---|---|---|---|---|"]
    for name, rows in scores.items():
        if name == "S" or name in ("note", "dedup", "block"):   # block: section 3, on the states where it applies
            continue
        per_task = defaultdict(lambda: [0.0, 0, 0.0, 0])
        lr_f, lr_r = [], []
        cell = defaultdict(lambda: defaultdict(list))   # class -> role -> [kl...]
        cell_lr = defaultdict(lambda: defaultdict(list))
        tasks_of = defaultdict(lambda: defaultdict(set))
        for key, r in rows.items():
            s = states[key]
            for role, kl, lr in zip(s["roles"], r["kl"], r["lr"]):
                if role not in CALL_ROLES:
                    continue
                cell[s["cls"]][role].append(kl)
                cell_lr[s["cls"]][role].append(lr)
                tasks_of[s["cls"]][role].add(s["task"])
                t = s["task"]   # the unit of the bootstrap: a task, over all runs scored
                if s["cls"] in STUCK:
                    per_task[t][0] += kl
                    per_task[t][1] += 1
                    lr_f.append(lr)
                elif s["cls"] == "routine":
                    per_task[t][2] += kl
                    per_task[t][3] += 1
                    lr_r.append(lr)
        fk = sum(x[0] for x in per_task.values()) / max(sum(x[1] for x in per_task.values()), 1)
        rk = sum(x[2] for x in per_task.values()) / max(sum(x[3] for x in per_task.values()), 1)
        point, lo, hi = bootstrap_ratio(dict(per_task), a.reps, a.seed)
        verdict = ("PASS" if point is not None and lo is not None and point >= a.gate_ratio and lo > a.gate_low
                   else "fail" if point is not None else "-")
        ci = f" ({fmt(lo, 2)}-{fmt(hi, 2)})" if lo is not None else ""
        lines.append(f"| {name} | {fmt(fk)} | {fmt(rk)} | {fmt(point, 2)}{ci} | "
                     f"{fmt(mean(lr_f))} | {fmt(mean(lr_r))} | {verdict} |")
        out["teachers"][name] = {
            "failure_kl": fk, "routine_kl": rk, "ratio": point, "ci": [lo, hi], "verdict": verdict,
            "failure_lr": mean(lr_f), "routine_lr": mean(lr_r),
            "by_class_role": {cls: {role: {"kl": mean(v), "lr": mean(cell_lr[cls][role]), "tokens": len(v),
                                           "tasks": len(tasks_of[cls][role])} for role, v in roles.items()}
                              for cls, roles in cell.items()}}
    lines += ["", "T-hint differs from S only at stale and malformed calls, so it has no routine KL and no ratio: "
              "read its failure KL against the other teachers'. A negative log-ratio on the failure tokens means the "
              "teacher puts less mass than the student on what the student did.", ""]
    for name in ("T-think", "T-hint", "T-patch"):
        t = out["teachers"].get(name)
        if not t:
            continue
        lines += [f"**{name}, mean KL by class and role** (tokens in brackets)", "",
                  "| class | " + " | ".join(CALL_ROLES) + " |", "|---|" + "---|" * len(CALL_ROLES)]
        for cls in CLASSES:
            row = t["by_class_role"].get(cls, {})
            lines.append(f"| {cls} | " + " | ".join(
                f"{fmt(row[r]['kl'], 2)} ({row[r]['tokens']})" if r in row else "-" for r in CALL_ROLES) + " |")
        lines.append("")

    # 2. the replay: what breaks the copy?
    lines += ["## 2. The replay: the probability of the student's own call in each context", "",
              f"Mean over states of p(call), at the run's sampling (T={a.temperature}, top_p {a.top_p}) unless marked. "
              "At a copy, p(call) is the probability that the student copies again. T-think's own call: one sample "
              "after its own thought; same = the student's call exactly, stale = a stale repeat in that context.", "",
              "| class | states | S | S at T=1 (top_p) | C-none | T-think | note | dedup | T-hint | T-think's own call: same / stale / malformed |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for cls in CLASSES:
        keys = [k for k, s in states.items() if s["cls"] == cls and k in scores["S"]]
        if not keys:
            continue
        row = {"states": len(keys)}
        cells = []
        for name, field in (("S", "logp_call_run"), ("S", "logp_call_t1"), ("C-none", "logp_call_run"),
                            ("T-think", "logp_call_run"), ("note", "logp_call_run"), ("dedup", "logp_call_run"),
                            ("T-hint", "logp_call_run")):
            got = [(states[k]["task"], pexp(scores[name][k][field])) for k in keys if name in scores and k in scores[name]]
            m = mean(x for _, x in got)
            by_task = defaultdict(list)
            for t, x in got:
                by_task[t].append(x)
            tw = mean(mean(xs) for xs in by_task.values())
            row[f"{name}:{field}"] = {"mean": m, "task_weighted": tw, "n": len(got), "tasks": len(by_task)}
            cells.append(f"{fmt(m, 2)} ({len(got)})" if got else "-")
        g = [gen[k] for k in keys if k in gen]
        if g:
            same, stale, mal = (sum(x[f] for x in g) / len(g) for f in ("same_as_student", "stale", "malformed"))
            row["t_think_call"] = {"n": len(g), "same": same, "stale": stale, "malformed": mal,
                                   "no_call": sum(not x["has_call"] for x in g) / len(g),
                                   "thought_len_median": st.median(x["thought_len"] for x in g)}
            cells.append(f"{same:.0%} / {stale:.0%} / {mal:.0%} ({len(g)})")
        else:
            cells.append("-")
        out["replay"][cls] = row
        lines.append(f"| {cls} | {len(keys)} | " + " | ".join(cells) + " |")
    lines += ["", "Columns are means over the states each context applies to (n in brackets): note only where the "
              "previous call found nothing or repeated the one before it, dedup only where the context holds two or "
              "more copies of the last call, T-hint only for stale or malformed calls.", "",
              "The same, as the mean of per-task means (a few looping tasks hold many states):", "",
              "| class | tasks | S | S at T=1 (top_p) | C-none | T-think | note | dedup |", "|---|---|---|---|---|---|---|---|"]
    for cls, row in out["replay"].items():
        cols = ("S:logp_call_run", "S:logp_call_t1", "C-none:logp_call_run", "T-think:logp_call_run",
                "note:logp_call_run", "dedup:logp_call_run")
        lines.append(f"| {cls} | {row['S:logp_call_run']['tasks']} | " + " | ".join(
            fmt(row[c]["task_weighted"], 2) if row.get(c, {}).get("n") else "-" for c in cols) + " |")
    lines.append("")
    if gen:
        tl = [x["thought_len"] for x in gen.values()]
        lines += [f"T-think's thoughts: {len(tl)}, median {st.median(tl):.0f} tokens, "
                  f"{sum(x['truncated'] for x in gen.values())} cut at the budget, "
                  f"{sum(not x['thought'] for x in gen.values())} without a thought.", ""]

    # 3. the forced empty thought: the block, or thinking on?
    if "block" in scores and "C-none" in scores:
        more, out["block"] = report_block(a, states, scores, gen)
        lines += more

    # 4. the forced thought: does thinking break the copy where the model would not think on its own?
    if gens["T-open"]:
        more, out["open"] = report_open(a, states, scores, gens)
        lines += more
    (a.work / "gate1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (a.work / "gate1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="run dirs: shard_*/{calls.jsonl,traces/,task_results.jsonl}")
    p.add_argument("--data", type=Path, help="the competition data dir (tasks.jsonl), for the patched teachers")
    p.add_argument("--on-runs", type=Path, nargs="*", default=[],
                   help="thinking-on run dirs whose own prompts check the thinking-on edit (e.g. base_v30_l40s)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply every class's total cap (0.1 for a quick try)")
    p.add_argument("--seed", type=int, default=0)
    for name, chunk in (("generate", 8192), ("score", 2048), ("own", 8192)):
        p = sub.add_parser(name)
        p.add_argument("--max-len", type=int, default=32768)
        p.add_argument("--gpu-mem", type=float, default=0.85)
        p.add_argument("--chunk-tokens", type=int, default=chunk,
                       help="vLLM's max_num_batched_tokens: score keeps it small, since prompt logprobs need a "
                            "log-softmax over the whole vocabulary at every position of a chunk")
        p.add_argument("--max-seqs", type=int, default=32)
        p.add_argument("--topk", type=int, default=20, help="top logprobs per position (vLLM's max_logprobs)")
    sub.choices["generate"].add_argument("--context", choices=sorted(GEN), default="T-think",
                                         help="T-think (the model chooses whether to think) or T-open (forced to)")
    sub.choices["generate"].add_argument("--budget", type=int, default=512, help="thinking tokens at most")
    sub.choices["generate"].add_argument("--min-thought", type=int, default=MIN_THOUGHT,
                                         help="T-open: the thought can't close before this many tokens")
    sub.choices["generate"].add_argument("--call-tokens", type=int, default=1024, help="tokens for the own call")
    sub.choices["generate"].add_argument("--chunk", type=int, default=128,
                                         help="states per batch written to the context's file (gen.jsonl for T-think)")
    sub.choices["score"].add_argument("--contexts", type=lambda s: [x for x in s.split(",") if x],
                                      default=["S", "C-none", "T-think", "T-hint", "note", "dedup", "block"])
    sub.choices["score"].add_argument("--batch", type=int, default=8, help="prompts per vLLM call (prompt logprobs are big)")
    sub.choices["own"].add_argument("--contexts", type=lambda s: [x for x in s.split(",") if x], default=list(OWN))
    sub.choices["own"].add_argument("--call-tokens", type=int, default=1024, help="tokens for the own call")
    sub.choices["own"].add_argument("--chunk", type=int, default=128,
                                    help="states per batch written to own_<context>.jsonl")
    p = sub.add_parser("report")
    p.add_argument("--reps", type=int, default=2000, help="bootstrap resamples of tasks")
    p.add_argument("--gate-ratio", type=float, default=2.0)
    p.add_argument("--gate-low", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    for p in sub.choices.values():
        p.add_argument("--work", type=Path, required=True, help="the work dir every step reads and writes")
        if p is not sub.choices["report"]:
            p.add_argument("--model", type=Path, required=True, help="the model dir (tokenizer and chat template)")
        p.add_argument("--temperature", type=float, default=0.2, help="the run's sampling (off_args: 0.2)")
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--top-k", type=int, default=64, help="vLLM's default from the model's generation_config.json")
    sub.choices["report"].add_argument("--topk", type=int, default=20)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):   # a thought can hold any text; a log that can't encode it escapes it
        sys.stdout.reconfigure(errors="backslashreplace")
    cmds = {"prep": cmd_prep, "generate": cmd_generate, "score": cmd_score, "own": cmd_own, "report": cmd_report}
    return cmds[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
