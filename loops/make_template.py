"""The always-block chat template (README.md, experiment 2): the model's own chat template with one edit.

With thinking off, the template prefills the empty thought <|channel>thought\\n<channel|> only where it opens a model
turn, after the task prompt. After a tool response it adds nothing, so the model writes that block itself or goes
straight to the call. In off_args's first v30 run it wrote the block at 82% of calls 1-5 and ~35-45% after call 10.
In Gate 1's replay, forcing the block halved the probability of a third identical copy (0.95 -> 0.50).
The edit makes the template prefill the same block after a tool response too, so every thinking-off call starts with
it. vLLM's gemma4 reasoning parser reads a prompt ending in <channel|> as reasoning already ended, as it does for every
task's first call.

    python make_template.py --model <model dir> --out <file>

The checks render probe conversations with the old and the new template, through the tokenizer's own
apply_chat_template (as vLLM renders a request), and fail unless:
- thinking off, and the prompt ends after a tool response: the new prompt is the old one plus the empty thought, token
  for token, and the old one doesn't already end with it;
- thinking off, anywhere else (after the task prompt, after a model reply, a call with no response yet): unchanged;
- thinking on, and without a generation prompt: unchanged everywhere.
Writes the template, and the checks to <out>.check.md. Exits 1 if a check fails or the template has changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EMPTY = "<|channel>thought\n<channel|>"
# The end of the model's chat_template.jinja (the June one, 16,934 bytes): the generation prompt.
OLD = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- endif -%}
{%- endif -%}"""
NEW = """{%- if add_generation_prompt -%}
    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}
        {{- '<|turn>model\\n' -}}
        {%- if not enable_thinking | default(false) -%}
            {{- '<|channel>thought\\n<channel|>' -}}
        {%- endif -%}
    {%- elif ns.prev_message_type == 'tool_response' and not enable_thinking | default(false) -%}
        {#- always-block (Gemma4_paper_track loops/): with thinking off, the empty thought after a tool response too -#}
        {{- '<|channel>thought\\n<channel|>' -}}
    {%- endif -%}
{%- endif -%}"""

TOOLS = [{"type": "function", "function": {
    "name": "run_command", "description": "Run a shell command in the repository.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command."}},
                   "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"filepath": {"type": "string", "description": "The path."}},
                       "required": ["filepath"]}}}]


def call(i: int, name: str = "run_command", **args) -> dict:
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": args or {"command": "grep -rn foo ."}}}


def result(i: int, name: str = "run_command", out: str = "a.py:1:foo\n") -> dict:
    return {"role": "tool", "tool_call_id": f"c{i}", "name": name,
            "content": json.dumps({"status": "ok", "stdout": out, "stderr": "", "exit_code": 0})}


def probes() -> dict[str, tuple[list[dict], bool]]:
    """name -> (messages, does the prompt end after a tool response)"""
    base = [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": "Fix the bug in a.py."}]
    turn = lambda i: [{"role": "assistant", "content": "", "tool_calls": [call(i)]}, result(i)]
    return {
        "after the task prompt": (base, False),
        "after a tool response": (base + turn(1), True),
        "after two identical calls": (base + turn(1) + turn(2), True),
        "after an empty result": (base + [{"role": "assistant", "content": "", "tool_calls": [call(1)]},
                                          result(1, out="")], True),
        "after two calls in one reply": (base + [{"role": "assistant", "content": "",
                                                  "tool_calls": [call(1), call(2, "read_file", filepath="a.py")]},
                                                 result(1), result(2, "read_file", "x = 1\n")], True),
        "after a reply with text and a call (the turn is closed)": (
            base + [{"role": "assistant", "content": "Let me look.", "tool_calls": [call(1)]}, result(1)], True),
        "after a call with its reasoning in the history": (
            base + [{"role": "assistant", "content": "", "reasoning": "Search first.", "tool_calls": [call(1)]},
                    result(1)], True),
        "after a model reply with no call": (base + turn(1) + [{"role": "assistant", "content": "Done."}], False),
        "after a call with no response yet": (base + [{"role": "assistant", "content": "", "tool_calls": [call(1)]}],
                                              False),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="the model dir: chat_template.jinja and the tokenizer")
    ap.add_argument("--out", type=Path, required=True, help="where to write the always-block template")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    old = (a.model / "chat_template.jinja").read_text(encoding="utf-8")
    lines = [f"# The always-block template: checks", "",
             f"Model template: {a.model / 'chat_template.jinja'}, {len(old.encode())} bytes, "
             f"sha256 {hashlib.sha256(old.encode()).hexdigest()[:12]}.", ""]
    if old.count(OLD) != 1 or not old.rstrip().endswith(OLD):
        print("\n".join(lines) + f"FAIL: the template's generation prompt isn't the expected one (found {old.count(OLD)} "
              "times); the model's template has changed, so check the edit by hand")
        return 1
    new = old.replace(OLD, NEW)
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    empty = tok.encode(EMPTY, add_special_tokens=False)
    fails = 0
    lines += ["| probe | thinking | generation prompt | old prompt ends with | new against old | check |",
              "|---|---|---|---|---|---|"]
    for name, (msgs, after_tool) in probes().items():
        for think in (False, True):
            for gen in (True, False):
                r = {}
                for which, tpl in (("old", old), ("new", new)):
                    text = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=gen,
                                                   enable_thinking=think, chat_template=tpl)
                    r[which] = tok.encode(text, add_special_tokens=False)
                o, n = r["old"], r["new"]
                if gen and not think and after_tool:
                    ok = n == o + empty and o[-len(empty):] != empty
                    how = "old + the empty thought" if n == o + empty else "differs"
                else:
                    ok = n == o
                    how = "the same" if ok else "differs"
                fails += not ok
                tail = repr(tok.decode(o[-6:])).replace("|", "\\|")   # a | inside a table cell ends the cell
                lines.append(f"| {name} | {'on' if think else 'off'} | {'yes' if gen else 'no'} | "
                             f"{tail} | {how} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", f"{fails} check(s) failed." if fails else "All checks passed."]
    report = "\n".join(lines) + "\n"
    print(report)
    if fails:
        print("FAIL: the template is not written")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(new, encoding="utf-8", newline="\n")   # the same bytes, and sha256, on any OS
    sha = hashlib.sha256(new.encode()).hexdigest()
    report += f"\nWrote {a.out}: {len(new.encode())} bytes, sha256 {sha[:12]} (eval_gpu.slurm's .chat_template).\n"
    Path(str(a.out) + ".check.md").write_text(report, encoding="utf-8")
    print(f"wrote {a.out} (sha256 {sha[:12]}) and {a.out}.check.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
