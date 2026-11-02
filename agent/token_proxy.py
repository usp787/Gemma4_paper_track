"""Pass-through proxy between the harness and vLLM that records the exact tokens of every chat call.

- Each /v1/chat/completions request gets "return_token_ids": true added.
- The proxy appends the prompt and completion token ids to --out as one JSON line, then removes
  those fields again, so the harness receives vLLM's normal response.
- These records are the RFT training data: what the served model actually saw (after ADK's
  compaction and the chat template) and produced. Training therefore needs no re-rendering
  (post_training_research.md, train/serve matching).
- baseline/run_eval.py posts the current task id to /_task before each task. Tasks run one at a
  time, so every call is tagged with the task that was current when the request arrived.
- If the harness drops the connection mid-call, the proxy drops its own request to vLLM, and vLLM
  aborts the generation (its chat route runs under `with_cancellation`). That happens when the
  4-min task timeout cancels the agent. On the scorer the harness talks to vLLM directly, so the
  same abort happens there.
  - Before this fix, such a request kept decoding for up to minutes into the next task, and its
    reply was recorded under that task.
  - In base_roll_p1 and the two smoke runs, 24 of 43 tasks got one of these stray calls, 67k
    completion tokens in total. They sat in RFT samples and inflated time estimates.
  - Aborted calls are logged to stdout, not recorded.
- Streaming requests (the harness sends none) are forwarded unrecorded.
- Each record also keeps the request's thinking_token_budget and how many of its messages carry
  earlier thinking (`reasoning`). Both are new with wheelhouse v28 (post_train/RFT/v28.md); before it,
  the budget was never sent and earlier thinking never reached the model.
- Kaggle pacing (--pace-call-s, --pace-tok-s; eval_gpu.slurm PACE=kaggle): each reply is held back
  until the call has taken as long as it would on Kaggle's 4x L4, call_s + completion tokens / tok_s.
  The agent then fits as much work into its wall-clock budget as on the scorer (post_train/v28/README.md).
  - The record keeps `secs`, vLLM's own time, and adds `pace_hold`, the time held back.
  - If the harness disconnects during the hold (the task timeout), it gets nothing and the call is not
    recorded. On Kaggle that call would still have been decoding.
- Dedup (--dedup N; eval_gpu.slurm DEDUP=N; off by default, as on Kaggle): for the paper track's loop study
  (Gemma4_paper_track loops/README.md, round 2's G4). Before forwarding, every run of N or more identical tool turns in
  a row is collapsed to its first turn, in what the model sees. A tool turn is an assistant message with tool calls and,
  right after it, one result per call (role "tool", or "tool_responses" as ADK's LiteLlm names them for gemma4
  models). Identical means the same tool names, arguments (as JSON, so key order aside) and result contents. The
  harness and its trace keep every call; only the request changes.
  - So while the agent re-sends a call that returns the same result, each request is exactly the one that followed
    the run's first call: the model is asked the same question again. A different result (a budget warning, a
    timestamp in the output) breaks the run.
  - The record adds `n_messages_in` (before) and `dedup`: tool turns, collapsed runs, dropped turns and messages, the
    longest run, and `tail`, the run that ends the request.
  - A request with a seed would repeat its reply when its prompt repeats; the proxy logs that once.

Runs with venvs/vllm_host (fastapi, uvicorn and httpx come with vLLM):
    python baseline/token_proxy.py --upstream http://127.0.0.1:8001 --port 8002 --out calls.jsonl [--dedup 2]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

FORWARD_HEADERS = ("content-type", "authorization")
TOOL_ROLES = ("tool", "tool_responses")   # ADK's LiteLlm writes tool_responses when the model string says gemma4


def _call_sig(c) -> list:
    f = (c.get("function") or {}) if isinstance(c, dict) else {}
    args = f.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return [f.get("name"), args]


def tool_turns(msgs: list) -> list[tuple[int, int, str]]:
    """(start, end, signature) of every tool turn: an assistant message with tool calls, followed at once by exactly one
    result for each of its call ids. end is exclusive. Anything else (a call without its results, results without a
    call) is no turn, so it is never collapsed."""
    out, i = [], 0
    while i < len(msgs):
        m = msgs[i]
        calls = m.get("tool_calls") if isinstance(m, dict) and m.get("role") == "assistant" else None
        if calls:
            ids = [c.get("id") if isinstance(c, dict) else None for c in calls]
            got, j = {}, i + 1
            while (j < len(msgs) and len(got) < len(ids) and isinstance(msgs[j], dict)
                   and msgs[j].get("role") in TOOL_ROLES and msgs[j].get("tool_call_id") in ids
                   and msgs[j].get("tool_call_id") not in got):
                got[msgs[j]["tool_call_id"]] = msgs[j].get("content")
                j += 1
            if None not in ids and len(set(ids)) == len(ids) and len(got) == len(ids):
                sig = json.dumps([[_call_sig(c) for c in calls], [got[x] for x in ids]], sort_keys=True, default=str)
                out.append((i, j, sig))
                i = j
                continue
        i += 1
    return out


def dedup_messages(msgs: list, min_run: int) -> tuple[list, dict]:
    """Collapses every run of min_run or more identical tool turns in a row to its first turn. Returns the new message
    list (the same objects, fewer of them) and what was done."""
    turns = tool_turns(msgs)
    drop, runs, k, tail = set(), [], 0, 0
    while k < len(turns):
        m = k
        while m + 1 < len(turns) and turns[m + 1][0] == turns[m][1] and turns[m + 1][2] == turns[k][2]:
            m += 1
        n = m - k + 1
        if n >= min_run:
            for s, e, _ in turns[k + 1:m + 1]:   # every turn of the run but its first
                drop.update(range(s, e))
            runs.append(n)
        if turns[m][1] == len(msgs):
            tail = n
        k = m + 1
    stats = {"turns": len(turns), "runs": len(runs), "dropped_turns": sum(n - 1 for n in runs),
             "dropped_messages": len(drop), "longest": max(runs, default=0), "tail": tail}
    return [x for i, x in enumerate(msgs) if i not in drop], stats


def build_app(upstream: str, out_path: str, pace: tuple[float, float] | None = None,
              dedup: int | None = None) -> FastAPI:
    """pace: (seconds per call, decode tokens per second) to hold replies to, or None for no pacing.
    dedup: collapse runs of this many identical tool turns or more (module docstring), or None for no edit."""
    app = FastAPI()
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(None), trust_env=False)
    out = open(out_path, "a", buffering=1, encoding="utf-8")
    state: dict = {"task": None, "n": 0, "seed_warned": False}

    def headers(req: Request) -> dict:
        return {k: v for k, v in req.headers.items() if k.lower() in FORWARD_HEADERS}

    def relay(r: httpx.Response) -> Response:
        return Response(r.content, r.status_code, media_type=r.headers.get("content-type"))

    async def client_gone(req: Request) -> None:
        """Returns once the harness has closed the connection. The body is already read, so the next
        ASGI message is the disconnect (vLLM's listen_for_disconnect does the same)."""
        while (await req.receive())["type"] != "http.disconnect":
            pass

    @app.post("/_task")
    async def set_task(req: Request):
        state["task"] = (await req.json()).get("instance_id")
        state["n"] = 0
        return {"task": state["task"]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        raw = await req.body()
        body = json.loads(raw)
        if body.get("stream"):
            return relay(await client.post("/v1/chat/completions", content=raw, headers=headers(req)))
        body["return_token_ids"] = True
        dd = None
        if dedup:
            n_in = len(body.get("messages") or [])
            body["messages"], dd = dedup_messages(body.get("messages") or [], dedup)
            if body.get("seed") is not None and not state["seed_warned"]:
                state["seed_warned"] = True
                print(f"WARN: requests carry a seed ({body['seed']}); a collapsed request repeats its prompt, "
                      "so it would repeat its reply too", flush=True)
        task, i = state["task"], state["n"]   # the task this call belongs to, fixed on arrival
        state["n"] += 1
        t0 = time.time()
        fwd = asyncio.create_task(client.post("/v1/chat/completions", json=body, headers=headers(req)))
        gone = asyncio.create_task(client_gone(req))
        await asyncio.wait({fwd, gone}, return_when=asyncio.FIRST_COMPLETED)
        if not fwd.done():
            fwd.cancel()   # closes the connection to vLLM, which then aborts the generation
            print(f"aborted: task={task} i={i}: the harness disconnected after {time.time() - t0:.1f} s", flush=True)
            return Response(status_code=499)
        try:
            r = fwd.result()
            secs = time.time() - t0
            if r.status_code != 200:
                return relay(r)
            data = r.json()
            prompt_ids = data.pop("prompt_token_ids", None)
            completion_ids = [c.pop("token_ids", None) for c in data.get("choices", [])]
            hold = 0.0
            if pace:
                n_out = (data.get("usage") or {}).get("completion_tokens") or len(completion_ids[0] or [])
                hold = max(0.0, pace[0] + n_out / pace[1] - secs)
                if hold > 0 and (await asyncio.wait({gone}, timeout=hold))[0]:
                    print(f"aborted: task={task} i={i}: the harness disconnected during the {hold:.1f}-s pace hold", flush=True)
                    return Response(status_code=499)
            rec = {
                "task": task, "i": i, "t": round(t0, 3), "secs": round(secs, 2),
                "model": body.get("model"), "n_tools": len(body.get("tools") or []),
                "n_messages": len(body.get("messages") or []),
                "thinking_budget": body.get("thinking_token_budget"),
                "n_reasoning": sum(1 for m in body.get("messages") or [] if isinstance(m, dict) and m.get("reasoning")),
                "finish": [c.get("finish_reason") for c in data.get("choices", [])], "usage": data.get("usage"),
                "prompt_ids": prompt_ids, "completion_ids": completion_ids[0] if len(completion_ids) == 1 else completion_ids,
            }
            if pace:
                rec["pace_hold"] = round(hold, 2)
            if dd is not None:
                rec["n_messages_in"], rec["dedup"] = n_in, dd
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            return Response(json.dumps(data), media_type="application/json")
        finally:
            gone.cancel()

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def passthrough(path: str, req: Request):
        return relay(await client.request(req.method, "/" + path, content=await req.body(),
                                          headers=headers(req), params=req.query_params))

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="vLLM server base URL, without /v1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True, help="calls.jsonl to append to")
    ap.add_argument("--pace-call-s", type=float, default=0.0, help="Kaggle pacing: fixed seconds per call (needs --pace-tok-s)")
    ap.add_argument("--pace-tok-s", type=float, help="Kaggle pacing: decode tokens per second; holds each reply to that speed")
    ap.add_argument("--dedup", type=int, default=0, metavar="N",
                    help="collapse every run of N >= 2 identical tool turns in a row to its first turn (0: off)")
    args = ap.parse_args()
    if args.dedup == 1 or args.dedup < 0:
        ap.error("--dedup takes 0 (off) or a run length of 2 or more")
    pace = (args.pace_call_s, args.pace_tok_s) if args.pace_tok_s else None
    if pace:
        print(f"pacing each call to {pace[0]} s + completion tokens / {pace[1]} tok/s", flush=True)
    if args.dedup:
        print(f"dedup: collapsing runs of {args.dedup}+ identical tool turns to their first turn", flush=True)
    uvicorn.run(build_app(args.upstream, args.out, pace, args.dedup or None), host="127.0.0.1", port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
