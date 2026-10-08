# Adaptive inference: a per-call thinking controller

<!-- discussion session 2026-10-08; trace numbers from the v28 runs of 2026-10-04/05 -->

**The idea:** before each agent call (each round), decide whether the model thinks, and how much. The version here keeps thinking off by default, checks each drafted tool call mechanically, and turns thinking on only where it pays. It needs no training.

This file answers three questions:
1. Can we add it to the [OPSD study](../opsd_assessment.md) in the time we have?
2. Is it new?
3. What exactly do we build and run, and when?

## Bottom line

- **Yes: add it as one small, training-free arm of the OPSD study, not as a new direction.**
  - Cost: about 1.5 days of engineering in the token proxy, plus about 10–12 L40S-hours for 2 seeds on the 70 dev tasks.
  - It is the inference-time baseline the judges will ask about: if a controller gets thinking on's call hygiene, why train?
  - If OPSD fails a gate, it turns the fallback paper (the thinking-budget study) from measurement only into measurement plus a method.
  - The main track can't use it, because its harness has no callbacks. Our local harness has no such limit.
- **On its own, the novelty is low.**
  - Per-step effort routers for agents are published (Ares, CogRouter, TAB, ALAR). All are trained, and none is tested on SWE tasks.
  - Thinking in parallel while the agent acts, the asynchronous version, is published too, with SWE-Bench Pro: Second Thought (Aug 2026).
  - Still open, as far as we found: a training-free controller that checks each drafted call before it runs, for an open hybrid-thinking model, in a multi-turn SWE agent under a wall clock.
  - The comparison only our study can make: **who decides when a thinking-off agent thinks?** A rule (this controller), the model itself (the `agentic_v2` advisor), or training (OPSD).
- **The traces support it.** The two bad call types, stale repeats and malformed arguments, can be detected on the draft before the call runs. Once one gets into the context, the next call is bad 88% of the time with thinking off.
- **Start in shadow mode during the baseline runs.** The proxy logs what it would have done and changes nothing. Then a 10-task smoke, a go/no-go together with Gate 1 on Oct 13, and the full run on Oct 14–17.
- **Priority:** OPSD's Gate 1 work comes first. If that week slips, drop the controller first. If Gate 1 fails on Oct 13, the controller becomes the fallback paper's method and moves up.

## Why the main track couldn't do it, and why we can

- **The main track** (in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo), `agentic_workflow/README.md`):
  - The harness has no callbacks, since the registry is empty. Thinking is fixed per agent, in its `generate_content_config`.
  - So the only dynamic options are the agent's structure, an advisor sub-agent that the model calls itself (`agentic_v2`), or an adapter.
  - The same doc names the catch: "A looping model doesn't notice its loop, so the trigger must be mechanical."
- **Our local evals** send every call through `baseline/token_proxy.py`, which already rewrites each request body (it adds `return_token_ids`).
  - The harness sends thinking as two request fields: `thinking_token_budget`, and `enable_thinking` for the chat template. adk_submission maps a budget of 0 to `enable_thinking=false`.
  - vLLM runs with `--reasoning-parser gemma4` and `--reasoning-config`, so both fields work per request.
- **So a controller in the proxy changes nothing in ADK, swegemma or the agent zip,** and it works for any agent that talks to an OpenAI-compatible server. That also helps on the Quality criterion.

## What the traces say

We ran [trigger_replay.py](trigger_replay.py) over the 70 dev tasks of two v28 runs: `baseline_nothink_v28_zip` (thinking off) and `base_v28_zip` (thinking on, run A).
- **Bad** means a stale repeat or a malformed call. Both can be seen on the drafted call before it runs.
  - A stale repeat has the same tool and arguments as an earlier call, with no successful edit in between, so re-running a test after an edit doesn't count.
  - Malformed arguments are unknown or missing names, by `diagnose_run.py`'s rule.
- The main track's figures (50% repeats, 24% malformed) count any repeat, and count unknown and missing names separately.

| | Thinking off | Thinking on |
|---|---|---|
| Calls (tasks resolved) | 2,684 (19) | 1,545 (25) |
| Stale repeats / malformed / either | 47.2% / 29.8% / 52.6% | 4.1% / 2.7% / 6.0% |
| Next call bad, after a bad call | **88.2%** | 34.1% |
| Next call bad, after a clean and successful call | **14.8%** | 4.6% |
| Error result from a clean, well-formed call | 9.1% | 8.7% |
| Tasks with a bad call; the first at call (median, IQR) | 50/70; 9 (5–14) | 36/70; 13 (6–18) |
| Calls that edit source | 2.6% | 7.2% |

- **Bad calls chain through the context.** With thinking off, one bad call makes the next bad 88% of the time, against 15% after a clean call.
  - Discarding a bad draft before it runs keeps it out of the context.
  - It also saves the call budget. With thinking off, 28 of the 70 tasks used 55 or more of their 60 calls.
- **Thinking would fire on about a quarter of the calls.** Starting from a clean state, about 15% of drafts are bad and about 9% of results are errors. Source edits and the first call come on top.
  - On the thinking-on run's own trajectories, the full trigger set fires on 27% of calls.
- **Thinking on spends most of its thinking on routine calls:**

  | Call's state (thinking on) | Share of calls | Output tokens, median / mean | Share of all output tokens |
  |---|---|---|---|
  | routine | 76% | 70 / 209 | 71% |
  | source edit | 6% | 267 / 478 | 13% |
  | after a bad or errored call | 13% | 113 / 214 | 13% |
  | first call | 5% | 162 / 190 | 4% |

  - Output tokens are mostly thinking, since a plain call takes 20–40.
  - Estimate: if each triggered call thinks as long as thinking on does in that state, the controller keeps about 29% of thinking on's output tokens.
  - **The open question is whether the routine 71% is needed.** Most of the 74 calls over 1,000 tokens are routine. 19 come before a `run_command`, 19 before a `write_file` (mostly scratch scripts) and 14 before a `read_file`. Only 13 come before source edits and 6 after failures.
  - If the controller resolves fewer tasks than thinking on, this is the likely cause. The next trigger to try is "think before writing a new script".
- **Limit:** a replay can't show what the agent does after an intervention, because the trajectory changes. These numbers size the triggers; they don't predict the score.

## Novelty check (2026 literature)

Read on arXiv on 2026-10-08: abstracts only, plus Second Thought's results table.

| Work | What it does | Overlap with us |
|---|---|---|
| SwiftSage [2305.17390](https://arxiv.org/abs/2305.17390) (NeurIPS 2023) | a heuristic hands control from a small fast model to a slow LLM planner; ScienceWorld | the idea of a mechanical switch from fast to slow |
| CogRouter [2602.12662](https://arxiv.org/abs/2602.12662) (Feb) | 4 levels of reasoning depth chosen per step; SFT then RL; Qwen2.5-7B; ALFWorld, ScienceWorld; 62% fewer tokens | per step, trained, no SWE |
| Ares [2603.07915](https://arxiv.org/abs/2603.07915) (Mar) | a router picks each step's reasoning effort, trained on labels of the least effort that still succeeds; TAU-Bench, BrowseComp-Plus, WebArena; up to 52.7% fewer reasoning tokens | **closest:** per-step effort for agents; trained; no SWE; no clock |
| TAB [2604.05164](https://arxiv.org/abs/2604.05164) (Apr, v3 Aug) | per-turn thinking budgets; a policy trained with GRPO under a per-problem token limit; up to 35% fewer tokens and 30% less latency | per turn, trained |
| Cognitive Companion [2604.13759](https://arxiv.org/abs/2604.13759) (Apr) | a parallel monitor for loops and drift; 52–62% less repetition on loop-prone tasks; Gemma 4 E4B | loop detection on Gemma 4; no thinking switch; no SWE |
| HRBench [2605.28398](https://arxiv.org/abs/2605.28398) (May) | a benchmark of think/no-think switching strategies; 6 models; single-turn math, science and code | switching, but single-turn |
| ALAR [2606.02871](https://arxiv.org/abs/2606.02871) (Jun) | latent reasoning on routine turns, explicit reasoning on hard ones; trained; search and tool use | per turn, trained, no SWE |
| Real-Time Detection and Repair [2608.02464](https://arxiv.org/abs/2608.02464) (Aug) | telemetry monitors and deterministic checks; a flagged run is rolled back and re-run; success 52% → 73% | mechanical detection; the repair is a re-run, not thinking |
| Second Thought [2608.13667](https://arxiv.org/abs/2608.13667) (Aug) | training-free: 4 reasoning branches decode while each action runs, and are merged into the observation; SWE-Bench Pro, Terminal-Bench 2.1, τ3-bench; API models | **the asynchronous version.** On SWE-Bench Pro, Pass@1 moved −0.7 to +3.3 points, none significant; turns fell 6–11%. |

- **Taken:**
  - per-step effort routers that are trained, on non-SWE benchmarks;
  - thinking in parallel while acting, including on SWE;
  - heuristic switching from fast to slow (2023);
  - loop monitors.
- **Open, as far as we found:**
  - a training-free controller that checks each drafted call before it runs;
  - on an open hybrid model, in a multi-turn SWE agent, under a wall clock;
  - compared against the model asking for help itself, and against training (OPSD).
- **Cite:** SwiftSage, Ares, CogRouter, TAB, ALAR, HRBench and Second Thought. Also "Demystifying Hybrid Thinking" ([2510.12680](https://arxiv.org/abs/2510.12680)), which finds that current hybrids keep the two modes only partly apart: reasoning leaks into no-think mode.

## Design v0 (to settle during the baseline runs)

**Where:** in `baseline/token_proxy.py` in the main-track repo, behind a flag that is off by default.
- Both tracks' evals use this proxy, so a bug in the controller must not reach their baselines.
- It acts on the main agent's calls only. Sub-agent calls, such as `baseline_nothink`'s analyzer, pass through the same proxy.

**Per call:**
1. **Draft** the call with thinking off. That is the agent's own request, unchanged.
2. **Check the draft** against the request itself. The check holds no state, so compaction needs no special case. It flags:
   - **malformed arguments:** names checked against the request's `tools` schemas, unknown or a required one missing;
   - **a stale repeat:** the same name and arguments as a call in the current messages, with no successful `edit_file` or `write_file` since;
   - **no tool call before `submit_patch`,** which would cost a "continue" nudge;
   - **(variant) a source edit,** to think before committing to one.

   It also thinks without drafting first on the first call (the plan) and after an error result.
3. **If the draft is flagged, discard it and regenerate** with `enable_thinking=true` and a budget, starting at 1024. The harness only ever sees the regenerated call, so the bad one costs no tool call and no turn.
4. **Caps:** at most one regeneration per call, and at most about 12 thinking calls per task, so time stays bounded.
5. **Log both calls.**
   - The draft gets `"draft": true`. OPSD's packing and `build_rft_data.py` must skip it.
   - The regeneration records which trigger fired.
   - Kaggle pacing (`PACE=kaggle`) holds the reply for both calls' tokens.

**Base config:** the OPSD student config, once it is frozen. Then the controller and OPSD differ only in the method.
- `baseline_nothink`: both static baselines exist already: 19/70 off, and 25/70 on (20/70 in a second run). The two zips differ only in `thinking_budget`.
- `agentic_v1`: its static thinking-on arm needs one more run, since `agentic_v1_think` samples at 0.2 rather than 1.0.

**Metrics:** OPSD's metrics, plus two of its own.
- OPSD's: resolved, with 2 seeds and paired tests; the stale-repeat and malformed shares; how bad calls chain; time and tokens per task.
- The controller's own: the share of calls that think, and the share of drafts discarded.
- Resolved against the clock comes free: replay the recorded timings at shorter clocks, as the main track did for v1.2.

## How it fits the OPSD plan

| OPSD step | What the controller adds |
|---|---|
| Baseline runs (Oct 8–14) | Run `trigger_replay.py` on each new run's traces, free. Once the proxy code lands, run shadow mode on the runs that happen anyway, such as `agentic_v1` and the round-0 rollouts. That checks the live detector against `diagnose_run.py`'s counts at no GPU cost. |
| Gate 1 diagnostic (Oct 13) | Mark the controller's flags on the teacher-KL figure, over the same 70 trajectories. If T-think's KL sits on the flagged calls, both methods target the same failures: one at inference time, one in the weights. |
| Round 1 (Oct 17–21) | The controller's 2-seed run, as a comparison arm. |
| Gate 2 (Oct 22) | OPSD student vs. the controller vs. static off and on. |
| Round 2 (Oct 22–Nov 2), optional | OPSD student plus the controller: does training cut the trigger rate, and so the time? Also the reactive variant, which thinks on the call after a bad one and doesn't discard it. It tests whether the chain runs through the context. |
| Training data | Every flagged call gives a state, the student's bad draft and a thinking-on correction, all with exact token ids. These are free T-think thoughts at the states that matter. |

**The paper's new angle is the three-way comparison:** who decides when a thinking-off SWE agent thinks, a rule, the model, or training? The arm where the model decides is `agentic_v2` (advisor), which the main track runs locally anyway.

## Plan and cost

| Date | Work | Cost |
|---|---|---|
| Thu Oct 8 | Replay (this file) | done |
| Fri–Sat Oct 9–10 | Proxy: shadow mode, then active mode; unit tests on captured requests | ~1.5 days |
| Sun–Mon Oct 11–12 | Smoke, 10 tasks, active mode | ~1 L40S-h |
| **Tue Oct 13** | **Go/no-go, together with Gate 1.** The bar: regenerated calls are mostly clean, at most ~35% of calls think, no harness errors, and no thinking leaks into thinking-off calls. | |
| Oct 14–17 | Full run: 2 seeds × 70 dev tasks, before round 1's evals need the L40S slots | ~10–12 L40S-h |
| Oct 22–Nov 2 | Optional arms: the reactive variant; OPSD plus the controller | ~20–25 L40S-h |

**Our odds** (judgement, not measurement):

| Outcome | Chance |
|---|---|
| The bad-call share falls clearly, from 53% toward thinking on's 6% | ~75% |
| Resolved at least 3 above the thinking-off base (22/70), over 2 seeds | ~50% |
| Matches thinking on (24/70 or more) in at most 60% of its agent time | ~35% |

## Not doing, for lack of time

- **A learned router** (Ares, TAB): it needs labels or RL, takes a week at least, and is published.
- **Asynchronous thinking:** it is published as Second Thought. It would need concurrent requests and injecting text into the context, and it couldn't run on Kaggle either.
- **A budget sweep per trigger,** beyond one ablation.
- **A main-track version:** the harness has no callbacks. The main track gets the comparison with its own advisor instead.

## Risks, and what the smoke checks

- **Mixing thinking settings within one conversation is untested on Gemma 4.**
  - Since v28, earlier thoughts stay in the prompt. Check how the template renders them on a thinking-off call.
  - Check whether thinking-off calls start writing reasoning once thoughts are in the context, since hybrids keep the modes only partly apart (2510.12680). Track the output tokens of thinking-off calls.
- **A regenerated call can still be bad:** a state that produced a bad draft is harder than average. Measure how often the regeneration is bad; if often, raise that trigger's budget.
- **Losing the routine thinking** (above):
  - Watch how much the agent reads before its first edit.
  - Watch the "fast wrong fix" failures: 3 of the 8 tasks that thinking off lost against run A.
- **The request fields:** check, on the first smoke request, where the harness puts `enable_thinking` (probably `chat_template_kwargs`), and that vLLM honours both fields per request.
- **The prefix cache:** a regeneration resends the same prompt, so its prefill should hit the cache. Check vLLM's hit rate in the smoke.
- **A fair comparison:** the controller run keeps the base config's `eval_config.yaml` (4 min, 60 calls), sampling and prompt.

## Open questions for our discussion

1. **Triggers:** failures only, or also the first call and source edits? Recommended: include both, since together they are only about 5–10% of calls.
2. **Budget:** one budget for every trigger (1024), or one per trigger (e.g. 256 for argument fixes, 1024 for loops and edits)? Or a budget that shrinks as the clock runs out?
3. **The model-decides arm:** `agentic_v2` also adds two skills, and its advisor sees only the request it is sent, not the conversation. Is V1 plus the advisor alone, a one-line change to `agent.yaml`, the cleaner arm?
4. **Where in the paper:** a comparison arm if OPSD works; the main method if OPSD fails a gate.

## Sources

- **Main-track repo** ([Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo)):
  - `agentic_workflow/README.md`: What the platform allows; Dynamic thinking; What the traces show;
  - `baseline/token_proxy.py`; `baseline/eval_gpu.slurm`, for vLLM's flags; `baseline/diagnose_run.py`, for `TOOL_ARGS`;
  - `docs/issue_743508_lora_wipe.md`: how the budget maps to `enable_thinking`;
  - `agents/agentic_v2/`.
- **Traces:** `logs/explorer_audit_20261005/{baseline_nothink_v28_zip,base_v28_zip}`, 70 tasks each. Every number in "What the traces say" comes from [trigger_replay.py](trigger_replay.py) in this folder.
- **arXiv:** the abstract pages linked in the novelty table, read 2026-10-08; for Second Thought, also its HTML results table.
