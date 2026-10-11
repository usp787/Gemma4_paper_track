# gate1/: Gate 1 of the OPSD study

<!-- written 2026-10-10; not run on Explorer yet. Trace numbers from off_args's first v30 run, pulled 2026-10-10 -->

Gate 1 is the go/no-go of [../opsd_assessment.md](../opsd_assessment.md#14-what-to-change) (§1.4, change 1). It scores recorded trajectories of the thinking-off student, with no training and no new rollouts, and answers two questions:
1. **The gate:** does some teacher put clearly more KL on the student's failure tokens than on its routine tokens? If none does, the fallback paper starts.
2. **The replay:** at the student's own stuck states, what breaks the copy?
   - thinking: T-think's own next call, and the copy's probability after T-think's thought;
   - the thinking-on template alone;
   - a controller's note;
   - fewer copies in the context;
   - a higher temperature.

   That is the premise of the rule-based controller ([../adaptive_inference/README.md](../adaptive_inference/README.md)) and of the temperature fix, measured one step ahead at the exact states, without an L40S run.

By default it scores `off_args`'s first v30 run, `baseline_nothink_args_v30_zip`. Add `_r2` once step 3 has run.

## Why these classes: the stuck loops of `off_args`

From the v30 traces, by `trigger_replay.py`'s rules. Each arm has one run.

| | `off_args` | `on` (thinking on) |
|---|---|---|
| stale repeats / of them `run_command` | 659 / 580 | 106 / 13 |
| after a search that found nothing: the next call changes approach (pooled; median of tasks) | 11.5%; 0% | 39%; 33% |
| after two identical calls in a row: the third follows | 72% (n=32) | 60% (n=10, nearly all `edit_file` argument errors) |
| after a stale `run_command`: the next call is bad | 83% (481 of 577) | 1 of 12 |

- **What repeats.** Of the 580 repeated `run_command` calls:
  - 299 got the same non-empty output again;
  - 270 got nothing: 163 were a grep's silent exit 1, 107 an empty "ok" from a pipe.
  - 91% of all stale repeats got exactly the result of their earlier copy.
- **How a loop starts.** An empty result is rarely copied at once: the next call is a variant of the same search 78% of the times, an identical copy 1%.
  - 114 of the 131 chains of bad calls start by re-sending a call from 2 or more calls back.
  - 58 of those 131 (44%) re-send a call whose earlier copy was compacted out of view.
  - 33 (25%) cycle back within 2–5 calls, after the variants failed too.
  - In the examples, the model searches for something that isn't there: a function name it made up, or a string that exists only in the tests.
- **Lock-in.** After two identical calls in a row, every further copy follows 83–94% of the time.
  - `agentic_v2` on v28 shows the same lock-in (73%) under temperature 1.0 and V1's rule "If two calls in a row return the same result, don't make a third".
  - What that bundle did cut is how often copies start: 1.9% of calls, against 6.7% for `off` on v28. `off_args` is at 1.6%.
- **Escapes come from outside.** Of `off_args`'s 23 runs of 3 or more identical calls:
  - 8 end at a compaction;
  - 7 when the budget warning arrives, after which the model submits, mostly an empty patch;
  - 5 on their own;
  - 3 with the task.
- **Spread and ceiling.**
  - **Spread:** the stale repeats fall in 45 tasks, and the 10 worst hold 59%. Without the 5 worst, the next call after a bad one is still bad 70% of the time (`on`: 19%).
  - **Ceiling:** of the 23 failures with at least 20% stale calls, only 6 were solved by any of 9 other runs (v28–v30). So removing every loop would gain about 2 tasks.
  - Loops are where thinking-off fails visibly, and where its behaviour differs most from thinking on.
- **Prior work agrees.**
  - Repetition reinforces itself (Holtzman et al., ICLR 2020; Xu et al., NeurIPS 2022).
  - Models err more once their own errors are in context, and thinking removes that (Sinha et al., ICLR 2026).
  - Prompts asking the model to check itself don't help without outside feedback (Huang et al., ICLR 2024; Kamoi et al., TACL 2024).

## What it measures

**States:** a state is the context before one main-agent call, together with the call the student made there. Each call gets the first class that fits.

| Class | The student's call | Default pick: per task, in all |
|---|---|---|
| `copy_2nd` | a stale repeat of the call right before it: the second identical call in a row, where lock-in starts | all, 80 |
| `copy_3rd_plus` | the third or later identical call in a row | 3, 80 |
| `return_to_earlier` | a stale repeat of an earlier, not adjacent, call, right after a clean call: where most chains start | 4, 110 |
| `stale_other` | any other stale repeat (inside a chain) | 2, 50 |
| `malformed` | unknown or missing argument names | all, 40 |
| `after_miss` | a clean call right after a `run_command` that found nothing | 3, 100 |
| `after_bad` | a clean call right after a bad one: an escape | 2, 50 |
| `after_error` | a clean call right after another error | 2, 40 |
| `first` | the task's first call | 1, 35 |
| `routine` | everything else | 2, 120 |

About 650 states on one run. The picks spread over tasks round-robin, with a fixed seed; `--scale` multiplies the totals.

**Token roles,** from Gemma 4's call syntax `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>`:
- `tool`, `argname`, and a value's role: `command`, `path`, `code` or `value`;
- `struct` for the syntax;
- `think` for the student's own empty thought block, which is never scored.

**Contexts:** each is a token-level edit of the recorded prompt; the student's completion is scored after it.

| Context | What the scorer sees | Tests |
|---|---|---|
| S | the recorded prompt: the student itself | the baseline of every log-ratio and KL; the copy's probability at the run's sampling and at T=1 |
| C-none | the thinking-on template (`<|think|>` in the system turn), then an empty thought | the template's effect alone |
| T-think | the thinking-on template, then a thought the same weights write there (at most 512 tokens, T=0.2) | **the proposed teacher**, and the controller's premise |
| T-hint | the recorded prompt plus a hindsight note on stale and malformed calls: what the call repeats and what that returned, or the right argument names | the hindsight teacher (HERO/SDPO idea) |
| note | the recorded prompt plus what a controller could say before the call: "printed nothing" after a miss, "identical to the one before it" after a repeat | a controller that injects text instead of thinking (OpenHands' and Gemini CLI's loop nudges), local only |
| dedup | the recorded prompt without the earlier copies of its last call | self-conditioning: is it the copies in context? |
| T-patch / T-patch-on | the gold patch at the end of the system prompt, thinking off / on (empty thought) | PSP's test at 31B |
| T-loc | the patch's files and functions only | a weaker hint |
| C-wrong | another task's gold patch | the control for T-patch |

- The notes go into the last tool response's JSON as one more key, where the harness puts its `budget_warning`.
- The gold patches come from `/scratch/$USER/gemma4/data/tasks.jsonl`, cut at 8,000 characters. They stay in the work dir and never reach this repo.

**Per token:**
- the log-ratio log p_T − log p_S of the student's own token (negative: the teacher disagrees with what the student did);
- the forward KL over the teacher's top 20 tokens. A token missing from the student's top 20 gets the student's lowest listed logprob, so the KL is understated.

**Per call:** the probability of the student's whole call (`<|tool_call>` to `<tool_call|>`):
- at T=1, raw;
- at the run's sampling (T=0.2, top_p 0.95, top_k 64);
- at T=1 with top_p 0.95, which is Gemma's prescribed sampling.

The last two are computed from the top 20 logprobs.

**T-think's own call:** one sample after its thought, checked three ways:
- the student's call exactly;
- a stale repeat in that context;
- malformed.

**The gate (proposed; you decide).**
- Failure tokens are the call tokens (roles `tool` to `value`) of the classes `copy_2nd` to `malformed`; routine tokens are those of `routine`.
- A teacher passes if its failure KL is at least 2× its routine KL, and the lower end of a 95% CI from 2,000 task resamples is above 1.
- T-hint has no routine KL by construction, so read its failure KL against the other teachers'.
- `--gate-ratio` and `--gate-low` change the rule.

## Commands

On the Explorer login node, from `~/Gemma4_paper_track`, after `git pull`. Nothing is queued without `GO=1`.

```bash
bash gate1/submit_gate1.sh                  # dry run: prints the jobs and what blocks them
GO=1 bash gate1/submit_gate1.sh             # prep (CPU) -> core (A100) -> report (CPU), on off_args's first v30 run
GO=1 PATCH=1 bash gate1/submit_gate1.sh     # the same, plus the patched teachers (a second A100 job, after core)
GO=1 PREP_ARGS="--scale 0.1" NAME=gate1_try bash gate1/submit_gate1.sh   # a quick try: about 65 states
```

- **What the jobs do:**
  - prep checks the prompt edits against the model's chat template and against `on`'s recorded thinking-on prompts (`base_v30_l40s`); a failed check stops the chain;
  - core runs `generate` then `score`;
  - report writes `gate1.md` and `gate1.json`;
  - logs go to this repo's `logs/` as `gate1_<step>_<id>.out`.
- **Where the outputs go:** `/scratch/$USER/gemma4/paper/gate1/<name>/`:
  - `prep.md` (the checks, the class counts, examples);
  - `states.jsonl`, `prompts.jsonl`, `priv.jsonl`;
  - `gen.jsonl`, `score_<context>.jsonl`;
  - `gate1.md` and `gate1.json`.
- **Resume:** generate and score skip what is written. After a time limit, `GO=1 STEPS="core report" bash gate1/submit_gate1.sh` continues.
- **After step 3:** `GO=1 NAME=gate1_off_args_2runs bash gate1/submit_gate1.sh baseline_nothink_args_v30_zip baseline_nothink_args_v30_zip_r2`.
  - `AFTER=<step 3's driver job>` holds the chain until that run has ended.
- **By hand** (one step): `WORK=<dir> STEPS=report sbatch -J gate1_report -p short -c 2 --mem=16G -t 00:30:00 gate1/gate1.slurm`.
  - GPU steps take `-p gpu --gres=gpu:a100:1 --constraint=a100@80g -c 8 --mem=128G -t 06:00:00`.
  - Not an H200: vLLM's int4 kernels can't run there (the main track's `baseline/loss_check.slurm`).

## Cost

All estimates are for about 650 states of about 10k prompt tokens each.

| Job | Where | Estimate |
|---|---|---|
| prep | CPU, short | 10–20 min (it reads about 120 MB of `calls.jsonl`) |
| core: T-think's thoughts and calls | A100 | ~0.5–1 h (prefix caching; at most 512 + 1,024 tokens per state) |
| core: score S, C-none, T-think, T-hint, note, dedup (~2,600 requests) | A100 | ~2–3 h. Prompt logprobs re-prefill every prompt, so this is about 27M prompt tokens at an assumed ~3k tok/s |
| patch: score 4 patched teachers (~2,600 requests) | A100 | ~2–3 h |

- In total, about 5–7 A100-hours, inside the plan's ~10, and no L40S time.
- The score step prints its measured prompt tokens per second, so the first job shows whether the estimate holds.

## Caveats

- **One step only.** The replay scores the next call; it can't show what the agent would do after it. That is `trigger_replay.py`'s limit too: a controller run measures the rest.
- **One sample.** T-think's own call is one sample per state; class-level rates rest on ~30–200 states each.
- **No real-model test yet.** The edits were checked on probes and stand-ins locally; prep repeats the checks with the real tokenizer and template before any GPU time.
- **The KL is a lower bound.** It is over the teacher's top 20 tokens only, so it understates. The probabilities at T=1 with top_p slightly overstate the kept tokens' share.
- **The `first` class:** the off prompt already ends with the prefilled empty thought, which the thinking-on edit removes.

## Files

| File | What it does |
|---|---|
| [gate1.py](gate1.py) | prep, generate, score and report. prep and report need only a tokenizer; generate and score need vLLM (`venvs/vllm_host`: vLLM 0.19, transformers 5) |
| [gate1.slurm](gate1.slurm) | the job: one or more steps (`STEPS`), the resources given on the command line |
| [submit_gate1.sh](submit_gate1.sh) | queues the chain, with dependencies; dry unless `GO=1` |

## Tested (locally, 2026-10-10)

No GPU and no model weights were used.
- **The stand-ins:**
  - a character-level tokenizer with Gemma 4's special tokens and the model's real `chat_template.jinja`, read from Explorer;
  - a variant of it that merges a newline with the next capital letter, and a digit or quote with a following `}`;
  - a fake `vllm` with token prompts, stop tokens (sometimes left out of `token_ids`, as vLLM may), seeds and lazy prompt logprobs;
  - synthetic runs built from 3 shards of `off_args`'s pulled traces (21 tasks, 780 calls). Their `calls.jsonl` files are rendered by the real template, carry the traces' token counts, and are shuffled, with analyzer, compaction and stray records mixed in.
- **prep:**
  - all template checks pass;
  - 778 of 780 calls align (the 2 others have no token counts in their trace);
  - the call parser agrees with the trace on 776 (the 2 misses are malformed edits with code outside the delimiters);
  - every class is populated, and every edit applies where expected.
  - With the merging tokenizer it falls back to the text edits, and the checks still pass.
  - On a thinking-on run, the check finds `<|think|>\n` where the token-level edit puts it, and leaves out the analyzer and stray records.
- **generate:**
  - thoughts that close, thoughts cut at the budget (closed with `\n<channel|>`, then a call), and replies with no thought;
  - stop tokens left out by vLLM are put back;
  - a rerun does nothing.
- **score:** all 10 contexts, resume, the skip counts, and prompts over `--max-len`.
- **report:** the gate table, the class-by-role tables, the replay with pooled and task-weighted means.
- **34 unit checks** of the sampling probabilities (T=0.2 and T=1, top_p, top_k), the top-k KL, token roles and parsing (malformed keys, code with braces and commas, arrays, empty arguments), and the edits (dedup, note, system text, thinking on).
- **`submit_gate1.sh`,** against stub `sbatch` and `squeue` and a fake scratch tree (14 checks):
  - a dry run queues nothing;
  - a missing run, gate1 jobs already queued, and core without prep's states all block;
  - `GO=1` chains prep → core (`afterok`) → report (`afterany`), and `PATCH=1` puts patch after core;
  - `AFTER` holds prep, and a resume needs no prep.
  - `bash -n` passes on both shell files.
