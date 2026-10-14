# gate1/: Gate 1 of the OPSD study

<!-- written 2026-10-10; first run 2026-10-11 (below), then the context block and the step own added. Trace numbers from off_args's first v30 run, pulled 2026-10-10 -->

Gate 1 is the go/no-go of [../opsd_assessment.md](../opsd_assessment.md#14-what-to-change) (§1.4, change 1). It scores recorded trajectories of the thinking-off student, with no training and no new rollouts, and answers two questions:
1. **The gate:** does some teacher put clearly more KL on the student's failure tokens than on its routine tokens? If none does, the fallback paper starts.
2. **The replay:** at the student's own stuck states, what breaks the copy?
   - thinking: T-think's own next call, and the copy's probability after T-think's thought;
   - the thinking-on template with a forced empty thought (C-none), and that empty thought alone, with thinking off (block);
   - a controller's note;
   - fewer copies in the context;
   - a higher temperature.

   That is the premise of the rule-based controller ([../adaptive_inference/README.md](../adaptive_inference/README.md)) and of the temperature fix, measured one step ahead at the exact states, without an L40S run.

By default it scores `off_args`'s first v30 run, `baseline_nothink_args_v30_zip`. Add `_r2` once step 3 has run.

## First run (2026-10-11)

`gate1_baseline_nothink_args_v30_zip`: 581 states in 70 tasks, without `PATCH=1`. prep's checks passed. Jobs 10991390–92; the core job took 3 h 16 min on one A100. Its `gate1.md` is in the work dir.
- **The gate:** T-think fails (ratio 0.51, 95% CI 0.28–0.96), and C-none passes (2.82, 1.08–9.41).
- **T-think thinks where the student is unsure, and rarely inside a locked-in loop.**
  - It wrote a thought at 72% of `copy_2nd` states and 23% of `copy_3rd_plus` ones.
  - Its own call was a stale repeat at 45% and 83% of them. With a thought, 33% and 36%; without one, 75% and 97%.
- **C-none's effect comes from its forced empty thought, not from the thinking-on template.** This is from a follow-up on the work dir's files:
  - At the 36 `copy_3rd_plus` states where T-think went straight to the call, the copy's probability is:
    - 0.99 for the student;
    - 0.97 with thinking on and no block, T-think's own path there;
    - 0.39 with the block forced, which is C-none.
  - Where the student wrote the block itself, C-none's KL is near 0 (0.004 per token).
- **The per-token gate is sensitive to call length.** On the 292 states where the student went straight to the call:
  - C-none's ratio is 0.96 per token (0.34–4.36), but 3.77 per call, each call at its largest token KL (1.61–12.68).
  - Long routine edits with low KL dominate the per-token mean.
- **Next: the context `block`** forces the same empty thought with thinking off, to test whether the block alone breaks the copy. The step `own` samples what the model does there instead. One resume of the same work dir runs both ([Commands](#commands)).
- **Then (2026-10-12): the forced thought, T-open.** It is T-think with the thought already open, which can't close before 32 tokens: does a real thought break the copy where the model wouldn't think on its own? On the looping states, T-think's own calls split by its choice: where it skipped the thought (114 states), 89% were stale; where it thought (99), 62% escaped. The step `open` runs it on this work dir, and the report gains section 4. The why and how are in [../loops/README.md](../loops/README.md), the loop study this belongs to now.

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
  - **Ceiling:** of the 23 failures with at least 20% stale calls, only 6 were solved by any of 9 other runs (v28–v30). So removing every loop would gain about 2 tasks. That is an observed rescue estimate across runs, not an upper bound: a new intervention could solve tasks none of those runs did.
  - Loops are where thinking-off fails visibly, and where its behaviour differs most from thinking on.
- **Prior work agrees.**
  - Repetition reinforces itself (Holtzman et al., ICLR 2020; Xu et al., NeurIPS 2022).
  - Models err more once their own errors are in context (Sinha et al., ICLR 2026). On their synthetic, tool-free execution task, Qwen3's thinking models no longer did, perhaps because earlier thoughts are dropped from the history. Chain-of-thought prompting didn't stop it.
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
| C-none | the thinking-on template (`<|think|>` in the system turn), then an empty thought, forced | planned as the template's effect alone; the first run found the forced block does the work |
| block | the recorded prompt (thinking off), then the same empty thought, forced where the student went straight to the call. Where it wrote the block itself, or the prompt ends with the template's own (the first call), block would be S, so it isn't scored there | C-none without the thinking-on template: is it the block? Also a prefill a controller could make |
| T-think | the thinking-on template, then a thought the same weights write there (at most 512 tokens, T=0.2) | **the proposed teacher**, and the controller's premise |
| T-open | as T-think, but the thought is already open (`<|channel>thought\n`) and can't close before 32 tokens (vLLM's `min_tokens`) | a real thought at every state, without T-think's choice of where to think (step `open`, report section 4) |
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
- the forward KL over the teacher's top 20 tokens. A token missing from the student's top 20 gets the student's lowest listed logprob, which understates that token's term. The dropped tail can add a negative term, so the sum is not a bound either way.

**Per call:** the probability of the student's whole call (`<|tool_call>` to `<tool_call|>`):
- at T=1, raw;
- at the run's sampling (T=0.2, top_p 0.95, top_k 64);
- at T=1 with top_p 0.95, which is Gemma's prescribed sampling.

The last two are computed from the top 20 logprobs.

**T-think's own call:** one sample after its thought, checked three ways:
- the student's call exactly;
- a stale repeat in that context;
- malformed.

**Own calls (the step `own`):** one sample after block's prompt and one after C-none's, checked the same three ways. Both use the state's seed, so the two contexts draw the same random numbers.

**Section 3 of the report: the forced empty thought.**
- On the states where block applies, by class: the probability of the student's call under S, block and C-none, and both own calls.
- The same states where T-think also went straight to the call: thinking off or on, without or with the forced block.
- The gate's KL ratio for block and for C-none on these states, per token and per call. "Per call" counts each call once, at its largest token KL.

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
STEPS="core report" CORE_TIME=02:00:00 bash gate1/submit_gate1.sh   # dry run of the resume below
GO=1 STEPS="core report" CORE_TIME=02:00:00 bash gate1/submit_gate1.sh   # adds block and own to the first run's work dir
GO=1 STEPS="open report" bash gate1/submit_gate1.sh   # adds the forced thought, T-open, to the first run's work dir
```

- **The forced thought** (2026-10-12) needs only the first run's work dir. Its job runs `generate --context T-open`, then `score` with S and T-open. It writes `gen_T-open.jsonl` and `score_T-open.jsonl`, and the report adds T-open to the gate table and writes section 4. `OPEN_ARGS="--min-thought 64"` changes the minimum.

- **The resume of the first run's work dir** (2026-10-11): generate, the other contexts' scores and prep are all done, so core only scores block (292 states) and runs `own` (292 + 581 calls), then the report rewrites `gate1.md` with section 3. Sections 1 and 2 come out unchanged, apart from the "Scored:" line.
- **What the jobs do:**
  - prep checks the prompt edits against the model's chat template and against `on`'s recorded thinking-on prompts (`base_v30_l40s`); a failed check stops the chain;
  - core runs `generate`, `score`, then `own`;
  - report writes `gate1.md` and `gate1.json`;
  - logs go to this repo's `logs/` as `gate1_<step>_<id>.out`.
- **Where the outputs go:** `/scratch/$USER/gemma4/paper/gate1/<name>/`:
  - `prep.md` (the checks, the class counts, examples);
  - `states.jsonl`, `prompts.jsonl`, `priv.jsonl`;
  - `gen.jsonl`, `score_<context>.jsonl`, `own_<context>.jsonl`;
  - `gate1.md` and `gate1.json`.
- **Resume:** generate, score and own skip what is written. After a time limit, `GO=1 STEPS="core report" bash gate1/submit_gate1.sh` continues.
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
| core: score block, then own (~290 + ~870 requests) | A100 | ~1 h (estimate): block at the first run's ~4.4 s per state, own at about T-think's 2.2 s per state or less, since it writes no thought |
| patch: score 4 patched teachers (~2,600 requests) | A100 | ~2–3 h |
| open: T-open's thoughts and calls, then score T-open (581 + 581 requests) | A100 | ~1.3 h (estimate): a thought at every state, then ~6M prompt tokens at the first run's ~2,000 tok/s |

- In total, about 6–8 A100-hours, inside the plan's ~10, and no L40S time.
- The score step prints its measured prompt tokens per second. On the first run it was ~2,000 tok/s, and the core job took 3 h 16 min.

## Caveats

- **One step only.** The replay scores the next call; it can't show what the agent would do after it. That is `trigger_replay.py`'s limit too: a controller run measures the rest.
- **One sample.** T-think's own call and the own calls are one sample per state; class-level rates rest on ~30–200 states each.
- **Seeds shared within a task (the first run).** Until 2026-10-12 the seed kept only 4 bytes of the state's key, so the first run's 581 states had 75 seeds, about two per task. Its T-think and own samples share random numbers within a task: their rates aren't biased, but the states are less independent than the counts suggest. Since then each state has its own seed, T-open's included; rows already written keep theirs.
- **Real-model tests.** The first run's prep passed every check with the real tokenizer and template. block and own are tested on the real tokenizer and the first run's prompts with a fake vLLM, not yet on a GPU.
- **The KL is approximate, not a lower bound** (corrected 2026-10-13). It covers the teacher's top 20 tokens only. The floor for missing student tokens lowers the kept terms, but the dropped tail's terms can be negative, so it can over- or understate the full KL ([../loops/loops_reason_10_13.md](../loops/loops_reason_10_13.md) §9). The probabilities come from the top 20 while the run samples from the top 64, so at T=1 with top_p they can overstate the kept tokens' share; that error isn't bounded.
- **The `first` class:** the off prompt already ends with the prefilled empty thought, which the thinking-on edit removes.

## Files

| File | What it does |
|---|---|
| [gate1.py](gate1.py) | prep, generate, score, own and report. prep and report need only a tokenizer; generate, score and own need vLLM (`venvs/vllm_host`: vLLM 0.19, transformers 5) |
| [gate1.slurm](gate1.slurm) | the job: one or more steps (`STEPS`), the resources given on the command line |
| [submit_gate1.sh](submit_gate1.sh) | queues the chain, with dependencies; dry unless `GO=1` |

## Tested (locally, 2026-10-11): block and own

No GPU was used, and nothing was written on Explorer. The first run's work dir and the model's tokenizer files were copied to a scratch dir outside this repo.
- **The report**, on the copied files:
  - With no block files, the new `gate1.md` and `gate1.json` are byte-identical to the old script's. The old script reproduces Explorer's `gate1.md`.
  - With synthetic block and own files, section 3 reads them right: 4 invariants hold, such as a column fed S's numbers equalling S's.
  - The new KL helper reproduces the gate's C-none row: KLs, ratio and CI.
- **On the real tokenizer** (its `tokenizer.json`, through `tokenizers`) and the first run's 581 prompts:
  - block applies to exactly the 292 states where the student went straight to the call and the prompt doesn't end with the template's block. The 35 prompts that do are the first calls.
  - All 254 blocks the student wrote itself are the forced one, token for token: `<|channel>thought\n<channel|>`.
  - block's sequence is the recorded prompt, the block, then the student's completion. On these states, C-none differs from it only by `<|think|>\n`.
  - The longest block prompt is 15,317 tokens.
- **With a fake vLLM:**
  - own writes one row per state where each context applies.
  - The student's own call comes back as `same_as_student`, with the stop token restored where vLLM leaves it out, and its stale flag agrees with the student's.
  - own samples at the run's T, top_p and top_k, stops at `<tool_call|>` after at most 1,024 tokens, and uses the state's seed, the same in both contexts. A rerun does nothing, and other contexts are refused.
  - generate, with the call check now shared with own, writes the old script's rows exactly (48 states: thoughts closed, cut and absent).
  - score writes block rows only where it applies, aligned with S's.
- **`submit_gate1.sh`,** against stub `sbatch` and `squeue`:
  - the resume queues core (`generate score own`, block among the contexts) on a 2-h limit, then report (`afterany`);
  - a fresh chain still runs prep → core (`afterok`) → report.
  - `gate1.slurm` itself wasn't run, since its paths are Explorer's. `bash -n` passes on both shell files.

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
