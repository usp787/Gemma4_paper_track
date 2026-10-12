# loops/: the loop study

<!-- written 2026-10-12, when the paper was re-scoped from OPSD to the loop behaviour itself (proposed, not decided). Numbers from Gate 1's first run (gate1_baseline_nothink_args_v30_zip) and off_args's v30 runs. -->

**The question:** why does Gemma 4 31B, with thinking off, re-send the same tool call in a loop, and what breaks the loop?

**The working claim: a stuck agent is a confident agent.**
- At a third or later identical copy, the student gives the copy p = 0.94.
- Allowed to think, the same weights write a real thought at 72% of second copies, but only 23% of third-or-later copies and 6% of malformed calls. Where they skip it, their own call is stale 97% of the time.
- When they do think at a stuck state, they escape 8 times in 14.
- So the model's own choice of when to think fails exactly where thinking is needed. `v2`'s advisor shows the same pattern: the model calls it before edits, and almost never right after a bad call.

Everything comes from [../gate1/README.md](../gate1/README.md) and [../baseline/README.md](../baseline/README.md). This folder adds the two experiments below. The rule-based controller of [../adaptive_inference/](../adaptive_inference/README.md) is left out for now: it needs about 10 L40S-hours, and the main track's evals hold those slots.

| | 1. The forced thought (T-open) | 2. The always-block run |
|---|---|---|
| Tests | Does a real thought break the copy where the model would not think on its own? | Does the 4-token empty thought, forced at every call, cut loops over whole tasks rather than one step ahead? |
| Where | Gate 1's work dir, one step ahead ([gate1.py](../gate1/gate1.py), report section 4) | Two closed-loop arms: `off_args` with the always-block chat template (`block`) and with the model's own (`ctrl`) |
| Tasks | Gate 1's 581 states in 70 tasks | 70 dev-sound + 39 rich-sound tasks = 109 |
| GPU | 1 A100 job, about 1.3 h (2-h limit) | 2 A100 jobs side by side, about 1.7 h each (2 × 2-h limit per arm); no L40S |
| Command | `GO=1 STEPS="open report" bash gate1/submit_gate1.sh` | `GO=1 bash loops/submit_loops.sh` |

Run both from `~/Gemma4_paper_track` on the login node, after `git pull` there and in `~/Gemma4_kaggle_repo`. Experiment 2 needs the main track's `baseline/eval_gpu.slurm` with `CHAT_TEMPLATE`, which is in the same change. Without `GO=1`, each command prints its jobs and what blocks them.

## 1. The forced thought

**Why:** T-think decides for itself whether to think, so its result mixes two things: what a thought does, and where the model chooses to think. At third-or-later copies, it skipped the thought 77% of the time. T-open removes the choice.

**What it does,** at every Gate 1 state:
- the thinking-on prompt, then `<|channel>thought\n`, so the thought is already open;
- vLLM's `min_tokens` keeps `<channel|>` (and every other stop token) back for the first 32 tokens. 32 is below 98% of the thoughts T-think wrote on its own (minimum 28, 5th percentile 57, median 244);
- otherwise as T-think: at most 512 thought tokens, the run's sampling (T = 0.2, top_p 0.95, top_k 64), then its own call, judged the same / stale / malformed;
- then `score` puts the student's own call after that thought: the replay's p(call).

**What it writes:** `gen_T-open.jsonl` (thoughts, with their text, and calls) and `score_T-open.jsonl`, in the work dir. `gate1.md` then gains:
- a T-open row in section 1's gate table;
- **section 4:**
  - by class: T-think's thought rate; p(call) under S, T-think and T-open; both own calls; the escape rates;
  - the self-selection: the looping states where T-think skipped the thought against those where it thought, for both;
  - on the 116 looping states where block applies, the escape rates of block, C-none, T-think and T-open. On the first run, block reached 28%, C-none 32% and T-think 13%;
  - two example thoughts at third-or-later copies and returns.

**Reading it:**
- If T-open escapes far more than T-think at the states T-think skipped, the model can fix the loop when made to think; it just doesn't choose to. That is the main claim's test.
- If T-open escapes about as rarely, the loop survives a real thought too. Then the paper's lever is the context (dedup, a note), not thinking.

**One fix it brings:** `seed_of` now gives each state its own seed. The old one kept only 4 bytes of the key, so the first run's 581 states had 75 seeds, about two per task. T-think's and own's samples there share random numbers within a task. That doesn't bias their rates, but their states are less independent than their counts suggest. Rows already written keep their seeds.

## 2. The always-block run

**Why:** the forced empty thought halved the probability of a third copy in the replay (0.95 → 0.50). But 80% of the sampled calls there were still stale, and the replay looks only one call ahead. Only a whole-task run shows whether the block cuts loops, and what it costs elsewhere. The 10-11 notes flagged `after_miss`: block's sample was stale 20% of the time there, against 4% for T-think without a thought.

**The template** ([make_template.py](make_template.py)):
- With thinking off, Gemma 4's template prefills the empty thought only when it opens a model turn, after the task prompt.
- The edit adds three lines that prefill it after a tool response too.
- The check renders 9 probe conversations with the old and the new template, thinking off and on, with and without a generation prompt (36 checks). After a tool response with thinking off, the new prompt must be the old one plus the 4 tokens. Everywhere else it must be unchanged.
- vLLM's gemma4 reasoning parser treats a prompt that ends in `<channel|>` as reasoning already ended. That's the same path every task's first call takes, so tool calls parse as before.
- The template applies to every thinking-off call the server gets, the analyzer sub-agent's included: it shares the main agent's sampling, so thinking is off there too.
- It can't reach Kaggle: a zip can't change the chat template. There, the block could only come through the weights, as a LoRA that learns the 4 tokens.

**The arms** ([arms.tsv](arms.tsv)):
- Both run `off_args`'s pinned zip (`532374430fe2`), PINS=1, unpaced, 4 tasks at once on one A100 (`eval_gpu.slurm`'s resume mode).
- `block` adds `CHAT_TEMPLATE`. `ctrl` is the same without it, in the same setting, so the pair differs only in the template.
- Compare `block` with `ctrl`, not with `off_args`'s L40S runs (24 and 22 of 70), which ran one task at a time.

**The chain** ([submit_loops.sh](submit_loops.sh)):
1. **template** (CPU, ~5 min): builds and checks the template. A failed check cancels everything after it.
2. **ctrl and block** (A100, side by side): two chained jobs per arm. The second finishes what the first left, or exits before starting vLLM.
3. **report** (CPU): writes `$R/paper/loops/report_p1_<time>/`:
   - `baseline_dev.md` and `baseline_rich.md`: [../baseline/baseline_report.py](../baseline/baseline_report.py) with this arms file. Resolved per run, `block` against `ctrl` task by task with a bootstrap CI and a sign test, call hygiene, output tokens and the thinking share;
   - `loops.md`: [loop_report.py](loop_report.py). Copy starts, lock-in, runs of 3+ and 5+, loop tasks, how each run of 3+ ends, and what the template did: the share of main-agent prompts that end with the empty thought, and of completions that open one, by the call's place in the task.

**Options:**
- `SERIAL=1` runs one A100 at a time (~3.5 h in all).
- `PASS=2` runs a second seed of both arms (`_r2`), and its report reads both passes.
- `STEPS=report` writes the report again.
- The header of [submit_loops.sh](submit_loops.sh) has the rest.

**What it refuses:**
- loops jobs already queued (`FORCE=1` overrides);
- a main-track checkout without `CHAT_TEMPLATE`;
- Kaggle on another wheelhouse than ours;
- missing id files, zip, model or venv;
- more jobs than the `gpu` partition lets us queue.

**Reading it:** trace metrics first, as everywhere in this paper. Look at stale repeats, copy starts, lock-in at k = 2 and runs of 5+. Resolved tasks come second: the ceiling says removing every loop gains about 2 of 70 tasks, and one run varies by ±2.2.

## Files

| File | What it does |
|---|---|
| [make_template.py](make_template.py) | The always-block template from the model's own, and its 36 checks |
| [loop_report.py](loop_report.py) | Loop metrics of any runs: by task group, with the template's effect on the prompts. stdlib only |
| [arms.tsv](arms.tsv) | The two arms, in [../baseline/arms.tsv](../baseline/arms.tsv)'s format |
| [loops.slurm](loops.slurm) | The CPU steps: `STEP=template` or `STEP=report` |
| [submit_loops.sh](submit_loops.sh) | Queues the chain; dry unless `GO=1` |
| [lit_review_prompt.md](lit_review_prompt.md) | The start prompt of the literature-review session |

The forced thought lives in [../gate1/](../gate1/README.md): `gate1.py --context T-open`, the step `open` of `gate1.slurm` and `submit_gate1.sh`.

## Tested (locally, 2026-10-12)

No GPU, and nothing queued on Explorer. The test data was copied to a scratch dir outside this repo: Gate 1's work dir, the model's tokenizer and chat template, and 2 shards of `off_args`'s first v30 run.
- **The report, unchanged where T-open is absent:** on the copied work dir, the new `gate1.md` is identical to Explorer's, apart from line endings.
- **T-open with a fake vLLM,** on 40 real states (4 per class), the real tokenizer and the real prompts (24 checks):
  - stage 1 asks for `min_tokens` 32, `max_tokens` 509, the run's sampling and one seed per state, after the thinking-on prompt plus `<|channel>thought\n`;
  - the thoughts come back closed, cut at the budget, or cut at a call written inside them;
  - stage 2 puts the call after the whole thought, and the calls are judged as T-think's;
  - a rerun does nothing;
  - score puts the student's completion after the thought, token for token;
  - fed T-think's rows as T-open's, every section-4 column equals T-think's, and sections 1-3 don't change;
  - T-think's own generate writes the same rows as the committed code, given the new seed.
- **The template:**
  - all 36 checks pass;
  - its diff from the model's template is the 3 inserted lines;
  - its sha256 (`5f875818d2a2`) is the same on any OS;
  - a template whose generation prompt has changed is refused, and nothing is written.
- **`loop_report.py`:**
  - on the 2 real shards, its stale, malformed and chain numbers equal `trigger_replay.py`'s;
  - lock-in is 83% at k = 2 and 93% at k = 3+, as the Gate 1 README has it;
  - the block drift matches the 10-11 notes (86% of calls 2-5 write the block, 26% after call 20);
  - on a synthetic always-block copy, 100% of prompts end with the block and 0% of completions open one;
  - groups, a missing run dir and the pairing work.
- **`submit_loops.sh`,** against stub `sbatch`, `squeue`, `scontrol` and `sacctmgr` and a fake scratch tree (14 cases):
  - the chain and its dependencies, where each job is submitted from, and `ctrl`'s empty `CHAT_TEMPLATE` against `block`'s path;
  - `SERIAL`, `PASS=2`, `STEPS=report`, `AFTER` and `FORCE`;
  - every refusal above, and an unknown step.
- **`submit_gate1.sh`'s `open`:** it is refused without `states.jsonl`; otherwise it queues `open` on an A100, then `report` (`afterany`).
- `bash -n` passes on every changed shell file, the main track's `eval_gpu.slurm` included.
- **Not run:** `eval_gpu.slurm` with `CHAT_TEMPLATE` (its paths are Explorer's); `make_template.py` under transformers 5 (here 4.57, with a test copy of `tokenizer_config.json` that 4.57 can read); a real vLLM.
