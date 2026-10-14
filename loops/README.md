# loops/: the loop study

<!-- written 2026-10-12, when the paper was re-scoped from OPSD to the loop behaviour itself (proposed, not decided). Numbers from Gate 1's first run (gate1_baseline_nothink_args_v30_zip) and off_args's v30 runs. Updated 2026-10-13: round 1's results, and round 2's plan (section 3). -->

**The question:** why does Gemma 4 31B, with thinking off, re-send the same tool call in a loop, and what breaks the loop?

**The working claim, narrowed after round 1 (2026-10-13): spontaneous thinking often skips the looping states where a forced thought helps.**
- Where T-think skipped the thought, a forced thought (T-open) escapes 58% of the time, against 11% (114 states in 36 tasks; paired difference +47 points, 95% interval 33 to 61). Where T-think thought, the two escape 62% and 63%.
- An escape is one step ahead: a call that is neither stale nor malformed, not yet a useful one.
- The always-block template reshapes loops over whole tasks: the stale share falls 5–6 points and runs fragment. Resolution doesn't move: 37 of 109 in both arms.
- "A stuck agent is a confident agent" stays an interpretation. The replay measures the probability of the recorded call, not calibrated confidence. `v2`'s advisor fits it: the model calls the advisor before edits, and almost never right after a bad call.
- Why loops happen, with the math, the caveats and the references: [loops_reason_10_13.md](loops_reason_10_13.md).

Everything before round 1 comes from [../gate1/README.md](../gate1/README.md) and [../baseline/README.md](../baseline/README.md). Round 1 added the two experiments below; both ran on 2026-10-12. The rule-based controller of [../adaptive_inference/](../adaptive_inference/README.md) is left out for now: it needs about 10 L40S-hours, and the main track's evals hold those slots.

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

**Result (2026-10-12):** the first reading holds: +47 points where T-think skipped the thought. The table and its limits are in [loops_reason_10_13.md](loops_reason_10_13.md) §5.

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

## 3. Round 2 (planned 2026-10-13): controlled ablations, with rollout value as a bonus

Nothing here is built or queued yet.

**Why:** round 1's claims rest on observational curves and on levers that change several things at once. The review in [loops_reason_10_13.md](loops_reason_10_13.md) names what is not yet isolated:
- the lock-in curve can come from survivor selection;
- no closed-loop arm runs at Google's T = 1;
- every lever changes meaning, length or mode along with the copy cue.

Round 2 manipulates one thing at fixed histories, at a bounded scale. The paper's claims stay one step ahead unless a whole-task effect is clear.

**The goals, in priority order:**

| | Question | Design | Size | A100-hours (est.) |
|---|---|---|---|---|
| A0 | Which failures carry usable feedback? Is the T-open gap a misallocation? | Existing files only (the bonus below) | – | 0 (CPU) |
| G1 | Do added copies raise p(copy) at a fixed history? | 1, 2 or 4 copies of one call/result block, against paraphrases of the same search and, where one exists, a different search | ~60 histories | 1–2 |
| G2 | Is the cue the result's surface match, or what it means? | 2 × 2 replay factorial with two anchors, plus one empty-thought context | ~120–160 states | 4–6 |
| G3 | Does lock-in survive T = 1? Do round 1's closed-loop findings replicate? | `ctrl` and `block` at T = 0.2, `ctrl` at T = 1; two passes each, on v31 | 109 tasks × 6 runs | 9–15 |
| G4 | (conditional) Does the best feedback rendering help over whole tasks? | G2's winner against its matched control, two passes each | 109 tasks × 4 runs | 6–10 |

- **Compute:** G1–G3 total 14–23 A100-hours. With G4, reruns and scoring fixes, reserve 30–45.
- **Basis of the estimates:** round 1's measured jobs: control 1 h 19 min and always-block 1 h 28 min for 109 tasks at 4 lanes; T-open 1 h 20 min for 581 states. G1 and G2 generate no thoughts, so they should land at the low end.
- **The binding constraints:** code, checks, queue and writing time, not GPUs.

### G1. Controlled repetition (the dose test, drafted 2026-10-13)

[dose.py](dose.py), with [dose.slurm](dose.slurm) and [submit_dose.sh](submit_dose.sh). Not yet run on Explorer.

- **Histories:** up to 60 contexts, each ending just after the first occurrence of an empty read-only search c.
  - "Empty" means nothing on stdout or stderr: exit 1 (grep found nothing), or exit 0 behind a pipe or from find.
  - "First occurrence" means c isn't repeated, and the same search wasn't written differently earlier.
  - They are picked by the history, whatever the student did next: round-robin over tasks, at most 2 per task.
  - They come from Gate 1's run (`baseline_nothink_args_v30_zip`) and round 1's `ctrl` run, which adds the rich tasks.
- **Arms:** each adds blocks that carry the identical result, so they differ only in the call.

| Arm | Blocks after the history | Tests |
|---|---|---|
| `dose1` | none: c once | the baseline |
| `dose2`, `dose4` | c again, once or three times | the dose effect |
| `para2`, `para4` | one or three paraphrases of c: swapped quotes, flag order, long flags, split flags, a dropped trailing slash, `-e`, `head -N` | the same search, another surface form. Each rule keeps grep's or find's empty output, so the result stays faithful. |
| `var2` | a different empty search the model made later in the task, with the same constant result | another search: a new pattern, or the same pattern elsewhere. Only where one exists. |

- **Why paraphrases replace the planned variant control:** real variants are scarce. In the two local shards, only 2 of 18 candidates had three different empty searches in their task, and about half had one. So `var2` runs on that subset, and `para` carries the length-, position- and failure-matched control.
- **Measures:**
  - p(c) at the next step, scored along the two ways a call starts: straight away, or after the empty thought the student writes itself (the template strips it from the history). Both at the run's sampling and at T = 1, rebuilt from the top-20 logprobs. p(the last added call) as well.
  - 3 samples per arm at the run's sampling, with every arm of a history on the history's seed. Each is judged: c exactly, the same search in any form, stale, malformed, escape, no call.
  - Paired differences with 95% intervals from resampling tasks: dose2 − dose1, dose4 − dose1, dose − para, var2 against dose.
- **Checks in prep, on the model's own template:**
  - k identical calls render as the first one plus k − 1 copies of its block;
  - a paraphrase renders as its call tokens plus the same response;
  - every picked call re-encodes to its own tokens;
  - `dedup` reads an added copy as a copy.
  A failed template check stops the chain.
- **Run it:** `GO=1 bash loops/submit_dose.sh` from `~/Gemma4_paper_track`, after `git pull`. Without `GO=1` it prints the jobs and what blocks them.
  - The chain: prep (CPU), then two chained A100 jobs (2-h limit each; the second finishes what the first left), then the report.
  - Expect about 1 h of one A100 for 60 histories. That is an estimate: ~1,000 scored contexts of ~12k tokens, plus 3 samples for each of ~360 contexts.
  - Outputs: `$R/paper/loops/dose_baseline_nothink_args_v30_zip/`, with `prep.md`, `score.jsonl`, `samples.jsonl` and `dose.md`.
  - It runs on v31's serving env, and every arm is scored there.
- **Precedents for the design:**
  - Holtzman et al. and Xu et al. repeat a phrase or sentence a chosen number of times in a constructed input.
  - Sinha et al. inject errors into fixed histories at chosen rates.
  - Liu et al.'s fixed-history probe measures p(3rd identical action).
- **Reading it:**
  - dose2 and dose4 above dose1, with para below dose: copies of the identical call raise the next copy at a fixed history.
  - para matching dose: repeating the same search in any form does it, or the added length and failures do.
  - Neither moving p(c): the lock-in curve's rise is selection. Drop the reinforcement wording, and keep the observed persistence.

### G2. What the result means, against its surface form

| | Neutral text | Faithful interpretation |
|---|---|---|
| Repeated results keep matching surface forms | placebo | semantic |
| Repeated results get varying arbitrary IDs | surface break | both |

- **Anchors:** the untouched original and dedup, scored again in the same environment.
- **States:** picked by history. The last call repeats an earlier one and got the identical result.
  - The primary subset holds verified deterministic, read-only searches, with the relevant files unchanged.
  - States are spread over tasks, with some first-failure states (`after_miss`) as controls.
  - An `ok` with no output is not automatically a failed search.
- **Renderings:**
  - Apply each rendering to every relevant result in the history, the same way each time.
  - The semantic text is a faithful reading of the result, such as "The search completed and found zero matches for this pattern in these files." It carries no instruction ("change approach").
  - Call strings stay unchanged. Match the added tokens' lengths and positions across cells.
  - A changing repeat counter changes both meaning and match, so it stays out of the factorial.
- **Measures:**
  - exact copies, valid non-stale calls, malformed calls, and changes of approach;
  - 3 continuations per arm. They sharpen each state's estimate but add no tasks, so compare in pairs, with task-cluster intervals.
- **One extra context:** keep the empty thought before every past call, as well as at the end.
  - If the block works by breaking the match between the result and the call, restoring that match in the history should bring the copy back.
  - If it works through the thinking mode ([Google's formatting notes](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4); HF PR #147), restoring it shouldn't.
- **Reading it:**
  - IDs help: the identical surface form contributes.
  - Meaning helps beyond neutral text: interpretable feedback contributes.
  - Both help: report the interaction.
  - Nothing reliable: narrow the claim rather than sweep wordings.

### G3. Temperature and replication

- **Arms:** `ctrl` (T = 0.2), `block` (T = 0.2) and `ctrl_t1` (T = 1.0; Google's top-p 0.95 and top-k 64). Each gets two passes on the same 109 tasks.
- **Environment:**
  - The serving env was rebuilt for v31 on 2026-10-13 (`venvs/vllm_host`), so round 1's v30 runs can't be repeated exactly.
  - Run all six on v31, and keep round 1 as a separate replicate.
  - Record, for every run: wheelhouse, serving env, template sha, zip sha, sampling, and budgets.
- **What it needs:**
  - a zip that differs from `off_args`'s only in `configs/sampling.yaml` (temperature 1.0);
  - a third arm in [arms.tsv](arms.tsv) and [submit_loops.sh](submit_loops.sh). The script now refuses arms with different zips.
- **Measures:**
  - fragmentation and re-entry of runs;
  - stale rates, per task and pooled;
  - malformed calls, source edits, resolved tasks, time and tokens;
  - loops by type (search, malformed, termination), and what escapes lead to.
- **Reading it:** the sample targets large behavioural effects. It can't establish a change of 1–2 resolved tasks, and an inconclusive comparison is not equivalence.

### G4. Conditional: feedback over whole tasks

- **When:** only if G2's semantic rendering has a clear, reproducible effect beyond placebo.
- **How it runs:** the rendering may use only history already observed.
  - The always-block arm's route, a chat template, can carry it if the template can express the rendering.
  - Otherwise it needs a harness change, plus smoke checks before full runs.
- **Measures:** executed-call usefulness, loop re-entry, time and resolution, not only fewer exact copies.
- If G2 is weak, or integration threatens writing time, drop G4 and keep the paper's claims one step ahead.

### Bonus: what makes a failed rollout useful (CPU only)

[loops_reason_10_13.md](loops_reason_10_13.md) §9 has the argument: distinct failures with interpretable, attributable feedback can teach, while loops spend the rollout budget on redundant states. Round 2 measures it on files that already exist, with no training:
- **Redundancy:** in round 1's closed-loop runs, the share of failed tasks' calls and output tokens spent on stale repeats. Also the distinct (call, result) states per failed rollout.
- **Feedback types at failures:**
  - the types: verified search misses (exit 1, no output); ambiguous empty successes (exit 0, no output, as `grep … | head` gives); explicit errors with a message; repeated non-empty results;
  - for each, how often a copy follows;
  - explicit errors loop too: the control run of `fastapi_12942` repeats an `edit_file` 35 times against "No changes to apply. old_string and new_string are identical."
- **Directed teacher disagreement:**
  - In Gate 1's score files (T-hint, T-think, T-open, C-none, note): does the teacher lower the stale call's probability at its decision tokens, the tool name and the first content tokens? How concentrated is the change? Split by feedback type.
  - A large KL alone doesn't show the correction points the right way, and the top-20 approximation applies.
- **A no-thought baseline for T-open:**
  - From `score_S.jsonl`, compare the recorded call's probability where T-think thought with where it skipped.
  - 1 − p(the recorded stale call) is an upper bound on escape without a thought, so it bounds what a thought gains in each group.
- **Where to run it:** the score files are small (S 13 MB, the others under 1 MB). Copy them and run locally, not on the login node.

**Out of scope for round 2:**
- localising the mechanism to attention heads;
- other models. A 12B `ctrl` run on v31 is the cheapest scale check, but any one-step claim would need its own replay states;
- broad grids over temperature or thinking budget;
- sweeps over note wordings;
- a failure-triggered thinking controller;
- OPSD or LoRA training.

**Schedule:** the deadline is November 12, 2026, 23:59 UTC, from Kaggle's timeline as read on 2026-10-13. The paper is capped at 3,000 words.

| Dates, 2026 | Work |
|---|---|
| Oct 14–16 | A0; the code for G1–G3, tested locally against a fake vLLM, as in round 1 |
| Oct 17–22 | G1–G3 runs and paired analysis |
| Oct 23–26 | G4, only if G2 warrants it |
| Oct 27 | Freeze the experiments |
| Oct 27–Nov 5 | Write the paper; reproducible figures and code |
| Nov 6–10 | Review, corrections, early submission |

- Delays cut scope, not writing time.
- Every job stays on one A100, under the 2-h limit. Ask before anything multi-GPU or longer.
- The user commits and submits.

**What it refuses:**
- loops jobs already queued (`FORCE=1` overrides);
- a main-track checkout without `CHAT_TEMPLATE`;
- Kaggle on another wheelhouse than ours;
- missing id files, zip, model or venv;
- more jobs than the `gpu` partition lets us queue.

**Reading it:** trace metrics first, as everywhere in this paper. Look at stale repeats, copy starts, lock-in at k = 2 and runs of 5+. Resolved tasks come second: one run varies by ±2.2. The "about 2 of 70 tasks" that removing every loop would gain is an observed cross-run rescue estimate (6 of 23 loop failures were solved by at least one of 9 other runs), not an upper bound.

**Result (2026-10-12):** the block lowers the stale share and fragments the runs, but more runs start, and resolution is unchanged (dev 25 → 24, rich 12 → 13). The full table is in [loops_reason_10_13.md](loops_reason_10_13.md) §1.

## Files

| File | What it does |
|---|---|
| [make_template.py](make_template.py) | The always-block template from the model's own, and its 36 checks |
| [loop_report.py](loop_report.py) | Loop metrics of any runs: by task group, with the template's effect on the prompts. stdlib only |
| [arms.tsv](arms.tsv) | The two arms, in [../baseline/arms.tsv](../baseline/arms.tsv)'s format |
| [loops.slurm](loops.slurm) | The CPU steps: `STEP=template` or `STEP=report` |
| [submit_loops.sh](submit_loops.sh) | Queues the chain; dry unless `GO=1` |
| [dose.py](dose.py) | Round 2's G1, the dose test: `prep`, `score`, `sample`, `report`. Imports [../gate1/gate1.py](../gate1/gate1.py) |
| [dose.slurm](dose.slurm) | The dose test's steps in one job: CPU for prep and report, one A100 for score and sample |
| [submit_dose.sh](submit_dose.sh) | Queues prep → 2 GPU jobs → report; dry unless `GO=1` |
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

## Tested: the dose test (locally, 2026-10-13)

No GPU, and nothing queued on Explorer. The test data sits in a session scratchpad, outside this repo: the model's tokenizer and chat template (with the `tokenizer_config.json` that transformers 4.57 can read), and 2 shards of `off_args`'s first v30 run (14 tasks).
- **prep, on the real shards:**
  - all 7 template checks pass;
  - it picks 12 histories in 7 tasks (2 per task at most). All 12 have three paraphrases, and 3 have a different empty search.
- **74 checks against a fake vLLM.** The fake copies the last call more readily the more identical copies end the context.
  - **Reruns:** a task rerun in a later shard is read from there, unless it was skipped at the deadline, as loop_report.py reads runs.
  - **Paraphrases:** every one keeps its search under the normaliser, and none repeats a command already in the context.
    - An unquoted grep pattern keeps its trailing slash.
    - Quotes aren't swapped around `$`.
  - **What counts as an empty read-only search:**
    - stderr, exit 2 and a budget warning all disqualify a result;
    - writes, `sed -i` and scripts aren't searches.
  - **The contexts of each real history:**
    - `dose2` is the history plus one copy of its last block, and `dose4` plus three;
    - `dedup` turns `dose4` back into the history;
    - each paraphrase and variant parses back to its command.
  - **score:**
    - one request per history, arm, target and path;
    - prefix caching off, prompt logprobs 20;
    - each request is the arm's context, the path, then the target;
    - a rerun does nothing.
  - **sample:**
    - one request per history and arm, n = 3 at T = 0.2, top-p 0.95, top-k 64, stopping at `<tool_call|>`;
    - every arm of a history shares the history's seed;
    - a rerun does nothing.
  - **The judge, on hand-made continuations:** an exact copy, a copy after the empty thought, a paraphrase, a fresh call, a malformed call and no call.
  - **The report recovers the fake's built-in effect.**
    - p(c) at T = 1 is 0.05, 0.40 and 0.80 for one, two and four copies.
    - At T = 0.2 it sharpens as the formula says: 0.40 becomes 0.116.
    - para stays low, and the paired dose4 − dose1 interval is above 0.
    - The same seed gives the same report.
- **`submit_dose.sh`:** 20 cases against stub `sbatch` and `squeue` and a fake scratch tree.
  - The chain and its dependencies: prep `afterok`, then the GPU jobs and the report `afterany`.
  - Resources, `JOBS`, `ARMS` and `PREP_ARGS`, `AFTER`, and runs given as arguments.
  - Every refusal: missing runs, `calls.jsonl`, model, venv or main checkout; no `states.jsonl` without prep; an unknown step; dose jobs already queued (`FORCE=1` overrides).
- `bash -n` passes on `dose.slurm` and `submit_dose.sh`.
- **Not run:**
  - a real vLLM;
  - prep under transformers 5;
  - the `ctrl` run's shards, whose layout loop_report.py already reads;
  - anything with Explorer's paths.
