# loops/: the loop study

<!-- written 2026-10-12, when the paper was re-scoped from OPSD to the loop behaviour itself (proposed, not decided). Numbers from Gate 1's first run (gate1_baseline_nothink_args_v30_zip) and off_args's v30 runs. Updated 2026-10-13: round 1's results, and round 2's plan (section 3). Updated 2026-10-14: G1's result, and G2 drafted (section 3). Updated 2026-10-15: G2's result; G1's and G2's records are in ablation_G1.md and ablation_G2.md. -->

**The question:** why does Gemma 4 31B, with thinking off, re-send the same tool call in a loop, and what breaks the loop?

**The working claim, narrowed after round 1 (2026-10-13): spontaneous thinking often skips the looping states where a forced thought helps.**
- Where T-think skipped the thought, a forced thought (T-open) escapes 58% of the time, against 11% (114 states in 36 tasks; paired difference +47 points, 95% interval 33 to 61). Where T-think thought, the two escape 62% and 63%.
- An escape is one step ahead: a call that is neither stale nor malformed, not yet a useful one.
- The always-block template reshapes loops over whole tasks: the stale share falls 5–6 points and runs fragment. Resolution doesn't move: 37 of 109 in both arms.
- Lock-in is partly self-reinforcement (round 2's G1, 2026-10-14). At fixed histories, identical copies raise the next copy: p(c) is 0.002, 0.098 and 0.434 for 1, 2 and 4 copies, while re-spelled copies of the same search stay near 0.03. But four forced copies give 43%, while in the closed loop a run of three or more gets one more copy 96% of the time. So the observed curve also reflects which histories loop.
- The results' content doesn't cue the copy (round 2's G2, 2026-10-15). Varying IDs and a faithful "zero matches" note leave p(c) where it was. Deleting the earlier copies (dedup) or ending the context unlike them (the block) cuts it. With the empty thought before every earlier call too, the copy comes back in full, so the block works as a pattern break ([ablation_G2.md](ablation_G2.md)).
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
- Since 2026-10-15 the script also queues G3's T = 1 arm and its zip checks by default (§3). `STEPS="template ctrl block report"` queues round 1's two arms alone.

## 3. Round 2 (planned 2026-10-13): controlled ablations, with rollout value as a bonus

G1 ran on 2026-10-14 and G2 on 2026-10-15. Each section ends with its result; the full records are [ablation_G1.md](ablation_G1.md) and [ablation_G2.md](ablation_G2.md). G3 is drafted and tested locally (2026-10-15), not yet run. Nothing else here is built or queued yet.

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
| G2 | Is the cue the result's surface match, or what it means? | 2 × 2 replay factorial with two anchors, plus one empty-thought context | ~120–160 states (drafted: 133, in 23 tasks) | 4–6 (drafted: ~2.5) |
| G3 | Does lock-in survive T = 1? Do round 1's closed-loop findings replicate? | `ctrl` and `block` at T = 0.2, `ctrl` at T = 1; two passes each, on v31 | 109 tasks × 6 runs | 9–15 |
| G4 | (conditional) Does the best feedback rendering help over whole tasks? | G2's winner against its matched control, two passes each | 109 tasks × 4 runs | 6–10 |

- **Compute:** G1–G3 total 14–23 A100-hours. With G4, reruns and scoring fixes, reserve 30–45.
- **Basis of the estimates:** round 1's measured jobs: control 1 h 19 min and always-block 1 h 28 min for 109 tasks at 4 lanes; T-open 1 h 20 min for 581 states. G1 and G2 generate no thoughts, so they should land at the low end.
- **The binding constraints:** code, checks, queue and writing time, not GPUs.

### G1. Controlled repetition (the dose test, drafted 2026-10-13, run 2026-10-14)

[dose.py](dose.py), with [dose.slurm](dose.slurm) and [submit_dose.sh](submit_dose.sh). Ran on Explorer on 2026-10-14; the result is at the end of this section.

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

**Result (2026-10-14):** the first reading holds, at a much smaller size than the closed-loop curve. The table and what it means are in [loops_reason_10_13.md](loops_reason_10_13.md) §1, and in [ablation_G1.md](ablation_G1.md) with the rest of G1's record.
- **The run:** jobs 11065759–62 at `6c16369`, all exit 0. Wheelhouse 31 (vLLM 0.19.1, transformers 5.13.1), one A100-SXM4-80GB.
  - prep took 2 min; every template check passed.
  - The first GPU job took 1 h 05: score 58 min, sample 6 min. The second found nothing left to do.
  - Outputs in `$R/paper/loops/dose_baseline_nothink_args_v30_zip/`. The local analysis beyond `dose.md` (exact probabilities, spread, splits) ran from a session scratchpad, not from this repo.
- **The histories:** 60 in 56 tasks: 32 from Gate 1's run and 28 from `ctrl`. 39 are fastapi, 16 rich, 4 requests and 1 httpx. 52 are `err_silent` and 8 `ok_empty`. All 60 have `para2`, 56 have `para4` and 24 have `var2`.
- **The numbers:** p(c) at the run's sampling is 0.002, 0.098 and 0.434 for `dose1`, `dose2` and `dose4`; `para2` 0.027 and `para4` 0.038; `var2` 0.000. The paired dose4 − para4 is +0.376 (0.258 to 0.496).
- **The scored and sampled measures agree.** Sampled exact copies are 0%, 12% and 41%. Per history, at dose4, scored and sampled correlate at 0.95.
- **Two quirks in `dose.md`:**
  - One history's p(c) is 1.126 at the run's sampling. The two paths are scored in separate requests, and T = 0.2 amplifies the small differences between requests near a tie. Its exact probabilities sum to 0.55. No other sum is above 1, for any arm, target or measure, and capping this one at 1 moves dose4's mean by 0.002.
  - The "student copied next" row is empty, because none of the 60 recorded next calls copied c. Splitting by `next_cls` (53 `after_miss`, 7 `return_to_earlier`) would say more.
- **For the rest of round 2:**
  - **G2:** G1's para arms vary the call and keep the result. G2 varies the result and keeps the call, so together they say which half of the block has to match.
  - **G3:** G1 predicts lock-in survives T = 1, at a lower level.
  - **A cheap add-on, not built:** score the empty-thought tokens alone. That gives p(c) after a forced thought at each dose, the always-block lever on controlled histories.

### G2. What the result means, against its surface form (drafted 2026-10-14, run 2026-10-15)

[feedback.py](feedback.py), with [feedback.slurm](feedback.slurm) and [submit_feedback.sh](submit_feedback.sh). Tested locally (the last section); ran on Explorer on 2026-10-15, with the result at the end of this section.

| | Neutral note | Semantic note |
|---|---|---|
| One ID: the repeated results keep matching surface forms | `placebo` | `semantic` |
| An ID of its own each: varying arbitrary IDs | `break` (the surface break) | `both` |

- **States:** picked by history, whatever the student did next.
  - A repeat state ends just after an empty read-only search c that repeats an earlier identical call and got the identical result, with no write since c first ran: no successful `edit_file` or `write_file`, and no shell write.
  - c's earlier copies must be in the context; a compaction can hide them.
  - The controls are first failures, where c ran once, as G1's histories.
  - They are picked round-robin over task runs. Each round takes one state with c run twice and one with c run 3 or more times, up to 2 of each kind per task run, different calls first. One control per task run.
- **Runs:** Gate 1's run, its replicate `baseline_nothink_args_v30_zip_r2`, and round 1's `ctrl`.
  - The replicate has the same zip (`532374430fe2`), wheelhouse 30, harness signature and vLLM build as Gate 1's run.
  - Only 23 tasks have a repeat state at all, so the replicate adds task runs, not tasks. The intervals rest on those 23 clusters.
- **The relevant results R:** every result in the context identical to c's, from a command where an empty result means "no match" (`faithful_miss`).
  - That leaves out inverted searches (`-v`, `-L`), which find nothing when every line matched. For exit 0, it also leaves out a quiet grep (`-q` exits 0 on a match), `xargs`, and a `find` that acts on what it finds.
  - An identical result from another command, such as a `sed -i` that printed nothing, stays as recorded.
  - c itself must be a faithful miss.
- **The renderings:** each adds two keys at the end of every result in R's JSON, where the harness puts its budget warning: `"note"`, then `"id"`.
  - The neutral note: "The tool server returned this response to the agent in the current session."
  - The semantic note: "The search completed and found zero matches for this pattern in these paths." That's the plan's wording, with "paths" so it reads right for `find` too. Both notes are 14 tokens, and neither tells the model what to do.
  - IDs are 8 digits, one token each, drawn from the state's key. Under "one ID", every result gets the ID that the last result has in `break`. So `placebo` and `break` end with identical tokens and differ only in whether the earlier results match the last one.
  - Every call stays as recorded, and the four cells add the same tokens to the same results.
- **The other arms:**
  - `orig` and `dedup`: the anchors, scored again in the same environment;
  - `block`: orig, then the empty thought (Gate 1's block, the always-block template's prefill);
  - `think_all`: the empty thought wherever a past model output starts (after the model-turn header and after each tool response), and at the end.
  - If the block works by breaking the match between "r, then c" in the history and the end of the context, `think_all` restores that match and the copy comes back. If it works through the thinking mode ([Google's formatting notes](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4); HF PR #147), the copy doesn't come back.
- **Measures:** as G1's.
  - p(c), the next call being c once more, at the run's sampling (rebuilt from the top 20 logprobs) and exact. It adds both ways the next call can start, except after `block` and `think_all`, which already end with the thought. orig's path through the thought and `block` share one request.
  - 3 samples per arm, the state's seed in every arm: exact copy, the same search, stale, malformed, escape.
  - An escape changes approach when it's a new search or another tool, not a variant of c's search (the same pattern, or one inside the other: a heuristic).
  - Paired contrasts with 95% intervals from resampling tasks: the two main effects and their interaction; the simple effects; and the anchors placebo − orig, dedup − orig, both − dedup, block − orig, think_all − block and think_all − orig.
  - By group: 2 copies in the context against 3 or more, copies against returns, exit 1 against exit 0, and the controls. These use the exact probability, because the run's sampling pushes p(c) towards 0 or 1, which bends additive contrasts.
- **Checks in prep, on the model's own template** (a probe of c, c, another empty search, c). A failed check stops the chain.
  - In each cell, the token-level edit is the template's rendering of the edited JSON.
  - Each edited result keeps its JSON and ends with the note, then the ID.
  - The cells add the same tokens to every result, and the two notes are as long.
  - `placebo` and `break` end with the same last result.
  - The template's history has no empty thought before past calls, so `think_all` doesn't double one; `think_all` puts one before each call.
  - `dedup` drops c's earlier copies and keeps the other search.
  - What counts as a faithful miss, on 10 cases.
- **Run it:** `GO=1 bash loops/submit_feedback.sh` from `~/Gemma4_paper_track`, after `git pull`. Without `GO=1` it prints the jobs and what blocks them.
  - The chain: prep (CPU), then three chained A100 jobs (2-h limit each; each finishes what the one before left), then the report.
  - Expect about 2.5 h of one A100: ~1,600 scored requests of ~9k tokens (about 2 h at G1's measured 2,070 prompt tok/s), and 3 samples for each of ~1,000 contexts. That's below the plan's 4–6 h.
  - Outputs: `$R/paper/loops/fb_baseline_nothink_args_v30_zip/`, with `prep.md`, `score.jsonl`, `samples.jsonl` and `feedback.md`.
- **What prep picks on the real runs** (run locally on copies, 2026-10-14):
  - 101 repeat states in 23 tasks (33 task runs), and 32 controls. That's 133 states, inside the plan's ~120–160; 2 per kind per task run caps it below the target of 128.
  - c has 2 copies in the context in 45 states, 3–4 in 9, and 5 or more in 47. 56 are copies (c was also the call before) and 45 are returns. 74 are `err_silent` and 27 `ok_empty`.
  - 70 states have another search's identical result in R. The cells add 348 tokens on average.
  - Left out: 6 states whose earlier copies of c a compaction hid, and 1 whose c is `-v` or `-q`.
  - The student's own next call: `copy_3rd_plus` 52, `stale_other` 26, `copy_2nd` 12, `after_miss` 11.
- **Reading it:**
  - IDs help (a negative surface effect): the identical surface form contributes.
  - Meaning helps beyond the placebo: interpretable feedback contributes.
  - Both help: report the interaction.
  - Nothing reliable: narrow the claim rather than sweep wordings.
  - placebo − orig shows what the added tokens do by themselves; both − dedup, how close the best rendering gets to deleting the copies.

**Result (2026-10-15):** nothing reliable on the result side, so the claim narrows. The full record is [ablation_G2.md](ablation_G2.md).
- **The run:** jobs 11091741–45 at `ed1b419`; every template check passed. The first GPU job hit its 2-h limit at 1,512 of 1,589 scored requests. The second finished scoring and sampling in 30 min.
- **The factorial:** in exact p(c), surface is −0.005 (−0.009 to −0.002) and meaning +0.020 (+0.005 to +0.036). The interaction is about 0, and placebo − orig −0.004.
- **The anchors:** dedup − orig is −0.407 and block − orig −0.317, both exact. think_all brings p(c) back to orig's level (+0.002), so the block works by breaking the match.
- **Where it happens:** nearly all of it is at the 56 copy states, where c was also the call before. The 45 returns cycle over several stale calls, and p(c) there barely moves.

### G3. Temperature and replication (drafted 2026-10-15)

[submit_loops.sh](submit_loops.sh), with [make_t1_zip.py](make_t1_zip.py), [loops.slurm](loops.slurm), [arms.tsv](arms.tsv) and [loop_report.py](loop_report.py). Tested locally (the last section), not yet run.

| Arm | Zip | Chat template | Sampling | Tests |
|---|---|---|---|---|
| `ctrl` | `off_args`'s pinned zip (`532374430fe2`) | the model's | T = 0.2, top-p 0.95 (top-k 64 from the model) | the control, and round 1's replicate |
| `block` | the same | always-block (`5f875818d2a2`) | as `ctrl` | round 1's block effect, again |
| `ctrl_t1` | the pinned zip with temperature 1.0 | the model's | T = 1.0, top-p 0.95, top-k 64: Google's | does lock-in survive T = 1? |

- **Two passes of each arm,** on the same 109 tasks (70 dev-sound and 39 rich-sound), PINS=1, unpaced, 4 tasks at once on one A100, as in round 1.
- **The T = 1 zip** ([make_t1_zip.py](make_t1_zip.py)):
  - it changes the one temperature line of `configs/sampling.yaml`. Every other entry keeps its bytes, name, order and timestamp;
  - the analyzer sub-agent includes the same file, so it runs at T = 1 too;
  - it is built once, then only checked. Its sha256 lands in every run's `run_meta.json`, so pass 2 must use the same bytes;
  - the main track's `slurm/package_submission.slurm` then checks it with the host's compiler, as for any zip we'd submit.
- **Why new runs on v31:**
  - The serving env was rebuilt for v31 on 2026-10-13, so round 1's v30 runs can't be repeated exactly.
  - For the 31B, v31 measurably changes nothing ([the main track's v31.md](../../Gemma4_kaggle_repo/post_train/RFT/v31.md)). The agent phase is byte-identical, and none of grading at 300 s, compaction interval 15 or the dropped `enable_thinking` server default changed a run of ours.
  - Round 1's runs stay in the report as a separate replicate (`ctrl_v30`, `block_v30`).
- **The chain, one pass at a time:**
  1. three checks: the template, the T = 1 zip, and the host compiler on that zip. A failed check cancels the pass;
  2. the three arms side by side, each two chained A100 jobs (2-h limit; the second finishes what the first left);
  3. the report.
- **Why one pass at a time:** the gpu partition lets a user have 8 jobs queued and 4 running (QoS `gpu`, 2026-10-15), and one pass is 6 GPU jobs.
- **Run it,** from `~/Gemma4_paper_track` after `git pull` there and in `~/Gemma4_kaggle_repo`:
  - `GO=1 bash loops/submit_loops.sh` queues pass 1;
  - once its GPU jobs have left the queue, `GO=1 PASS=2 bash loops/submit_loops.sh` queues pass 2. Its report reads both passes;
  - without `GO=1`, the script prints the jobs and what blocks them. `WIDTH=1` (or `SERIAL=1`) runs one arm at a time; the script's header has the rest.
- **Compute:** round 1 took 1 h 19 (`ctrl`) and 1 h 28 (`block`) for the 109 tasks. That's about 1.5 A100-hours per arm and pass, 4.5 per pass and 9 in all, inside the plan's 9–15. T = 1's time per task is unknown; each chain's second job covers 2 h more.
- **What the report writes,** in `$R/paper/loops/report_p<pass>_<time>/`:
  - `baseline_dev.md`, `baseline_rich.md`: resolved per run and pooled over the passes, each arm against `ctrl` task by task with a bootstrap CI and a sign test, call hygiene, agent time and output tokens;
  - `loops.md`:
    - every run's provenance: wheelhouse, harness, vLLM build, zip, template, sampling, PINS, pace, budget, GPU;
    - each arm's passes pooled: stale repeats split into copies and returns (G2's split), malformed calls, lock-in at k = 2 and 3+, runs of 3+ and 5+ per task, their length, loop tasks;
    - runs of 3+ by type: search, other command, read, edit, submit (termination), malformed;
    - after a run of 3+ that ends before the task does: is the call re-sent later (re-entry), does the next call work, does a source edit follow;
    - each pair's difference with a 95% interval from resampling tasks: `block` − `ctrl`, `ctrl_t1` − `ctrl`, round 1's `block` − `ctrl`, and `ctrl` and `block` against their round-1 runs;
    - each arm's pass 2 − pass 1, the noise floor.
- **Round 1 in these terms** (the new metrics on round 1's runs, `block` against `ctrl`). A replication should show the same directions:
  - lock-in at 3+ 95.7% → 89.0%;
  - runs of 3+ per task 0.50 → 0.80, and 23 → 11 calls long;
  - copies 31% → 26% of calls, returns 9% → 9%;
  - re-entry 18% → 45%;
  - the next call works 74% → 52%;
  - resolved 37 → 37.
- **Predictions, written before the run:**
  - **`ctrl_t1` locks in less, but still locks in.**
    - In the replay, the recorded 3rd-or-later copy keeps 0.73 at T = 1 with top-p 0.95 (loops_reason_10_13.md §3), and G2's copy states 0.79 in exact probability.
    - So expect lock-in at 3+ around 0.70–0.85, against 95.7%. Lock-in at k = 2 should fall below 77.5%.
    - Expect fewer and shorter runs of 5+, and a lower copy share.
  - **Malformed calls may rise at T = 1,** since argument tokens get sampled more widely.
  - **No prediction for resolution:** two passes can't detect a change of 1–2 tasks.
- **Reading it:**
  - Lock-in at 3+ in `ctrl_t1` far above its k = 1 rate means lock-in survives T = 1. Then the self-copying isn't a low-temperature artifact, as G1's and G2's exact probabilities predict.
  - Lock-in near the k = 1 rate means the closed loop's lock-in comes mostly from the sampler.
  - An arm difference no larger than the pass-to-pass row is no finding. Trace metrics come first; resolved tasks second, since one run varies by ±2.2.
  - The sample targets large behavioural effects. It can't establish a change of 1–2 resolved tasks, and an inconclusive comparison is not equivalence.

### G4. Conditional: feedback over whole tasks

- **When:** only if G2's semantic rendering has a clear, reproducible effect beyond placebo. It has none (G2, 2026-10-15), so G4 doesn't run as written. [ablation_G2.md](ablation_G2.md) §8 proposes a closed-loop dedup arm for its slot; not decided.
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
| [loop_report.py](loop_report.py) | Loop metrics of any runs: by task group, with the template's effect on the prompts. With `--arm` and `--pair` (G3): passes pooled per arm, copies and returns, loop types, what follows a run, paired intervals over tasks, pass to pass, provenance. stdlib only |
| [make_t1_zip.py](make_t1_zip.py) | G3's T = 1 zip: the pinned zip with only its temperature line changed, and the checks that say so. stdlib only |
| [arms.tsv](arms.tsv) | The three closed-loop arms (`ctrl`, `block`, `ctrl_t1`), in [../baseline/arms.tsv](../baseline/arms.tsv)'s format |
| [loops.slurm](loops.slurm) | The CPU steps: `STEP=template`, `STEP=t1zip` or `STEP=report` |
| [submit_loops.sh](submit_loops.sh) | Queues a pass of the closed-loop arms: checks → arms → report; dry unless `GO=1` |
| [dose.py](dose.py) | Round 2's G1, the dose test: `prep`, `score`, `sample`, `report`. Imports [../gate1/gate1.py](../gate1/gate1.py) |
| [dose.slurm](dose.slurm) | The dose test's steps in one job: CPU for prep and report, one A100 for score and sample |
| [submit_dose.sh](submit_dose.sh) | Queues prep → 2 GPU jobs → report; dry unless `GO=1` |
| [feedback.py](feedback.py) | Round 2's G2, the feedback factorial: `prep`, `score`, `sample`, `report`. Imports [dose.py](dose.py) and [../gate1/gate1.py](../gate1/gate1.py) |
| [feedback.slurm](feedback.slurm) | The factorial's steps in one job: CPU for prep and report, one A100 for score and sample |
| [submit_feedback.sh](submit_feedback.sh) | Queues prep → 3 GPU jobs → report; dry unless `GO=1` |
| [ablation_G1.md](ablation_G1.md) | G1's record: design, run and result, copied from §3 here and from [loops_reason_10_13.md](loops_reason_10_13.md) |
| [ablation_G2.md](ablation_G2.md) | G2's record: the run, the result, and the split into copies and returns |
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
- All four were then covered by the Explorer run of 2026-10-14 (§3, G1's result), with vLLM 0.19.1 and transformers 5.13.1.

## Tested: the feedback factorial (locally, 2026-10-14)

No GPU, and nothing queued on Explorer. The test data sits in a session scratchpad, outside this repo:
- the model's tokenizer and chat template (with the `tokenizer_config.json` that transformers 4.57 can read);
- 2 shards of Gate 1's run (14 tasks), and local copies of all three runs' records for one full prep.

The checks:
- **prep on the three runs:** all 17 template checks pass. It picks the 133 states described in §3, in 32 s.
- **64 checks against a fake vLLM.** The fake copies the last call as the dose test's fake does. It copies less when an earlier copy got a different result (the varying IDs, × 0.5), when the last result holds the semantic note (× 0.4), and when the context ends with a thought the earlier copy didn't have (× 0.5).
  - **Units:**
    - `faithful_miss` on 16 cases, `pattern_of` on 7, and `related`;
    - IDs: 8 digits and 8 tokens in place, all different, the same from the same key;
    - a response that isn't a JSON object is left alone.
  - **prep on the 2 shards:** 6 repeat states (both kinds, both results) and 7 controls, at most 2 of a kind per task run. The rerun rule works as in G1.
  - **Every arm's tokens, state by state:**
    - `orig` is the prompt, and `block` adds the thought.
    - `think_all` is the prompt with a thought before every call and at the end.
    - `dedup` drops c's k − 1 earlier copies.
    - The cells keep every call and edit exactly R: the original JSON, then the note and the cell's ID. Other results stay as recorded, and all four cells add the same tokens.
    - `placebo` and `break` end alike, and the semantic note appears only in `semantic` and `both`.
  - **score:**
    - one request per state, arm and path, with orig's thought path shared with `block` (rows = requests + states);
    - each request is the arm's context, the path, then c;
    - prefix caching off, prompt logprobs 20;
    - a rerun does nothing.
  - **sample:**
    - one request per state and arm, n = 3 at T = 0.2, top-p 0.95, top-k 64;
    - the state's seed in every arm;
    - a rerun does nothing.
  - **The judge, on 8 hand-made continuations:** a copy, a copy after the thought, a re-spelling, a broadened pattern, a new search, another tool, a malformed call and no call.
  - **The report recovers the fake's effects:**
    - negative surface and meaning effects;
    - `placebo` equal to `orig`;
    - `dedup` and `block` below `orig`, and `think_all` above `block`;
    - the same report twice. An unknown arm is refused.
- **`submit_feedback.sh`:** 22 cases against stub `sbatch` and `squeue` and a fake scratch tree.
  - The 5-job chain and its dependencies, with the three default runs and `NAME`.
  - `JOBS`, `ARMS`, `PREP_ARGS`, `AFTER`, and runs given as arguments.
  - Every refusal. A dose job of ours doesn't block it.
- `bash -n` passes on `feedback.slurm` and `submit_feedback.sh`.
- **Not run:** a real vLLM; the GPU steps' speed on these states (estimated from G1's); anything with Explorer's paths.

## Tested: G3's closed-loop runs (locally, 2026-10-15)

No GPU, and nothing queued on Explorer. The test data sits in a session scratchpad, outside this repo: round 1's `ctrl` and `block` runs (traces, `task_results.jsonl` and `run_meta.json`; `calls.jsonl` of one shard), Gate 1's run and its replicate.
- **[make_t1_zip.py](make_t1_zip.py),** 23 checks on a zip rebuilt from the main track's `agents/baseline_nothink_args` at `ebc522f`, whose six entries have the pinned zip's sizes:
  - the T = 1 zip keeps every other entry's bytes, every name, the order and the timestamps; `sampling.yaml` gains `temperature: 1.0` and changes nothing else;
  - a second build gives the same bytes. A rerun only checks, and leaves the file alone;
  - an existing zip that differs in another file, in `top_p` too, or at another temperature is refused and left as it was;
  - a source with no temperature line, two of them, or already at T = 1 is refused, and nothing is written. A comment on the line is kept.
- **[loop_report.py](loop_report.py):**
  - with `--runs` only, its output on round 1's two runs is byte for byte what it was before this change. Apart from the template table, which needs every `calls.jsonl`, it is round 1's `loops.md`;
  - 21 unit checks on hand-made call sequences:
    - copies, returns, lock-in counts, runs and their types;
    - re-entry, which a successful edit in between clears; the next call working; a source edit after, where a test edit doesn't count;
    - the paired interval: identical arms give 0; a constant shift gives the shift; only common tasks count, and an arm's passes pool within a task;
  - with arms on round 1's runs, it reproduces round 1's pooled lock-in: `ctrl` 77.5% and 95.7%, `block` 89.0% at 3+;
  - an arm whose runs don't exist yet, and a missing pass, are reported, not fatal. The full report takes 4 s.
- **[../baseline/baseline_report.py](../baseline/baseline_report.py)** reads the new [arms.tsv](arms.tsv). `ctrl_t1` shows "no runs yet", and `block` against `ctrl` on round 1's runs is as before: −1.0 (−6.0 to +3.0) on dev.
- **[submit_loops.sh](submit_loops.sh): 52 cases** against stub `sbatch`, `squeue`, `scontrol` and `sacctmgr` and a fake scratch tree:
  - the 10-job pass. Every arm's first job waits for all three checks (`afterok`), its second for its first, and the report for all six GPU jobs and the checks;
  - the zip check runs from the main checkout, after `t1zip`, and keeps its `singleton`;
  - the run names on v31, `PASS=2`'s `_r2` runs, and the report's arms and pairs, round 1's included;
  - `WIDTH` and `SERIAL=1`, `ZIPCHECK=0`, `AFTER`, `OLD`, `STEPS=report` and `ctrl_t1` alone;
  - every refusal: the QoS limit (3 queued + 6 > 8; 2 + 6 passes), Kaggle's wheelhouse, loops jobs queued (`FORCE=1` overrides), a missing template, zip, package job or wheelhouse version, an unknown step, and an `arms.tsv` with other zips.
- `bash -n` passes on [submit_loops.sh](submit_loops.sh) and [loops.slurm](loops.slurm).
- **Checked on Explorer** (light commands, 2026-10-15):
  - the pinned zip's sha256 (`532374430fe2…`) and its six entries;
  - the gpu QoS (8 queued, 4 running);
  - the validator venv, which already has `adk_submission` 0.2.13, the newest wheel, so the zip check needs no pip.
- **Not run:**
  - the eval jobs;
  - `package_submission.slurm` on the T = 1 zip;
  - `make_t1_zip.py` on the real pinned zip; its rebuilt twin has the same entries and sizes;
  - anything with Explorer's paths.
