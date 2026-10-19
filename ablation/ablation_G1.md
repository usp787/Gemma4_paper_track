# ablation/ablation_G1.md: the dose test (round 2's G1)

<!-- moved from loops/ on 2026-10-17. File names without a folder below (README.md, loops_reason_10_13.md, dose.py,
     dose.md, ...) are in ../loops/ or in the work dir; the links point there. The recap of every ablation is
     ablation_recap_10_17.html. -->

<!-- copied on 2026-10-15 from README.md §3 (G1) and from loops_reason_10_13.md (§1's dose-test block, and G1's notes in
     §2-§4, §7 and §8), as written on 2026-10-13 and 2026-10-14. Both files keep their text; this copy puts G1's
     design, run and result in one place. Only the section headings and the cross-references, which now name their file,
     are new. Numbers come from $R/paper/loops/dose_baseline_nothink_args_v30_zip (dose.md, score.jsonl, samples.jsonl,
     states.jsonl). The exact probabilities, the spread and the splits come from a local analysis of those files in a
     session scratchpad, not in this repo. G2's record is ablation_G2.md. -->

**The question:** do added copies raise p(copy) at a fixed history? (README.md §3, round 2's goals)

**The answer** (loops_reason_10_13.md, summary table): lock-in is observed here. Isolated at fixed histories: identical copies raise the next copy, re-spelled ones don't. The effect is far below the observed curve, so the curve also reflects which histories loop.

## 1. Design (from README.md §3)

[dose.py](../loops/dose.py), with [dose.slurm](../loops/dose.slurm) and [submit_dose.sh](../loops/submit_dose.sh). Ran on Explorer on 2026-10-14.

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

## 2. The run (from README.md §3)

**Result (2026-10-14):** the first reading holds, at a much smaller size than the closed-loop curve.
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

## 3. The result (from loops_reason_10_13.md §1)

The table and the paired differences (the design bullets of loops_reason_10_13.md §1 repeat §1 above):

| Arm | Added tokens | p(c), run's sampling | p(c), exact | Sampled: exact copy | Sampled: escape |
|---|---|---|---|---|---|
| `dose1`: $c$ once (the history) | 0 | 0.002 | 0.006 | 0% | 92% |
| `dose2`: $c$ twice in a row | 80 | 0.098 | 0.052 | 12% | 79% |
| `dose4`: four in a row | 239 | 0.434 | 0.307 | 41% | 53% |
| `para2`: $c$, then one re-spelling | 80 | 0.027 | 0.009 | 3% | 84% |
| `para4`: $c$, then three re-spellings | 246 | 0.038 | 0.015 | 4% | 84% |
| `var2`: $c$, then a different search (24 histories) | 80 | 0.000 | 0.003 | 0% | 97% |

Paired differences in p(c) at the run's sampling, with 95% intervals from 10,000 resamples of tasks:
- dose2 − dose1: +0.096 (0.040 to 0.165);
- dose4 − dose2: +0.336 (0.231 to 0.443);
- dose4 − para4: +0.376 (0.258 to 0.496);
- para4 − dose1: +0.036 (−0.002 to 0.092).

- **Identical copies raise the next copy at a fixed history.**
  - In the exact probabilities, p(c) rises from dose1 to dose2 to dose4 in 59 of 60 histories.
  - The median gain in log-odds is +5.3 from one copy to two, and +5.0 from two to four (+2.5 per added copy there).
  - Both source runs give the same curve: dose4 0.442 on Gate 1's histories, 0.425 on the control arm's.
- **It needs the identical call.**
  - Re-spelled copies of the same search, with the same empty result, barely move p(c).
  - They don't raise the re-spelling either: the last added call's p is 0.002 after `para2` and 0.000 after `para4`. Samples repeat the search in any spelling 3–4% of the time (a heuristic normaliser).
  - Length isn't the cause: `para4` adds as many tokens as `dose4`.
  - So seeing the same failure four times doesn't by itself make the model retry the search. The whole call has to recur.
- **It is far smaller than the observed curve.**
  - Two forced copies give a third about 10% of the time (scored 0.098, sampled 12%). In the closed loop, after two identical calls in a row, a third follows 77.5% of the time.
  - Four forced copies give about 43%, against 95.7% at k = 3+.
  - So on these histories, the copies alone don't reach the curve's height. Histories differ in how sticky they are, and the ones that reach a run in a rollout are likely the sticky ones.
  - Small groups point the same way. At dose4, p(c) is 0.64 for contexts longer than the median, against 0.23. It is 0.49 in tasks the run didn't resolve (48 histories), against 0.23 (12). It is 0.60 where the student next went back to an earlier call (7), against 0.41 (53).
- **What it doesn't show:**
  - *How to split the curve between reinforcement and selection.* The populations differ. Here: first occurrences of empty searches, picked by history. In the curve: every run of identical calls in the control arm, of any tool.
  - *That the closed loop behaves the same.* It is one step ahead, on v30 histories scored in v31's serving env.
  - *Generality across repositories.* 39 of the 60 histories are fastapi, 16 rich.

## 4. G1 in the other sections of loops_reason_10_13.md

- **§2, observation 3** (the empty thought puts four tokens after the result):
  - G1 points the same way, without testing it. At four copies, nearly all of p(c) goes straight to the call (0.400 of 0.434).
  - In the samples, 8 of the 70 that open the empty thought themselves copy $c$ (11%). Of the 110 that don't, 66 do (60%).
  - The model chooses when to write the thought, so this is selection, not the block's effect.
- **§2, observation 5: a repeated result followed by a different call doesn't cue that call.**
  - In `para2` the context ends "$c$, $r$, $c'$, $r$", where $c'$ re-spells $c$. The last time $r$ came back, $c'$ followed it, so a cue read from the result alone predicts $c'$ next.
  - Its probability is 0.002. After "$c$, $r$, $c$, $r$" in `dose2`, p(c) is 0.098.
  - So the build-up needs the call to recur, not only the result.
  - *But* a re-spelling may be a less likely string for the model in any context. Counting any spelling of the search, the samples repeat it 3% of the time after `para2`, against 12% after `dose2`.
  - Whether the identical result is needed as well is G2's question.
- **§3, low temperature, at fixed histories:**
  - The dose effect is in the model's own probabilities: exact p(c) is 0.006, 0.052 and 0.307 for one, two and four copies. So it isn't the sampler's doing, and G3's T = 1 arm should still lock in, less often.
  - T = 0.2 polarises the histories. At four copies, at the run's sampling, 27 of 60 have p(c) above 0.5 and 27 below 0.01. In the exact probabilities it's 14 and 1.
  - Scored and sampled copy rates agree there (0.098 against 12%, 0.434 against 41%), so the top-20 rebuild holds at T = 0.2 on these states. That doesn't settle the T = 1 rebuild.
- **§4, the onset:** G1 agrees at fixed histories. At the first occurrence of an empty search, p(c) is 0.002, and none of the 60 recorded next calls copies it. Loops rarely start with a copy of the first failure.
- **§7, the stylised model:**
  - From two copies to four, the exact log-odds of $c$ rise by a median 2.5 per copy, at T = 1. Read through this model, which isn't fitted, that is $\lambda$'s size there.
  - In `para2`, the re-spelling $c'$ follows the repeated result, as $m$ requires, but has only one copy. Its probability stays at 0.002, though $c'$ may have a lower prior than $c$.
  - So in this picture $m$ does little without $n$.
- **§8, what is established:**
  - measured: at fixed histories, identical copies raise the next copy and re-spelled ones don't. The effect is far below the observed curve;
  - untested: how much of the observed curve is selection of sticky histories (G1 shows only that the copies' own effect is far smaller);
  - untested: the repeated result as the copy cue, against format conditioning (G2). G1 shows the call has to recur too.

## 5. For the rest of round 2 (from README.md §3, as written on 2026-10-14)

- **G2:** G1's para arms vary the call and keep the result. G2 varies the result and keeps the call, so together they say which half of the block has to match.
- **G3:** G1 predicts lock-in survives T = 1, at a lower level.
- **A cheap add-on, not built:** score the empty-thought tokens alone. That gives p(c) after a forced thought at each dose, the always-block lever on controlled histories.

Added on 2026-10-15: G2 ran on 2026-10-15; its record is [ablation_G2.md](ablation_G2.md). The references are in [loops_reason_10_13.md](../loops/loops_reason_10_13.md) (References) and [literature.md](../loops/literature.md) §4.
