# ablation/ablation_G3.md: temperature and replication over whole tasks (round 2's G3)

<!-- written 2026-10-19 from $R/paper/loops/report_p1_1015-1425 and report_p2_1016-1640 (loops.md, baseline_dev.md,
     baseline_rich.md), with $R = /scratch/zha.j/gemma4. The tables of §3 and §4 are report_p2's, which pools both passes
     and round 1's v30 runs. The splits of §5 (hazard by run length, clean steps, how runs end, rescue, which tasks
     loop, costs) come from local analyses of copied traces in session scratchpads, not in this repo; they resample
     tasks 2,000 times (seed 0), as loop_report.py does, except §5.7 (10,000). §5.6's exit table and §5.7 were added on
     2026-10-19. The design and the predictions, written before the run, are in ../loops/README.md §3 (G3). File names
     without a folder below (README.md, loop_report.py, submit_loops.sh, ...) are in ../loops/. The recap of every
     ablation is ablation_recap_10_17.html; G1's and G2's records are ablation_G1.md and ablation_G2.md. -->

**The question:** G1 and G2 look one step ahead. G3 runs whole tasks to ask two things. Does lock-in survive Google's T = 1, or is it an artifact of sampling at 0.2? And do round 1's always-block results replicate on the v31 serving stack?

**The answer:**
- **Lock-in survives T = 1.** Once a run of identical calls is under way, T = 1 barely slows it. The well-formed hazard at 3+ is 0.87 against 0.95 at T = 0.2 (−0.08, −0.13 to −0.05).
- **T = 1 cuts the onset.**
  - Fewer runs start: the stale share falls from 38.6% to 20.4% of calls (−18.2 points, −24.8 to −11.0), and loop tasks from 24.8% to 9.2%.
  - A copy right after a stale call is half as likely (0.100 → 0.052).
- **Resolution doesn't move:** 71 of 218 task runs at T = 1, against 72 at T = 0.2 (−0.5 points, −5.0 to +4.1). T = 1 also costs 6.5 more timeouts per pass and 9.5 more tasks with a malformed call.
- **At T = 0.2, nothing inside a long run ends it.**
  - On clean steps (no budget warning in the last result, no compaction before the next call), the control sent one more copy after 5 or more identical calls every time: 290 of 290 steps at 5–9, and 1,052 of 1,052 at 10 or more.
  - None of the 58 well-formed runs of 10+ ended on a clean step.
- **Un-looping solves nothing on its own.** T = 1 un-looped 46 of the control's 54 loop task runs. The control solved 10 of those tasks, and T = 1 solved 9.
- **The block replicates its shape, not its stale drop.** Lock-in at 3+ falls, more runs start, runs are shorter and re-entry rises, as in round 1. But the stale share (+1.6 points, −4.5 to +7.6) and loop tasks (+1.8) don't fall. Withdraw loops/README.md's "the stale share falls 5–6 points" (§9).
- **The control replicates round 1:** every `ctrl` − `ctrl_v30` interval includes 0.

## 1. Design (from README.md §3)

[submit_loops.sh](../loops/submit_loops.sh), with [make_t1_zip.py](../loops/make_t1_zip.py), [loops.slurm](../loops/loops.slurm), [arms.tsv](../loops/arms.tsv) and [loop_report.py](../loops/loop_report.py). README.md §3 (G3) has the full design, the chain and the predictions; "Tested: G3's closed-loop runs" has the local tests.

| Arm | Zip | Chat template | Sampling | Tests |
|---|---|---|---|---|
| `ctrl` | `off_args`'s pinned zip (`532374430fe2`) | the model's | T = 0.2, top-p 0.95 (top-k 64 from the model) | the control, and round 1's replicate |
| `block` | the same | always-block (`5f875818d2a2`) | as `ctrl` | round 1's block effect, again |
| `ctrl_t1` | the pinned zip with temperature 1.0 (`c07fc22da743`) | the model's | T = 1.0, top-p 0.95, top-k 64: Google's | does lock-in survive T = 1? |

- **Tasks:** the same 109 in every arm (70 dev-sound and 39 rich-sound). PINS=1, unpaced, 4 tasks at once on one A100, as in round 1.
- **The T = 1 zip** differs from the pinned zip only in the temperature line of `configs/sampling.yaml`. The analyzer sub-agent includes the same file, so it runs at T = 1 too.
- **Round 1's runs** (`ctrl_v30`, `block_v30`, wheelhouse v30) stay in the report as a separate replicate. v30 can't be rerun exactly: the serving env was rebuilt for v31 on 2026-10-13.
- **Planned:** two passes of each arm. **Ran:** two passes of `ctrl` and `ctrl_t1`, and one of `block`. Pass 2 ran the controls only, the user's choice on 2026-10-16, as suggested after pass 1.

## 2. The runs

- **Pass 1** was queued on 2026-10-15 at 14:24–14:25 EDT, at `03b7890`.
  - Checks, all PASS:
    - the template (11108025; sha256 `5f875818d2a2`, byte-identical to round 1's);
    - the T = 1 zip (11108035; sha256 `c07fc22da743`);
    - the host compiler on that zip (11108039; `adk_submission` 0.2.13, wheelhouse v31, as on Kaggle).
  - Arms (first job / chained second job):
    - `ctrl` 11108041 / 11108042, on d1029, 15:05–16:15;
    - `block` 11108044 / 11108047, on d1029, 22:38–23:49;
    - `ctrl_t1` 11108051 / 11108053, on d1028, 23:16–00:50.
  - Report 11108055 → `report_p1_1015-1425/`.
  - `ctrl_t1`'s first job was killed 2 min in (batch CANCELLED, signal 9), before vLLM started, at the same second `ctrl`'s no-op second job ended on d1028. It wrote nothing, and its chained job ran all 109 tasks. The cause is unknown and on the node's side.
- **Pass 2** (`PASS=2 STEPS="ctrl ctrl_t1 report"`) ran on 2026-10-16.
  - `ctrl` 11144870 (d1029, 16:41–17:52) and `ctrl_t1` 11144873 (d1029, 17:51–19:03). Each chain's second job found nothing left.
  - Report 11144875 → `report_p2_1016-1640/`, which reads both passes and round 1's runs.
- **Every run is complete:** 109 of 109 tasks, none skipped.
- **Provenance:** every run has the same harness (`067143a2f3ec`) and vLLM build (`73c69bafca24`). Template, zip and sampling differ only where intended.
  - The main-track checkout moved between jobs: `a09f4fd` → `ba85fc5` → `28780bd`.
  - Under `baseline/`, the only change is `baseline/smoke_check.py`. The rest went to `agents/budgetfit_lora_probe/`, which no arm uses. The eval path is unchanged, and the harness signature agrees.
- **Compute:** about 6.3 A100-hours for 5 arm-passes (1 h 10 to 1 h 34 each). The estimate was 9 for 6.
- **Task endings per run** (resolved, timeout, context window):

| Run | Resolved | Timeouts | Context window |
|---|---|---|---|
| `ctrl` pass 1 / 2 | 35 / 37 | 11 / 12 | 4 / 4 |
| `ctrl_t1` pass 1 / 2 | 35 / 36 | 18 / 18 | 3 / 5 |
| `block` pass 1 | 38 | 14 | 3 |
| `ctrl_v30`, `block_v30` (round 1) | 37, 37 | 12, 16 | 1, 1 |

## 3. Arms, passes pooled (from report_p2's loops.md)

All 109 tasks. Copies repeat the call just before; returns repeat an earlier one. Loop tasks are task runs where at least half the calls are bad (stale or malformed). "After a run of 3+" counts runs that end before the task does.

| Arm | Runs | Task runs | Resolved | Stale: copies / returns | Malformed | Lock-in k=2 / 3+ | Runs 3+ / 5+ per task | Run length (3+) | Loop tasks | After a run of 3+: re-entered / next works / source edit after (runs) |
|---|---|---|---|---|---|---|---|---|---|---|
| ctrl | 2 | 218 | 33.0% | 38.6%: 28.0% / 10.6% | 5.5% | 69.0% / 94.8% | 0.53 / 0.42 | 19.81 | 24.8% | 21.5% / 60.7% / 37.4% (107) |
| block | 1 | 109 | 34.9% | 40.2%: 27.9% / 12.3% | 5.4% | 69.6% / 89.2% | 0.88 / 0.56 | 10.86 | 26.6% | 47.3% / 52.7% / 41.8% (91) |
| ctrl_t1 | 2 | 218 | 32.6% | 20.4%: 12.0% / 8.4% | 7.7% | 58.3% / 89.9% | 0.39 / 0.22 | 11.18 | 9.2% | 35.1% / 57.1% / 40.3% (77) |
| ctrl_v30 | 1 | 109 | 33.9% | 40.4%: 31.3% / 9.2% | 3.4% | 77.5% / 95.7% | 0.50 / 0.39 | 23.47 | 31.2% | 18.0% / 74.0% / 40.0% (50) |
| block_v30 | 1 | 109 | 33.9% | 34.7%: 25.8% / 8.9% | 3.1% | 73.1% / 89.0% | 0.80 / 0.61 | 10.71 | 21.1% | 44.6% / 51.8% / 45.8% (83) |

By task group:

| Arm | Dev (70): resolved per run | Dev: stale / loop tasks | Rich (39): resolved per run | Rich: stale / loop tasks |
|---|---|---|---|---|
| ctrl | 24, 25 | 32.7% / 17.9% | 11, 12 | 48.2% / 37.2% |
| block | 25 | 33.4% / 15.7% | 13 | 50.9% / 46.2% |
| ctrl_t1 | 27, 25 | 18.2% / 7.9% | 8, 11 | 24.6% / 11.5% |
| ctrl_v30 | 25 | 32.3% / 21.4% | 12 | 53.6% / 48.7% |
| block_v30 | 24 | 26.3% / 8.6% | 13 | 48.3% / 43.6% |

- `baseline_dev.md` against `ctrl`: `block` +0.5 tasks (−4.5 to +5.5), `ctrl_t1` +1.5 (−2.5 to +5.5). `baseline_rich.md`: `block` +1.5 (−2.5 to +5.5), `ctrl_t1` −2.0 (−5.5 to +1.0).
- **Runs of 3+ by type** (search = a read-only grep or find):

| Arm | Search | Command | Read | Edit | Malformed |
|---|---|---|---|---|---|
| ctrl (2 passes) | 39 | 50 | 1 | 0 | 26 |
| block | 42 | 38 | 0 | 1 | 15 |
| ctrl_t1 (2 passes) | 7 | 37 | 2 | 3 | 35 |
| ctrl_v30 | 21 | 27 | 1 | 1 | 5 |
| block_v30 | 46 | 35 | 0 | 2 | 4 |

- **How runs of 3+ end** (summed over dev and rich; loop_report.py checks task end, then the budget warning, then a compaction, else "own"):

| Run | Task end | Budget warning | Compaction | Own |
|---|---|---|---|---|
| ctrl pass 1 / 2 | 4 / 5 | 28 / 30 | 18 / 15 | 9 / 7 |
| block pass 1 | 5 | 18 | 11 | 62 |
| ctrl_t1 pass 1 / 2 | 4 / 3 | 8 / 12 | 7 / 14 | 13 / 23 |
| ctrl_v30 | 5 | 29 | 14 | 7 |
| block_v30 | 4 | 17 | 13 | 53 |

"Own" includes runs of 3–4 and malformed runs. §5.2 has the well-formed long runs.

## 4. Paired differences (from report_p2's loops.md)

A − B on the 109 tasks, with 95% intervals from 2,000 resamples of tasks. Shares are in percentage points. The round-1 column is the same report's v30 pair, and the last column is the v31 control against round 1's.

| Metric | ctrl_t1 − ctrl | block − ctrl | Round 1: block_v30 − ctrl_v30 | ctrl − ctrl_v30 |
|---|---|---|---|---|
| resolved | −0.5 (−5.0 to +4.1) | +1.8 (−3.7 to +7.8) | +0.0 (−5.5 to +5.5) | −0.9 (−6.0 to +4.1) |
| stale | **−18.2 (−24.8 to −11.0)** | +1.6 (−4.5 to +7.6) | −5.7 (−12.3 to +0.3) | −1.8 (−7.5 to +3.9) |
| copies | **−16.0 (−23.2 to −8.6)** | −0.1 (−6.9 to +6.5) | −5.4 (−12.4 to +0.9) | −3.2 (−10.2 to +3.9) |
| returns | −2.2 (−4.8 to +0.5) | +1.7 (−1.5 to +5.2) | −0.3 (−3.6 to +2.7) | +1.4 (−2.2 to +4.7) |
| malformed | +2.2 (−3.2 to +7.5) | −0.1 (−4.0 to +3.9) | −0.4 (−4.3 to +3.1) | +2.0 (−0.2 to +4.4) |
| lock-in k=2 | −10.7 (−25.0 to +2.6) | +0.5 (−13.4 to +14.4) | −4.4 (−16.3 to +6.6) | −8.4 (−24.3 to +6.7) |
| lock-in k=3+ | **−4.9 (−8.7 to −1.9)** | **−5.6 (−9.1 to −2.8)** | **−6.8 (−10.0 to −3.9)** | −1.0 (−2.3 to +0.4) |
| runs 3+ per task | −0.15 (−0.31 to +0.02) | **+0.35 (+0.13 to +0.60)** | **+0.29 (+0.09 to +0.51)** | +0.03 (−0.11 to +0.16) |
| runs 5+ per task | **−0.21 (−0.34 to −0.07)** | +0.14 (−0.00 to +0.28) | **+0.22 (+0.06 to +0.39)** | +0.03 (−0.07 to +0.13) |
| run length (3+) | **−8.63 (−12.76 to −4.53)** | **−8.95 (−12.63 to −4.80)** | **−12.76 (−17.48 to −8.33)** | −3.66 (−8.73 to +1.20) |
| loop tasks | **−15.6 (−23.4 to −8.3)** | +1.8 (−6.4 to +10.6) | **−10.1 (−18.3 to −0.9)** | −6.4 (−13.8 to +0.9) |
| re-entered | **+13.6 (+1.0 to +26.1)** | **+25.8 (+8.8 to +40.9)** | **+26.6 (+10.9 to +41.3)** | +3.5 (−7.8 to +15.5) |
| next call works | −3.6 (−17.7 to +10.1) | −8.0 (−21.8 to +7.5) | **−22.2 (−38.1 to −5.1)** | −13.3 (−27.8 to +2.0) |
| source edit after | +2.9 (−16.5 to +22.3) | +4.4 (−13.4 to +24.5) | +5.8 (−14.5 to +26.6) | −2.6 (−19.9 to +12.0) |
| tasks with a source edit | +3.7 (−3.2 to +11.0) | +5.0 (−2.8 to +12.8) | +4.6 (−2.8 to +12.8) | −4.1 (−11.5 to +3.7) |

Bold: the interval excludes 0.

**Pass to pass** (run 2 − run 1), the noise floor:

| Arm | Resolved | Stale | Copies | Returns | Lock-in k=3+ | Runs 3+ per task |
|---|---|---|---|---|---|---|
| ctrl | +1.8 (−2.8 to +7.3) | −0.5 (−7.1 to +6.5) | −1.7 (−9.1 to +6.4) | +1.2 (−2.5 to +5.0) | +0.2 (−1.3 to +1.7) | −0.02 (−0.19 to +0.16) |
| ctrl_t1 | +0.9 (−4.6 to +7.3) | +4.3 (−4.8 to +12.6) | +0.9 (−8.0 to +9.4) | **+3.4 (+0.8 to +6.1)** | −4.5 (−11.9 to +1.9) | +0.18 (−0.04 to +0.40) |

One of the 12 pass-to-pass intervals excludes 0. That is about what 95% intervals give by chance.

## 5. Local splits

### 5.1 The hazard by run length (well-formed calls)

P(the next call is identical | the last k calls were identical). Both passes are pooled, with paired intervals over tasks. "After a stale call" is the copy start where the call just copied was itself stale.

| k | ctrl_t1 | ctrl | Difference |
|---|---|---|---|
| 1 (copy start) | 0.016 (6,359) | 0.025 (5,451) | −0.009 (−0.016 to −0.003) |
| 1, after a stale call | 0.052 (613) | 0.100 (824) | −0.047 (−0.085 to −0.015) |
| 2 | 0.505 (97) | 0.672 (134) | −0.166 (−0.345 to −0.020) |
| 3–4 | 0.671 (82) | 0.913 (172) | −0.242 (−0.357 to −0.147) |
| 3+ | 0.869 (373) | 0.950 (1,747) | −0.082 (−0.130 to −0.054) |

- **Each pass alone** points the same way. Pass 1 at k = 3–4 is 0.769 against 0.931, and pass 2 is 0.625 against 0.894.
- **Malformed calls lock in alike at both temperatures:** 0.928 (391) against 0.935 (310) at 3+. In pass 1 they were mostly identical long `edit_file` argument errors; 17 of ctrl_t1's 32 runs of 3+ were malformed.
- **On clean steps only** (no budget warning in the last result, no compaction before the next call):

| Arm | k = 1 | k = 2 | k = 3–4 | k = 5–9 | k = 10+ |
|---|---|---|---|---|---|
| ctrl | 0.024 (4,985) | 0.678 (115) | 0.958 (142) | **1.000 (290)** | **1.000 (1,052)** |
| ctrl_t1 | 0.015 (5,861) | 0.488 (82) | 0.767 (60) | 0.938 (80) | 0.994 (174) |
| block | 0.048 (2,253) | 0.712 (104) | 0.812 (128) | 0.929 (182) | 0.948 (345) |
| ctrl_v30 | 0.024 (2,414) | 0.776 (58) | 0.938 (81) | 1.000 (168) | 0.999 (694) |
| block_v30 | 0.048 (2,215) | 0.752 (105) | 0.921 (139) | 0.899 (218) | 0.931 (276) |

- **The steps that aren't clean** are where runs end. In `ctrl` the hazard there is 0.69–0.70 from k = 3 on, and in `ctrl_t1` 0.41–0.73.
- **By type,** clean steps, at k = 2: ctrl's searches continue at 0.86 (36) and its commands at 0.63 (73); ctrl_t1's at 0.40 (10) and 0.52 (63). At 5+, both types continue at 0.97–1.00 in both arms.

### 5.2 How long runs end

All 58 well-formed runs of 10+ identical calls in `ctrl`, two passes, classed by the next call after the run:

| Exit | Runs |
|---|---|
| after the budget warning | 43 |
| at a context compaction | 14 |
| the task ended | 1 |
| on a clean step | **0** |

- The budget warning arrives in the 50th call's result: "Only 10 tool call(s) remaining (50/60 used). Finalize your edits and call submit_patch soon."
- A compaction summarises older turns of the history and can drop the earlier copies with them: an uncontrolled dedup.

### 5.3 Un-looping: the control's loop tasks at T = 1, same pass

| Pass | Loop tasks in ctrl | Not looping at T = 1 | Of those, solved by ctrl | Solved at T = 1 |
|---|---|---|---|---|
| 1 | 27 | 24 | 4 | 3 |
| 2 | 27 | 22 | 6 | 6 |
| Both | 54 | 46 | 10 | 9 |

- **In pass 1's 24 un-looped tasks, the freed budget went into exploration:**
  - stale calls fell from 1,019 to 123;
  - distinct calls per task rose from 15.1 to 32.8;
  - tasks with a source edit rose from 7 to 14, and source edits from 8 to 34.
- **The loops came early:** in 23 of the 24, ctrl's first stale call came before its first source edit. ctrl's longest run in these tasks was a command in 9, a search in 8, malformed in 6, and other in 1.
- **Loop task runs resolve less often:** 20% in ctrl (11 of 54), against 37% for the rest (61 of 164). For ctrl_t1 it is 25% (5 of 20) against 33% (66 of 198).
- **Of the 15 tasks that loop in both ctrl passes,** 4 are solved by at least one of the 7 runs. 12 of the 15 loop in neither T = 1 run.
- **The block, from the pass-1 split:**
  - The block un-looped 12 of pass 1's 27 and solved 5 of them. It solved 7 of the 27, against ctrl's 5.
  - In round 1, `block_v30` un-looped 18 of `ctrl_v30`'s 34 and solved 4.

### 5.4 Which tasks loop

| Pair | Loop tasks (each) | In both | Jaccard |
|---|---|---|---|
| ctrl pass 1, pass 2 | 27, 27 | 15 (chance ≈ 7) | 0.38 |
| ctrl pass 1, ctrl_v30 | 27, 34 | 20 | 0.49 |
| block pass 1, block_v30 | 29, 23 | 14 | 0.37 |
| ctrl_t1 pass 1, pass 2 | 8, 12 | **0** | 0.00 |
| ctrl pass 1, ctrl_t1 pass 1 | 27, 8 | 3 | 0.09 |

- **At T = 0.2 the task decides whether a loop happens, but not which call.**
  - 11 tasks loop in all three T = 0.2 control runs. Of those, T = 1 loops in 3 (in either pass).
  - Of the 15 tasks that loop in both ctrl passes, only 2 have their longest run on the same call.
  - Two ctrl passes share a median prefix of 3 identical calls (mean 4.1; 23 tasks diverge at call 0). Only 7 tasks are identical end to end.
- **At T = 1, loops are idiosyncratic:** the two passes share no loop task, and their call sequences share a median prefix of 1.

### 5.5 What T = 1 costs

- **Timeouts:** 18.0 against 11.5 per pass, +6.5 (+0.5 to +13.0), paired over tasks. In ctrl_t1 they fall almost all on tasks that don't loop (17 of 18 in pass 1).
- **Malformed calls:**
  - tasks with any: 24.5 against 15.0 per pass, +9.5 (+2.5 to +17.0);
  - tasks with a malformed run of 3+: +3.5 (−2.5 to +9.0);
  - the share of calls: +2.2 points (−3.2 to +7.5).
- **Agent time per task (dev):** 126 s against 111 s. Output tokens per task are 3,179 against 2,959.
- **Pass 1's "malformed doubling" was noise.** In pass 1 the v31 control had 261 malformed calls against v30's 138. Pass 2 had 174. They clump in a few tasks, such as `fastapi_14258` (54 and 44) and `rich_3043` (31 and 43).
- **The long T = 1 command runs** are mostly `python3 -c` probes that return output, in rich tasks: 37 runs, the longest 37 calls, on `rich.segment`.
  - The searches collapse: runs of 3+ fall from 39 to 7.
  - The commands fall less, from 50 to 37, and the two T = 1 passes differ there: 7 against 30.

### 5.6 What follows each exit (added 2026-10-19)

Well-formed runs of 3+ that end before the task does, classed as loop_report.py classes them.

| Arms | Exit | Runs | Mean length | Re-entered | Next call works | Source edit after | Stale share of the later calls |
|---|---|---|---|---|---|---|---|
| ctrl (2 passes) | budget warning | 58 | 26.2 | 10% | 67% | 28% | 38% |
| | compaction | 23 | 14.3 | 30% | 52% | 61% | 48% |
| | own | 6 | 3.5 | 50% | 33% | 33% | 88% |
| ctrl_v30 | budget warning | 29 | 30.4 | 10% | 86% | 34% | 33% |
| | compaction | 12 | 17.2 | 33% | 50% | 58% | 51% |
| ctrl_t1 (2 passes) | budget warning | 20 | 16.1 | 15% | 85% | 20% | 47% |
| | compaction | 9 | 6.9 | 0% | 67% | 67% | 30% |
| | own | 20 | 4.4 | 60% | 55% | 40% | 75% |
| block (both rounds) | budget warning | 35 | 16.3 | 9% | 80% | 31% | 27% |
| | compaction | 17 | 8.0 | 41% | 65% | 41% | 41% |
| | own | 107 | 9.3 | 62% | 41% | 47% | 73% |

- **At T = 0.2, a compaction ends a run but often doesn't end the loop.**
  - The agent sends the same call again later after 30–41% of compaction exits, against 9–10% after the budget warning.
  - About half the later calls are still stale.
  - Compaction exits do lead to a source edit more often (58–61% in the control, against 28–34%).
- **The two exits aren't comparable in length.** After the warning at most 10 calls remain, so re-entry there is low by construction.
- **Small counts:** 9–23 compaction exits per arm.

### 5.7 Where loop tasks sit by difficulty (added 2026-10-19)

The arXiv preprint 2610.07447 (§6.2) argues that outcomes vary most between runs at the frontier of what the agent can do. Here each task is placed by how many of the 7 closed-loop runs solve it (ctrl ×2, ctrl_t1 ×2, block, ctrl_v30, block_v30):

| Solved in | Tasks | Loop task runs | ctrl runs only (T = 0.2, 3 runs) | Tasks looping in at least one run |
|---|---|---|---|---|
| 0 of 7 | 58 | 97 of 406 (23.9%) | 54 of 174 (31.0%) | 38 |
| 1–6 of 7 | 29 | 53 of 203 (26.1%) | 29 of 87 (33.3%) | 20 |
| 7 of 7 | 22 | 10 of 154 (6.5%) | 5 of 66 (7.6%) | 4 |

- **Resolution is mostly all or nothing:** 80 of 109 tasks are solved by every run or by none. Solve counts are 0: 58, 1: 6, 2: 6, 3: 3, 4: 2, 5: 6, 6: 6 and 7: 22.
- **Loops are as common in sometimes-solved tasks as in never-solved ones,** and rare in tasks every run solves.
- **Within a task, a looping run loses.** In the 20 sometimes-solved tasks that have both looping and non-looping runs, the looping runs resolve 22.8 points less often (−41.1 to −4.3, over tasks).
  - With "has a well-formed run of 3+" as the split instead, the gap is −15.7 (−35.4 to +3.9), over 19 tasks.
- **This is a correlation within tasks, not an effect.** A run that is already lost may loop for that reason, and §5.3 shows that un-looping by temperature didn't recover these runs.

## 6. The predictions, against the result

Written in README.md §3 before the run.

| Prediction | Result | Verdict |
|---|---|---|
| `ctrl_t1` locks in less, but still locks in: lock-in at 3+ around 0.70–0.85, against 95.7% | 0.87 well-formed and 89.9% in the report, pooled at 3+. At k = 3–4: 0.67 on all steps, 0.77 on clean steps; the replay gave 0.73 | **Holds at the onset of a run.** The pooled 3+ figure is weighted by survivors: long runs, where the hazard is ~0.93, contribute most of its steps |
| Lock-in at k = 2 below 77.5% | 58.3% (report), 0.505 well-formed. But the v31 control is also lower: 69.0%, 0.672 | Holds. Against the v31 control, −10.7 (−25.0 to +2.6) in the report, −0.166 (−0.345 to −0.020) well-formed |
| Fewer and shorter runs of 5+, a lower copy share | Runs of 5+ per task 0.42 → 0.22; run length 19.8 → 11.2; copies 28.0% → 12.0% | Holds |
| Malformed calls may rise at T = 1 | Share +2.2 points (ns); tasks with any +9.5 per pass (+2.5 to +17.0) | Holds by tasks, not by share |
| No prediction for resolution | −0.5 points (−5.0 to +4.1) | – |
| Replication: round 1's directions recur in `block` − `ctrl` | lock-in 3+, runs 3+, run length and re-entry recur; next call works points the same way (ns); stale, copies and loop tasks don't | Partly. The shape replicates; the stale drop doesn't |

## 7. What it shows

**1. Lock-in is the model's, not the sampler's.**
- At T = 1 a run under way still continues 87% of the time at 3+, and 99% on clean steps at 10+.
- G1 and G2 predicted this. Their effects were in the exact probabilities (G1: exact p(c) 0.006 → 0.307 at four copies; G2: 0.79 at copy states). Low temperature hardens the preference; it doesn't create it.

**2. Temperature acts on the onset.**
- Most of T = 1's effect falls on the first two copies: copy starts after a stale call are halved, and k = 2 falls by 0.17. The hazard after that falls far less.
- Searches, whose empty results invite the same search again, almost stop looping (39 → 7 runs). Malformed runs don't change.
- That fits G2's selection result. Runs begin where the model already leans towards the call, and sampling at T = 1 escapes more of those leans before the copies lock them in.

**3. At T = 0.2, exits are exogenous.**
- After five identical calls, nothing the agent writes ends the run. 290 of 290 and 1,052 of 1,052 clean steps continue it.
- Only the harness's budget warning, a compaction, or the task's end changes the call. The warning is an instruction; the compaction removes history.
- This matches G2: an instruction moves the copy (round 1's note), and deleting the copies moves it most (dedup).

**4. Breaking loops is not the same as solving tasks.**
- T = 1 un-loops 46 loop task runs and frees their budget for distinct calls and source edits. The tasks solved stay at 9 against 10.
- In 23 of pass 1's 24 un-looped tasks the loop began before the first source edit, so the agent hadn't yet found where the change goes.
- Within sometimes-solved tasks, looping runs resolve 23 points less often (§5.7). So the loop marks a lost run. Breaking the loop with a global change of sampling doesn't find the way back.

**5. At T = 0.2, loop-proneness belongs to the task, not to the call.**
- The same tasks loop across passes (15 of 27, about twice chance) and across v30 and v31 (20).
- They rarely loop on the same call (2 of 15), and the passes diverge after about 3 calls. So it is the task that is sticky, not one particular state.
- At T = 1 the overlap vanishes.

**6. The block is a pattern break over whole tasks too.**
- It shortens runs (−9 calls), makes more of them start (+0.35 per task) and doubles copy starts (0.025 → 0.050).
- It also sends the agent back to old calls (re-entry +26 points). Its well-formed runs mostly end on their own (107 of 159 over both rounds, §5.6), and 62% of those re-enter.
- The net stale share doesn't move on v31. G2's one-step picture holds: the block turns copies into other stale calls.

**7. v31 changes nothing measurable for the 31B's loops.** ctrl − ctrl_v30 includes 0 on every metric. The largest shift, "next call works" −13.3 (−27.8 to +2.0), isn't significant.

## 8. What it doesn't show

- *Small effects on resolution.* Two passes of 109 tasks can't detect a change of 1–2 tasks, and an inconclusive difference is not equivalence.
- *A replicated block arm on v31.* `block` ran once. Its pairs pool one pass against two.
- *Why runs end at a compaction.* The harness's compaction is periodic and uncontrolled. §5.6's re-entry rates come from 9–23 runs per arm, and whether the summary keeps the latest copies isn't checked.
- *That loops cost resolution.* §5.7 is a correlation within tasks, and §5.3's natural experiment finds no rescue.
- *Other temperatures, or top-p/top-k.* Only Google's T = 1 setting was run, by design. README.md §3 rules out a grid.
- *Generality.* One model (Gemma 4 31B int4), thinking off, one harness, and four repositories (fastapi, rich, requests, httpx).

## 9. What it changes

- **The block's claim now uses v31's result** (the user's call, 2026-10-19). The old claim, "the stale share falls 5–6 points", held on v30 only, and even there its interval included 0 (−5.7, −12.3 to +0.3). On v31 the change is +1.6.
  - loops/README.md's summary (line 10) now states v31's result.
  - Round 1's result line ("Result (2026-10-12)") and loops_reason_10_13.md §1 keep their v30 text, each with a note that v31 supersedes it.
- **loops_reason_10_13.md §3's temperature argument** can now cite §5.1: lock-in at T = 1, measured in the closed loop. Not yet edited.
- **G4:** G2 failed G4's condition. G3 adds that exits come only from changes to the context, and that compactions make them by accident (§5.2) but are often followed by re-entry (§5.6).
  - The proposal for G4's slot stands: a closed-loop dedup arm against v31's `ctrl`, two passes, about 3 A100-hours (ablation_G2.md §8; ablation_recap_10_17.html, "What we would try next").
  - Measure it by stale share, re-entry and what the freed calls buy, before resolution.
  - §5.3 and §5.7 set the prior: fewer loops, little change in tasks solved.
  - Not decided.
- **For the paper:**
  - T = 1 is the judges' expected arm (literature.md). Its answer is "fewer loops start, they still lock in, and resolution doesn't move".
  - Report the hazard by run length (§5.1) and the clean-step exits (§5.2), not the pooled lock-in at 3+, which is weighted by survivors.
