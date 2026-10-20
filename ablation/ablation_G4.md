# ablation/ablation_G4.md: dedup over whole tasks (round 2's G4)

<!-- written 2026-10-20 from $R/paper/loops/report_g4_p1_1019-2124 and report_g4_p2_1019-2337 (loops.md, baseline_dev.md,
     baseline_rich.md), with $R = /scratch/zha.j/gemma4. The tables of §3 and §4 are report_g4_p2's, which pools both
     dedup passes with G3's arms and round 1's v30 runs. The splits of §5 come from local analyses of copied traces in
     session scratchpads, not in this repo; they resample tasks 2,000 times (seed 0), as loop_report.py does. The design
     and the predictions, written before the run, are in ../loops/README.md §3 (G4). File names without a folder below
     (README.md, loop_report.py, submit_loops.sh, ...) are in ../loops/. G1's, G2's and G3's records are ablation_G1.md,
     ablation_G2.md and ablation_G3.md; G4's correction to G2 is in ablation_G2.md §6.6. The consolidated findings of
     every round are in ../findings.md. -->

**The question:** G2's strongest one-step lever was deleting the earlier copies of the call. G4 carries it over whole tasks with a proxy that collapses every run of identical tool turns to its first. Does that cut loops? And with the model's request held the same at every step of a run, how much of the lock-in curve is selection?

**The answer:**
- **Dedup doesn't cut loops.** Copies are 25.4% of calls against the control's 28.0% (−2.6 points, −8.2 to +2.8). The stale share, runs of 3+ per task, run length and loop tasks don't move either. Every prediction about the loops failed (§6).
- **Resolution doesn't move:** 73 of 218 task runs, against 72 (+0.5 points, −3.7 to +4.6).
- **The state where a run starts carries most of the lock-in.** With the request held the same at every step, the agent sends the call again 78% of the time after its first copy, then 87%, 93% and 99.3% by the 10th (clean steps, both passes). The control's curve is 68%, 96%, 100% and 100%.
- **The visible copies add the last few points.** They remove the escapes left in the middle of a run.
  - Under dedup the model ends 47% of its runs of 3+ itself, against 15% in the control (+32 points, +20 to +45). Almost all of these are runs of 3–9 calls.
  - Runs of 10+ fall from 0.33 to 0.25 per task (−0.08, −0.16 to −0.01). So about three quarters of the long runs need no copies in view.
- **The same tasks loop.** A dedup pass and a control pass share as many loop tasks as two control passes do.
- **The model's own empty thought comes back inside runs** (60–84% of steps, against 6%). Every one is empty, and it doesn't stop the copy.
- **G2's dedup was mostly another intervention.** In 44 of its 56 copy states, the call had also been sent before the run, and G2's dedup deleted those calls too. It built histories the agent never had. [ablation_G2.md](ablation_G2.md) §6.6 is corrected.

## 1. Design (from README.md §3)

[submit_loops.sh](../loops/submit_loops.sh) (steps `smoke` and `dedup`), with [arms.tsv](../loops/arms.tsv), [loop_report.py](../loops/loop_report.py) and [g4_smoke_ids.txt](../loops/g4_smoke_ids.txt). In the main track: `baseline/token_proxy.py --dedup`, with `baseline/eval_gpu.slurm`'s `DEDUP` knob. README.md §3 (G4) has the full design and the predictions; "Tested: G4's dedup arm" has the local tests.

| Arm | Zip | Chat template | Proxy | Sampling |
|---|---|---|---|---|
| `dedup` | `off_args`'s pinned zip (`532374430fe2`) | the model's | `DEDUP=2` | T = 0.2, top-p 0.95, as `ctrl` |

- **Against G3's arms:** `ctrl` (two passes on v31) and `block` (one), on the same 109 tasks (70 dev-sound and 39 rich-sound). PINS=1, unpaced, 4 tasks at once on one A100.
- **What the proxy does:** before it forwards a chat request, every run of 2 or more identical tool turns in a row (the same calls with the same results) becomes its first turn.
  - The harness executes and counts every call, and its trace keeps them; only what the model sees changes.
  - So while the agent re-sends a call that gets the same result, it is asked the same question again: the request is byte for byte the one after the run's first call.
  - A different result breaks a run. Returns (a call re-sent after other calls) stay as they are.
  - No seed is sent, so each repeated request is a fresh draw at T = 0.2.
- **Why it measures selection:** in `ctrl` each copy also enters the request. Under dedup it doesn't, so P(one more | k) can rise with k only through which states keep looping. The gap between the two curves at each k is what the visible copies add.
- **Planned and ran:** a 4-task smoke test, then two passes.

## 2. The runs

- **The smoke test** (`STEPS=smoke`): job 11247110, on d1029, 21:13–21:19 EDT on 2026-10-19, into `loops_dedup_smoke_v31_a100`.
  - Its 4 tasks (`fastapi_13920`, `fastapi_14485`, `fastapi_15023`, `rich_3506`) each loop on one well-formed call with the same result in all three T = 0.2 control runs, and no run ever solved them.
  - The proxy collapsed 133 of 192 requests (the longest run 43 calls). The requests that should have been collapsed, 69 by the traces, are the 69 it collapsed.
  - All 4 still looped, and none was solved.
- **Pass 1** (`STEPS="dedup report"`): job 11247673 on d1028, 21:32–23:05 EDT. The chained job 11247675 found nothing left. Report 11247677 → `report_g4_p1_1019-2124/`.
- **Pass 2** (`PASS=2 STEPS="dedup report"`): job 11250150 on d1029, 23:55 on 2026-10-19 to 01:07 on 2026-10-20. The chained job 11250153 found nothing left. Report 11250154 → `report_g4_p2_1019-2337/`, which reads both passes.
- **Every run is complete:** 109 of 109 tasks, none skipped.
- **Provenance:** every run ran at the main track's `bd82ee8` and the paper track's `c632d6d`, with G3's harness (`067143a2f3ec`), vLLM build (`73c69bafca24`), zip and sampling. The report's provenance differs from `ctrl` only in the dedup column (`runs>=2`).
- **Compute:** about 2.9 A100-hours (smoke 6 min, passes 1 h 32 and 1 h 12, no-op jobs 2 min). The estimate was 3.
- **What the proxy did:**

| Pass | Requests | Collapsed | Turns dropped | Longest run | Expected / seen |
|---|---|---|---|---|---|
| 1 | 4,010 | 1,395 (34.8%) | 20,180 | 65 | 807 / 807 |
| 2 | 4,121 | 1,378 (33.4%) | 24,894 | 69 | 956 / 956 |

"Expected" counts the requests that followed a run of identical calls with the same result, from the traces; "seen" counts those whose last run the proxy collapsed.

- **Task endings per run:**

| Run | Resolved | Timeouts | Context window |
|---|---|---|---|
| `dedup` pass 1 / 2 | 37 / 36 | 15 / 17 | 4 / 3 |
| `ctrl` pass 1 / 2 (G3) | 35 / 37 | 11 / 12 | 4 / 4 |

## 3. Arms, passes pooled (from report_g4_p2's loops.md)

All 109 tasks. Copies repeat the call just before; returns repeat an earlier one. Loop tasks are task runs where at least half the calls are bad (stale or malformed). "After a run of 3+" counts runs that end before the task does.

| Arm | Runs | Task runs | Resolved | Stale: copies / returns | Malformed | Lock-in k=2 / 3+ | Runs 3+ / 5+ per task | Run length (3+) | Loop tasks | After a run of 3+: re-entered / next works / source edit after (runs) |
|---|---|---|---|---|---|---|---|---|---|---|
| ctrl | 2 | 218 | 33.0% | 38.6%: 28.0% / 10.6% | 5.5% | 69.0% / 94.8% | 0.53 / 0.42 | 19.81 | 24.8% | 21.5% / 60.7% / 37.4% (107) |
| dedup | 2 | 218 | 33.5% | 37.3%: 25.4% / 11.8% | 6.5% | 76.1% / 95.1% | 0.50 / 0.37 | 19.40 | 26.1% | 31.9% / 67.0% / 37.4% (91) |
| block | 1 | 109 | 34.9% | 40.2%: 27.9% / 12.3% | 5.4% | 69.6% / 89.2% | 0.88 / 0.56 | 10.86 | 26.6% | 47.3% / 52.7% / 41.8% (91) |
| ctrl_t1 | 2 | 218 | 32.6% | 20.4%: 12.0% / 8.4% | 7.7% | 58.3% / 89.9% | 0.39 / 0.22 | 11.18 | 9.2% | 35.1% / 57.1% / 40.3% (77) |

By task group:

| Arm | Dev (70): resolved per run | Dev: stale / loop tasks | Rich (39): resolved per run | Rich: stale / loop tasks |
|---|---|---|---|---|
| ctrl | 24, 25 | 32.7% / 17.9% | 11, 12 | 48.2% / 37.2% |
| dedup | 24, 26 | 29.5% / 17.1% | 13, 10 | 49.8% / 42.3% |

- `baseline_dev.md` against `ctrl`: `dedup` +0.5 tasks (−2.5 to +4.0), 4 tasks better and 4 worse. `baseline_rich.md`: +0.0 (−3.0 to +3.0), 4 better and 3 worse.
- **Runs of 3+ by type:** search 34, command 55, read 3, edit 3, malformed 13 (`ctrl`'s two passes: 39, 50, 1, 0, 26).
- **How runs of 3+ end** (summed over dev and rich):

| Run | Task end | Budget warning | Compaction | Own |
|---|---|---|---|---|
| dedup pass 1 / 2 | 6 / 11 | 20 / 21 | 4 / 3 | 28 / 15 |
| ctrl pass 1 / 2 | 4 / 5 | 28 / 30 | 18 / 15 | 9 / 7 |

## 4. Paired differences (from report_g4_p2's loops.md)

A − B on the 109 tasks, with 95% intervals from 2,000 resamples of tasks. Shares are in percentage points. The pass-1 column is report_g4_p1's, which held one dedup pass.

| Metric | dedup − ctrl | dedup − ctrl, pass 1 only | dedup − block |
|---|---|---|---|
| resolved | +0.5 (−3.7 to +4.6) | +0.9 (−4.1 to +6.0) | −1.4 (−8.3 to +5.0) |
| stale | −1.3 (−6.0 to +3.1) | −0.5 (−6.6 to +5.4) | −2.9 (−9.4 to +4.0) |
| copies | −2.6 (−8.2 to +2.8) | −4.2 (−11.0 to +2.4) | −2.5 (−10.1 to +5.1) |
| returns | +1.3 (−1.4 to +4.3) | **+3.7 (+0.2 to +7.7)** | −0.4 (−3.7 to +2.9) |
| malformed | +1.0 (−1.4 to +3.6) | −0.9 (−5.2 to +3.1) | +1.1 (−3.4 to +5.3) |
| lock-in k=2 | +7.0 (−6.8 to +20.4) | +12.6 (−3.4 to +28.7) | +6.5 (−4.0 to +17.6) |
| lock-in k=3+ | +0.3 (−1.0 to +1.6) | −0.9 (−3.4 to +1.0) | **+5.9 (+3.0 to +9.3)** |
| runs 3+ per task | −0.04 (−0.16 to +0.08) | +0.00 (−0.13 to +0.14) | **−0.39 (−0.63 to −0.16)** |
| runs 5+ per task | −0.06 (−0.15 to +0.04) | −0.06 (−0.17 to +0.04) | **−0.19 (−0.35 to −0.05)** |
| run length (3+) | −0.41 (−4.73 to +4.02) | −2.93 (−8.06 to +2.30) | **+8.53 (+4.20 to +12.76)** |
| loop tasks | +1.4 (−4.1 to +6.9) | +3.7 (−3.2 to +10.6) | −0.5 (−9.6 to +8.7) |
| re-entered | +10.4 (−0.1 to +20.8) | +9.3 (−4.4 to +23.2) | −15.4 (−30.6 to +0.2) |
| next call works | +6.3 (−6.2 to +18.7) | +2.7 (−13.2 to +16.8) | +14.3 (−1.1 to +27.5) |
| source edit after | −0.0 (−16.0 to +15.9) | +6.8 (−12.3 to +25.7) | −4.4 (−22.7 to +14.2) |
| tasks with a source edit | +0.0 (−6.9 to +6.9) | −0.5 (−8.7 to +8.3) | −5.0 (−12.8 to +3.2) |

Bold: the interval excludes 0. Against `ctrl`, nothing does once both passes are pooled. Against `block`, dedup differs where the block does (G3): the block's runs are shorter and more frequent.

**Pass to pass** (run 2 − run 1), the noise floor:

| Arm | Resolved | Stale | Copies | Returns | Lock-in k=3+ | Runs 3+ per task |
|---|---|---|---|---|---|---|
| ctrl | +1.8 (−2.8 to +7.3) | −0.5 (−7.1 to +6.5) | −1.7 (−9.1 to +6.4) | +1.2 (−2.5 to +5.0) | +0.2 (−1.3 to +1.7) | −0.02 (−0.19 to +0.16) |
| dedup | −0.9 (−8.3 to +5.5) | −1.7 (−9.5 to +5.3) | +3.1 (−4.8 to +11.3) | **−4.8 (−10.0 to −0.1)** | **+2.2 (+0.3 to +4.7)** | −0.07 (−0.23 to +0.08) |

Two identical dedup passes "differ" on returns and on lock-in at 3+. Pass 1's returns gain (+3.7) is the size of that noise, and pooling took it to +1.3.

## 5. Local splits

### 5.1 The lock-in curve with the request held fixed

P(the next call is identical | the last k calls were identical, the last one well formed), on clean steps: no budget warning in the last result, no compaction before the next call. "Fully collapsed" keeps the dedup steps where the proxy collapsed the whole run, so the request was exactly the one after the run's first call: 653 of 678 clean steps at k ≥ 2 in pass 1, and 664 of 675 in pass 2. The rest had a result that changed within the run, such as a timestamp.

| Steps | k = 1 | k = 2 | k = 3–4 | k = 5–9 | k = 10+ |
|---|---|---|---|---|---|
| ctrl, 2 passes | 0.024 (4,985) | 0.678 (115) | 0.958 (142) | **1.000 (290)** | **1.000 (1,052)** |
| dedup pass 1, fully collapsed | – | 0.833 (54) | 0.823 (79) | 0.913 (126) | 0.992 (394) |
| dedup pass 2, fully collapsed | – | 0.720 (50) | 0.924 (66) | 0.947 (131) | 0.993 (417) |
| **dedup, both passes, fully collapsed** | – | **0.779 (104)** | **0.869 (145)** | **0.930 (257)** | **0.993 (811)** |
| dedup, both passes, all clean steps | 0.022 (5,142) | 0.793 (111) | 0.877 (155) | 0.934 (271) | 0.993 (816) |

- **Paired over tasks, all steps** (report_g4_p2), dedup − ctrl: k = 1 −0.2 points (−1.0 to +0.4); k = 2 +7.6 (−8.9 to +22.4); k = 3–4 **−7.0 (−13.3 to −0.8)**; k = 5–9 **−4.1 (−7.3 to −1.1)**; k = 10+ +0.6 (−0.5 to +1.8).
- **The curve rises with k although the request doesn't change.** Each repeated request is a fresh draw from the same prompt, so a state's chance to continue is the same at every step of its run. The rise comes from which states keep looping: those near 1 survive, and by the 10th call almost only they are left.
- **At the first copy, the visible copy adds nothing measurable:** at k = 2 dedup is at or above the control. The control's two copies in view don't make a third likelier than one copy does.
- **From k = 3 the copies close the remaining escapes.** At 3–9 the control continues 96–100% and dedup 87–93%. In odds, 0.87 → 0.96 is about 3.5 times.
- **Each pass's k = 2 rests on about 55 steps in about 40 tasks,** and the two passes give 0.83 and 0.72. Quote the pooled value.
- **G4's prior,** built from G2's dedup states (README.md §3: 0.29 at k = 2, then 0.67, 0.79, 0.85, 0.88, 0.95), was far too low at the start of the curve. Those states were mostly histories the agent never had (§7.8).

### 5.2 How runs end

Runs of 3+ that end before the task does, both passes of each arm:

| Arm | Runs of 3–4: own exits | 5–9: own | 10+: own | 10+: budget warning / compaction / task end |
|---|---|---|---|---|
| ctrl | 12 of 24 (50%) | 3 of 20 (15%) | 1 of 72 (1%) | 43 / 23 / 5 |
| dedup | 19 of 28 (68%) | 18 of 26 (69%) | 6 of 54 (11%) | 31 / 2 / 15 |
| ctrl_t1 | 22 of 37 (59%) | 12 of 19 (63%) | 2 of 28 (7%) | 10 / 13 / 3 |
| block (1 pass) | 30 of 35 (86%) | 14 of 24 (58%) | 18 of 37 (49%) | 10 / 6 / 3 |

The run counts here include runs that end with the task; the "own" counts and shares below don't.

- **Own exits, over the runs that end before the task:** dedup 43 of 91 (47.3%) against ctrl's 16 of 107 (15.0%), +32.3 points (+19.6 to +44.5).
  - By pass: 28 of 52 (53.8%) and 15 of 39 (38.5%). Pass 2 − pass 1 is −15.4 (−37.8 to +4.7). Each pass alone is above the control.
- **After an own exit,** the next call is fresh 30 times and another stale call 13 times (ctrl: 9 fresh, 2 stale, 5 malformed).
- **Per task, paired** (dedup − ctrl, two passes each):
  - runs of 10+: 0.25 against 0.33, **−0.08 (−0.16 to −0.01)**;
  - runs of 3–9: 0.25 against 0.20, +0.05 (−0.04 to +0.13);
  - calls inside runs of 3+: 9.6 against 10.5, −0.9 (−3.3 to +1.3);
  - calls per task: 36.5 against 36.6.
- **So dedup moves about a quarter of the long runs into short ones,** and the calls they spend barely change. Its long runs end as the control's do: at the budget warning or with the task. Compactions end few of them (2 against 23), since the proxy has already removed what a compaction would.

### 5.3 Re-entry, by how the run ended

Runs of 3+ that end before the task; re-entered means the run's call is sent again, stale, later in the task.

| Arm | Own | Budget warning | Compaction | All |
|---|---|---|---|---|
| ctrl | 7 of 16 (44%), 29 calls left | 6 of 58 (10%), 4 left | 10 of 33 (30%), 18 left | 21% |
| dedup | 24 of 43 (56%), 34 left | 2 of 41 (5%), 3 left | 3 of 7 (43%), 28 left | 32% |
| ctrl_t1 | 19 of 36 (53%), 25 left | 3 of 20 (15%), 4 left | 5 of 21 (24%), 18 left | 35% |
| block | 36 of 62 (58%), 20 left | 3 of 18 (17%), 3 left | 4 of 11 (36%), 17 left | 47% |

- **Dedup's re-entry gain (+10.4 points, −0.1 to +20.8) comes from its mix of exits.** An own exit leaves about 30 calls and is re-entered about half the time in every arm. The budget warning leaves 3–4. Dedup simply has more own exits.
- Within own exits, 56% against 44% rests on 16 control runs.

### 5.4 Which tasks loop

| Pair | Loop tasks (each) | In both | Chance | Jaccard |
|---|---|---|---|---|
| ctrl pass 1, pass 2 | 27, 27 | 15 | 6.7 | 0.38 |
| dedup pass 1, pass 2 | 31, 26 | 17 | 7.4 | 0.42 |
| ctrl and dedup (4 pairs) | 26–31 | 15–19 | 6.4–7.7 | 0.39–0.49 |
| ctrl_t1 pass 1, pass 2 | 8, 12 | 0 | 0.9 | 0.00 |

- Over the four T = 0.2 runs (two of each arm), 59 tasks never loop, 19 loop once, 10 twice, 12 three times and 9 in all four.
- **Every task that loops in both control passes loops in at least one dedup pass.** Two tasks (`rich_3535`, `rich_3777`) loop in both dedup passes and in neither control pass.
- **So at T = 0.2, which tasks loop is a property of the task,** with or without the copies in view (G3 §5.4). Only T = 1 changes it.

### 5.5 The model's own thoughts inside runs

Clean steps after a well-formed call, from the proxy's `calls.jsonl` (the control has it for pass 2 only).

| Run | Opens a thought at k = 1 | At k ≥ 2 | Continues the run, with / without a thought | Thoughts with text |
|---|---|---|---|---|
| ctrl pass 2 | 43.8% (2,529) | 6.3% (790) | 78.0% (50) / 98.6% (740) | 0 |
| dedup pass 1 | 40.3% (2,581) | 59.6% (678) | 96.5% (404) / 91.6% (274) | 0 |
| dedup pass 2 | 40.9% (2,561) | 83.9% (675) | 96.6% (566) / 90.8% (109) | 0 |

- **Under dedup the model keeps writing the empty thought inside runs,** and more the longer the run: 90% of steps at k = 10+ in pass 2. In the control it stops: 0 of 518 steps at k = 10+.
- **Every opened thought is empty,** `<|channel>thought\n<channel|>` and then the call, in both passes and in the control. At T = 1 too: in ctrl_t1's pass 2, one opened thought isn't empty, and it holds only a stray channel token.
- **It doesn't stop the copy:** with it the run continues 96.6%, without it 90.8%.
- G2 §7 found this one step ahead: deleting the copies brings back the empty thought (11% → 58% at copy states). The closed loop replicates it. The run of identical turns in view is what suppresses the thought; the request after the run's first call doesn't.

### 5.6 Resolution

| | ctrl, 2 passes | dedup, 2 passes |
|---|---|---|
| Resolved | 72 of 218 | 73 of 218 |
| Loop task runs: resolved | 11 of 54 (20%) | 6 of 57 (11%) |
| Other task runs: resolved | 61 of 164 (37%) | 67 of 161 (42%) |

- **Solves per task** (ctrl, dedup; out of 2 each): 63 tasks 0 and 0; 27 tasks 2 and 2; the other 19 split 5 (0, 1), 1 (0, 2), 2 (1, 0), 4 (1, 1), 2 (1, 2), 1 (2, 0) and 4 (2, 1).
- **Un-looping by dedup:** of the 15 tasks that loop in both control passes, 6 don't loop in one dedup pass. Those 6 task runs resolve 2 times, and the control solves the same 6 tasks 4 times in 12 runs.
- No change of 1–2 tasks can be seen with two passes.

## 6. The predictions, against the result

Written in README.md §3 before the run (`dedup` against `ctrl`, passes pooled, well-formed calls for the hazards).

| Prediction | Result | Verdict |
|---|---|---|
| A check: copy starts (k = 1) within 1 point of `ctrl`'s 2.5% | 2.3% (5,677 steps); per run 1.2–2.8% | Holds |
| Lock-in at k = 2 falls from 67% to 15–35% | 76.1% in the report, 0.748 well formed, 0.779 on fully collapsed clean steps | **Fails** |
| The hazard still rises with k: k = 3–4 above k = 2; k = 10+ at 85% or more on clean steps, on at most a third of `ctrl`'s 1,052 clean steps | 0.78 → 0.87 → 0.93 → 0.99; 816 clean steps at 10+ (78% of the control's) | The rise holds; the count fails |
| Copies fall by half or more (28.0% → 14% or less) | 25.4% | **Fails** |
| Runs of 5+ per task 0.42 → 0.25 or less; runs of 3+ shorter | 0.37; 19.4 calls against 19.8 | **Fails** |
| The stale share falls by 8 points or more (→ 31% or less) | 37.3% (−1.3) | **Fails** |
| Returns rise by 0–4 points | +1.3 | Holds |
| After a run of 3+, the next call works at least as often as in `ctrl` (60.7%), and more often than in `block` (52.7%) | 67.0%: +6.3 (−6.2 to +18.7) and +14.3 (−1.1 to +27.5) | Holds in direction; neither interval excludes 0 |
| No prediction for resolution | +0.5 (−3.7 to +4.6) | – |

The failed predictions share one cause: they were scaled on G2's dedup effect, and that effect came mostly from deleting calls made before the run (§7.8).

## 7. What it shows

**1. Deleting the copies doesn't cut loops, because the state does most of the work.**
- With the request held the same, a state that has produced one copy produces another 78% of the time. By the 10th re-send it is 99%: the states that keep looping are near-certain of the call.
- This is the review's open question in the closed loop (loops_reason_10_13.md). Most of the lock-in curve's height is selection: runs start where the model already strongly prefers the call.

**2. The copies make a run absorbing.**
- They add little at the first copy and close the last escapes after it: at 3–9 identical calls, 96–100% against 87–93%.
- In the control, nothing inside a long run ends it at T = 0.2 (G3 §5.2). Without the copies in view, short runs end on their own (68–69% of runs of 3–9), and long runs still end from outside.
- With G1: at fixed histories, added copies raise p(c) (0.002 → 0.43 for 1 → 4 copies). So both halves are real. The closed loop shows their sizes: the state takes the hazard to about 0.8–0.9, and the copies take it the rest of the way to 1.

**3. Two kinds of long run.**
- About three quarters of the control's runs of 10+ persist with the copies out of view; the rest end early under dedup (runs of 10+ per task −0.08).
- The first kind is a near fixed point of the policy. The second is built by the copies, as in G1.
- G2's clean long-run states look like the second kind: with the run collapsed, p(c) at the run's sampling falls from 1.00 to 0.08–0.35 (4 states in 2 tasks, v30 histories; ablation_G2.md §6.6).

**4. Which tasks loop doesn't change.** The overlap between a dedup pass and a control pass equals the overlap between two control passes (§5.4). Loop-proneness at T = 0.2 belongs to the task.

**5. The empty thought goes with the copies in view, and it isn't a lever.** Deleting the copies brings it back, and the copy goes on through it (§5.5). The always-block template works differently: it puts the thought into the request, which breaks the match with the earlier copies (ablation_G2.md §6.3). A real thought is another lever again: T-open's forced thoughts escape 58% where the model skipped thinking (loops_reason_10_13.md §5).

**6. Breaking loops is again not solving tasks.** Resolution stays at 73 against 72. With T = 1 (G3) and the block, this is the third whole-task lever that changes loops, here only their ends, and leaves resolution where it was.

**7. "A stuck agent is a confident agent," measured in the closed loop.** From the same request, re-drawn at T = 0.2, the agent sends the same call again 78–99% of the time. This is the probability of a fresh draw, not calibrated confidence, but it is the agent's own behaviour, not a replay.

**8. G2's dedup was mostly another intervention.** G2's dedup (`gate1.dedup`) deletes every earlier copy of the call anywhere in the context and keeps the latest. In 44 of its 56 copy states the call had also been sent before the run, so the result was a history the agent never had; there exact p(c) fell from 0.90 to 0.07. In the 12 clean states, where the call occurs only in the run, it fell from 0.40 to 0.08 (6 tasks). G4's proxy collapses only the run, so its request is one the agent did see. [ablation_G2.md](ablation_G2.md) §6.6 has the correction.

## 8. What it doesn't show

- *Small effects on resolution.* Two passes of 109 tasks can't detect a change of 1–2 tasks, and an inconclusive difference is not equivalence.
- *Whether the selection is by state or by task.* The fixed request includes the whole history so far, and the same tasks loop across passes (§5.4); the two aren't separated.
- *A closed-loop version of G2's dedup.* Deleting earlier returns too builds histories the agent never had; G4 didn't run it, by design.
- *Dedup at the 3rd identical call or later* (`DEDUP=3` and up). Only runs of 2 or more were collapsed.
- *Precise k = 2 hazards.* About 50–60 steps in about 40 tasks per pass.
- *Kaggle.* A zip can't change the requests, so this tests a mechanism, not a submission.
- *Generality.* One model (Gemma 4 31B int4), thinking off, one harness, and four repositories (fastapi, rich, requests, httpx).

## 9. What it changes

- **[ablation_G2.md](ablation_G2.md) is corrected** (2026-10-20): its summary, §6.4 and §6.6 now say what G2's dedup deleted, with the 44 / 12 split; its old numbers stay, marked.
- **[README.md](../loops/README.md) §3:** G4's status and result; the summary at its top gains a G4 bullet.
- **The gated training ideas stay closed.** Exit distillation and the clipped-rollout A/B were gated on G4 gaining whole tasks; it gains none.
- **The experiments are frozen** (the user's call, 2026-10-20), and A0, the CPU rollout-value audit, is dropped. A 12B control pass is still open as a generality check.
- **For the paper:**
  - Report the fixed-request curve (§5.1) beside the control's. The gap is what the visible copies add; the height is selection.
  - Report own exits by run length (§5.2) instead of the pooled lock-in at 3+.
  - The consolidated findings are in [../findings.md](../findings.md).
