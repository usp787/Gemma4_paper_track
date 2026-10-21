# ablation/ablation_control.md: the 12B control (generality)

<!-- written 2026-10-21 from $R/paper/loops/report_12b_p1_1020-1836 and report_12b_p2_1020-2212 (loops.md,
     baseline_dev.md, baseline_rich.md), with $R = /scratch/zha.j/gemma4. The tables of §3 and §4 are report_12b_p2's,
     which pools both 12B passes with every earlier arm. The splits of §5 come from local analyses of copied traces in
     a session scratchpad, not in this repo; they resample tasks 2,000 times (seed 0), as loop_report.py does. The
     design and the predictions, written before the run, are in ../loops/README.md §3 ("The 12B control"). File names
     without a folder below (README.md, loop_report.py, submit_loops.sh, ...) are in ../loops/. The 31B's arms are
     recorded in ablation_G3.md (ctrl) and ablation_G4.md; the consolidated findings are in ../findings.md. -->

**The question:** every result so far comes from one model, Gemma 4 31B. Do the paper's descriptive claims hold on a second size? The claims: lock-in rises with run length, long runs end only from outside at T = 0.2, loops mark lost runs, and the same tasks loop. One closed-loop arm, `ctrl` with only the model changed to Gemma 4 12B, run twice.

**The answer:**
- **The lock-in curve is Gemma 4's, not the 31B's.** Once a copy has been sent, the two sizes continue at the same rates. On clean steps the 12B sends the call again 69%, 95%, 99.0% and 99.7% of the time (k = 2, 3–4, 5–9, 10+), against the 31B's 68%, 96%, 100% and 100%.
- **Long runs end from outside.** Of the 12B's 119 runs of 10+, 4 end on their own (3%). The rest end at the budget warning, a compaction or the task's end.
- **Size changes how often a run starts, not how firmly it locks in.** The 12B copies the call just before 2.4 times as often after a fresh call: 5.8% against 2.4%. So it has more runs (0.88 against 0.53 per task), more loop tasks (44.5% against 24.8%) and more stale calls (53.4% against 38.6%).
- **It loops on different calls, the same way.** The 12B writes almost no malformed calls (0.2% against 5.5%), so it has no malformed loops. Instead it repeats edits that can't apply: 25 of its 26 edit runs return `FileEditError` at every call. It also re-sends truncated reads from the same line. In both sizes, an error coming back doesn't stop the copy.
- **Outside exits work less well on the 12B.** After the budget warning it ends 51 of 81 long runs before the task does (the 31B: 43 of 44). It also comes back to a call it left more often after every kind of exit (re-entry 40.5% against 21.5%).
- **Loops mark lost runs only weakly at this size.** The 12B's loop task runs resolve 8%, its other runs 13% (+5.0 points, −3.0 to +13.3). Most of the size gap isn't loops: at its non-loop rate, the 12B would gain about 5 of the 48 tasks it trails by.
- **The same tasks loop, in part.** The two sizes share more loop tasks than chance gives within each split (p = 0.015), and 14 of the 15 tasks that loop in both 31B passes loop for the 12B too. The 12B also loops on many tasks the 31B never loops on.
- **Resolution falls to 24 of 218,** against 72 (−22.0 points, −29.8 to −15.1), as predicted.
- **The same bytes, different tools.** The 12B calls the analyzer sub-agent in about half its tasks; the 31B never does in any closed-loop run. The sub-agent's calls count toward the 60-call cap, and about 40% of its invocations end on the sub-agent's context window.

## 1. Design (from README.md §3)

[submit_loops.sh](../loops/submit_loops.sh) (steps `modelzip` and `ctrl_12b`), with [make_model_zip.py](../loops/make_model_zip.py), [loops.slurm](../loops/loops.slurm) (`STEP=modelzip`), [arms.tsv](../loops/arms.tsv) and [loop_report.py](../loops/loop_report.py) (a model column). Nothing changed in the main track: `baseline/eval_gpu.slurm` already serves the 12B, with its `--hf-overrides`, when the zip declares it. README.md §3 "The 12B control" has the full design and the predictions; "Tested: the 12B control" has the local tests.

| Arm | Zip | Chat template | Proxy | Sampling | Model |
|---|---|---|---|---|---|
| `ctrl_12b` | the pinned zip with its two `model:` lines set to the 12B (`…_ebc522f_12b.zip`, `b290324c906c`) | the model's | none (`DEDUP=0`) | T = 0.2, top-p 0.95, as `ctrl` | `gemma-4-12b-it-qat-w4a16-ct` (int4, w4a16) |

- **Against `ctrl`** (two passes on v31, G3), on the same 109 tasks (70 dev-sound and 39 rich-sound). PINS=1, unpaced, 4 tasks at once on one A100.
- **Only the model changes.** The agent, its prompts, tools, sampling and budget (4 min, 60 calls, 100 turns) are `ctrl`'s byte for byte.
  - The analyzer sub-agent runs on the 12B too, since a submission declares one model for all its agents.
  - The 12B's tokenizer is byte-identical to the 31B's. Its chat template differs only in 9 lines for image and audio parts, with the same empty-thought prefill when thinking is off.
- **Why closed loop only:** the one-step claims (G1, G2, G4's selection) would need their own replay states, so they weren't repeated.
- **Planned and ran:** two passes, no smoke test (the user's call, 2026-10-20).

## 2. The runs

- **Pass 1** (`STEPS="modelzip ctrl_12b report"`), 2026-10-20 EDT:
  - modelzip 11269587 (52 s) built the zip on Explorer, and the host compiler's check of it, 11269589, took 1 min. Its bytes equal the local build (`b290324c906c`), and `12b_zip.check.md` passes its 10 checks. Exactly two lines differ from the pinned zip: `agent.yaml:14` and `sub_agents/code_analyzer.yaml:3`.
  - The arm: job 11269599 on d1028, 18:49–20:13 (1 h 23), all 109 tasks. The chained 11269601 found nothing left.
  - Report 11269602 → `report_12b_p1_1020-1836/`.
- **A resubmission at 21:36 without `PASS=2`** (11272919–11272924) re-ran pass 1's chain: both GPU jobs found 0 tasks left, and it wrote a second report of the same pass, `report_12b_p1_1020-2137/`.
- **Pass 2** (`PASS=2 STEPS="modelzip ctrl_12b report"`): modelzip 11273521 (check only), zip check 11273523, the arm 11273524 on d1028 from 22:22 on 2026-10-20 to 23:46 (1 h 24), all 109 tasks into `loops_ctrl_12b_v31_a100_r2`. The chained 11273525 found nothing left. Report 11273526 → `report_12b_p2_1020-2212/`, which reads both passes.
- **Provenance:** the main track's `bd82ee8` and the paper track's `ff614e5`, with G3's harness (`067143a2f3ec`), vLLM build (`73c69bafca24`), wheelhouse v31 and sampling. The report's provenance differs from `ctrl` only in the zip and model columns (`12b`).
- **The checks written before the run pass.** Every pass has all 109 tasks. Call 1's prompt ends with the empty thought 100% of the time, the template's prefill with thinking off.
- **Compute:** about 2.8 A100-hours (1 h 23 and 1 h 24, no-op jobs about 3 min), plus a few CPU minutes. The estimate was 2–3.
- **Task endings per run:**

| Run | Resolved | Ended normally | 60-call cap | Timeouts | Context window |
|---|---|---|---|---|---|
| `ctrl_12b` pass 1 / 2 | 14 / 10 | 31 / 38 | 50 / 36 | 26 / 34 | 2 / 1 |
| `ctrl` pass 1 / 2 (G3) | 35 / 37 | 94 / 92 | 0 / 1 | 11 / 12 | 4 / 4 |

Every resolved task ended normally, in both sizes. The 12B makes 42.9 main-agent calls per task against 36.6, and its tasks take 148 and 166 s on average against 129 and 140 s.

## 3. Arms, passes pooled (from report_12b_p2's loops.md)

All 109 tasks. Copies repeat the call just before; returns repeat an earlier one. Loop tasks are task runs where at least half the calls are bad (stale or malformed). "After a run of 3+" counts runs that end before the task does.

| Arm | Runs | Task runs | Resolved | Stale: copies / returns | Malformed | Lock-in k=2 / 3+ | Runs 3+ / 5+ per task | Run length (3+) | Loop tasks | After a run of 3+: re-entered / next works / source edit after (runs) |
|---|---|---|---|---|---|---|---|---|---|---|
| ctrl | 2 | 218 | 33.0% | 38.6%: 28.0% / 10.6% | 5.5% | 69.0% / 94.8% | 0.53 / 0.42 | 19.81 | 24.8% | 21.5% / 60.7% / 37.4% (107) |
| ctrl_12b | 2 | 218 | 11.0% | 53.4%: 39.5% / 13.8% | 0.2% | 62.7% / 95.6% | 0.88 / 0.71 | 19.73 | 44.5% | 40.5% / 50.7% / 23.0% (148) |

By task group:

| Arm | Dev (70): resolved per run | Dev: stale / loop tasks | Rich (39): resolved per run | Rich: stale / loop tasks |
|---|---|---|---|---|
| ctrl | 24, 25 | 32.7% / 17.9% | 11, 12 | 48.2% / 37.2% |
| ctrl_12b | 10, 7 | 43.9% / 31.4% | 4, 3 | 68.1% / 67.9% |

- **`baseline_dev.md` against `ctrl`:** −16.0 tasks (−22.5 to −10.0), 1 task better and 22 worse (sign test p < 0.001). The 12B solves 5 dev tasks in both runs and 12 in either (the 31B: 21 and 28). **`baseline_rich.md`:** −8.0 (−13.0 to −3.0), 0 better and 8 worse (p = 0.008).
- **Behaviour (dev, rich):**
  - median tool calls per task 46 and 54, against 36 and 48;
  - output tokens per task 4,300 and 3,823, against 2,959 and 3,164;
  - no source edit in 77 of 140 and 58 of 78 task runs, against 42 and 25. When it does edit, its first source edit comes earlier (median 29 s and 20 s, against 37 s and 79 s).
- **Per run:**
  - Copies start at 3.0–3.7% of calls, against 1.9–2.3%.
  - A bad call follows a bad one as often as in the 31B (dev 82–84%, rich 91–93%). After a clean call the next is bad more often: dev 13–15% against 9%, rich 19–20% against 11%.
- **Runs of 3+ by type:** search 74, command 58, read 34, edit 26, malformed 0. `ctrl`'s two passes: 39, 50, 1, 0, 26.
- **How runs of 3+ end** (summed over dev and rich):

| Run | Task end | Budget warning | Compaction | Own |
|---|---|---|---|---|
| ctrl_12b pass 1 / 2 | 21 / 23 | 29 / 34 | 33 / 27 | 14 / 11 |
| ctrl pass 1 / 2 | 4 / 5 | 28 / 30 | 18 / 15 | 9 / 7 |

## 4. Paired differences (from report_12b_p2's loops.md)

`ctrl_12b` − `ctrl` on the 109 tasks, with 95% intervals from 2,000 resamples of tasks. Shares are in percentage points. The pass-1 column is report_12b_p1's, which held one 12B pass.

| Metric | ctrl_12b − ctrl | ctrl_12b − ctrl, pass 1 only |
|---|---|---|
| resolved | **−22.0 (−29.8 to −15.1)** | **−20.2 (−28.0 to −13.3)** |
| stale | **+14.8 (+8.9 to +20.7)** | **+13.6 (+7.2 to +20.3)** |
| copies | **+11.5 (+4.7 to +18.3)** | **+11.3 (+3.0 to +19.5)** |
| returns | **+3.3 (+0.1 to +6.3)** | +2.3 (−1.4 to +6.2) |
| malformed | **−5.3 (−9.2 to −2.0)** | **−5.2 (−9.3 to −1.9)** |
| lock-in k=2 | −6.3 (−21.6 to +9.1) | −3.5 (−20.2 to +12.4) |
| lock-in k=3+ | +0.8 (−0.4 to +2.0) | +0.7 (−0.8 to +2.2) |
| runs 3+ per task | **+0.35 (+0.15 to +0.54)** | **+0.36 (+0.14 to +0.57)** |
| runs 5+ per task | **+0.29 (+0.13 to +0.45)** | **+0.28 (+0.10 to +0.47)** |
| run length (3+) | −0.08 (−3.92 to +3.60) | −0.08 (−4.66 to +4.50) |
| loop tasks | **+19.7 (+11.5 to +27.5)** | **+19.3 (+9.2 to +28.9)** |
| re-entered | **+19.0 (+7.7 to +30.2)** | **+15.3 (+0.6 to +29.9)** |
| next call works | −10.1 (−21.8 to +1.6) | −9.4 (−24.7 to +6.6) |
| source edit after | −14.4 (−29.9 to +0.3) | −12.4 (−29.2 to +4.1) |
| tasks with a source edit | **−31.2 (−39.9 to −23.4)** | **−29.8 (−39.0 to −20.6)** |

Bold: the interval excludes 0. How often and how much the 12B loops differs; how firmly a run, once under way, holds (lock-in at 3+, run length) doesn't.

**Pass to pass** (run 2 − run 1), the noise floor:

| Arm | Resolved | Stale | Copies | Returns | Lock-in k=3+ | Runs 3+ per task |
|---|---|---|---|---|---|---|
| ctrl | +1.8 (−2.8 to +7.3) | −0.5 (−7.1 to +6.5) | −1.7 (−9.1 to +6.4) | +1.2 (−2.5 to +5.0) | +0.2 (−1.3 to +1.7) | −0.02 (−0.19 to +0.16) |
| ctrl_12b | −3.7 (−9.2 to +0.9) | +2.3 (−3.9 to +8.3) | +0.3 (−8.0 to +7.8) | +2.0 (−2.0 to +6.0) | +0.1 (−1.2 to +1.5) | −0.02 (−0.25 to +0.21) |

The two 12B passes agree: no interval excludes 0. Every difference between the sizes in bold above is several times larger.

## 5. Local splits

### 5.1 The lock-in curve at two sizes

P(the next call is identical | the last k calls were identical, the last one well formed). Clean steps: no budget warning in the last result, no compaction before the next call.

| Steps | k = 1 | k = 2 | k = 3–4 | k = 5–9 | k = 10+ |
|---|---|---|---|---|---|
| ctrl, 2 passes, clean | 0.024 (4,985) | 0.678 (115) | 0.958 (142) | 1.000 (290) | 1.000 (1,052) |
| ctrl_12b pass 1, clean | 0.055 (2,275) | 0.731 (119) | 0.943 (159) | 0.990 (294) | 0.998 (907) |
| ctrl_12b pass 2, clean | 0.061 (2,212) | 0.645 (124) | 0.958 (143) | 0.989 (278) | 0.997 (898) |
| **ctrl_12b, both passes, clean** | **0.058 (4,487)** | **0.687 (243)** | **0.950 (302)** | **0.990 (572)** | **0.997 (1,805)** |
| ctrl_12b, both passes, all steps | 0.056 (5,453) | 0.630 (305) | 0.905 (367) | 0.965 (678) | 0.962 (2,315) |
| ctrl, 2 passes, all steps | 0.025 (5,451) | 0.672 (134) | 0.913 (172) | 0.956 (338) | 0.954 (1,237) |

- **Paired over tasks, all steps** (report_12b_p2), ctrl_12b − ctrl:
  - k = 1: **+3.2 points (+2.2 to +4.2)**;
  - k = 2: −4.2 (−22.0 to +13.1);
  - k = 3–4: −0.8 (−6.0 to +4.6);
  - k = 5–9: +0.9 (−2.0 to +3.9);
  - k = 10+: +0.8 (−0.3 to +2.0).
- **Only the start of the curve moves.** A copy follows a fresh call 2.4 times as often in the 12B. From the first copy on, the two curves agree within their noise.
- **Not quite absorbing at 12B:** 11 of its 2,377 clean steps at k ≥ 5 end the run (0.5%), against 0 of the 31B's 1,342.
- **Each pass's k = 2 rests on about 120 steps,** twice the 31B's, and the passes give 0.73 and 0.65. Quote the pooled value.

### 5.2 How runs end

Well-formed runs of 3+ (the 12B has no malformed ones), both passes of each arm. The run counts include runs that end with the task.

| Arm | Runs of 3–4: own exits | 5–9: own | 10+: own | 10+: budget warning / compaction / task end |
|---|---|---|---|---|
| ctrl, well formed | 6 of 16 (38%) | 0 of 16 | 0 of 58 | 43 / 14 / 1 |
| ctrl, all runs (ablation_G4.md §5.2) | 12 of 24 (50%) | 3 of 20 (15%) | 1 of 72 (1%) | 43 / 23 / 5 |
| ctrl_12b | 15 of 37 (41%) | 6 of 36 (17%) | 4 of 119 (3%) | 51 / 34 / 30 |

- **Own exits, over the runs that end before the task:** the 12B 25 of 148 (16.9%).
  - Against the 31B's well-formed runs, 6 of 87 (6.9%): +10.0 points (+1.9 to +18.1).
  - Against all its runs, 16 of 107 (15.0%): +1.9 (−7.5 to +11.7). Most of the 31B's own exits end malformed runs.
  - By pass: 14 of 76 (18.4%) and 11 of 72 (15.3%); pass 2 − pass 1 −3.1 (−13.5 to +6.7).
- **The budget warning stops fewer long runs.** Of the well-formed runs of 10+ that receive the warning ("Only 10 tool call(s) remaining …"):
  - the 31B ends 43 of 44 before the task does; the 12B ends 51 of 81;
  - the 12B's other 30 run on into the 60-call cap (20) or the timeout (10);
  - a warned run goes on for 5.5 copies after the first warning in the 12B, 3.2 in the 31B.
- **Per task, paired** (ctrl_12b − ctrl, two passes each):
  - runs of 10+: 0.55 against 0.33, **+0.22 (+0.10 to +0.33)**;
  - runs of 3–9: 0.33 against 0.20, **+0.13 (+0.01 to +0.26)**;
  - calls inside runs of 3+: 17.4 against 10.5, **+6.8 (+3.2 to +10.4)**;
  - calls per task: 42.9 against 36.6, **+6.3 (+1.8 to +10.8)**.
- **So the 12B's extra calls are its loops:** its +6.3 calls per task are about the +6.8 calls it spends inside runs.

### 5.3 Re-entry, by how the run ended

Runs of 3+ that end before the task; re-entered means the run's call is sent again, stale, later in the task.

| Arm | Own | Budget warning | Compaction | All |
|---|---|---|---|---|
| ctrl | 7 of 16 (44%), 29 calls left | 6 of 58 (10%), 4 left | 10 of 33 (30%), 18 left | 21.5% |
| ctrl_12b | 14 of 25 (56%), 28 calls left | 18 of 63 (29%), 5 left | 28 of 60 (47%), 23 left | 40.5% |

- **Unlike dedup's gain (ablation_G4.md §5.3), this isn't the mix of exits.** The 12B re-enters more after every kind: own +12, budget warning +19, compaction +17 points.
- After a compaction the 12B sends the run's call again about half the time. That is the exit that leaves the most calls.

### 5.4 What the 12B loops on

- **Failing edits:** 26 runs of 3+ identical `edit_file` calls, 25 of which return `FileEditError` ("old_string not found") at every call.
  - Their `old_string` carries the arguments' JSON escaping (`\\n`, `\\"`) inside the string, so it can't match the file.
  - In `fastapi_14605` (pass 1), the same failing edit runs 22 times, and later 6 more times into the budget warning.
- **Truncated reads:** 34 runs of identical `read_file` calls, 24 of them on a result marked `is_truncated`. The 12B re-sends the read from the same `start_line` instead of paging on.
- **Edits overall:**

| Both passes | ctrl | ctrl_12b |
|---|---|---|
| Task runs that try an edit | 168 of 218 (77%) | 98 of 218 (45%) |
| Edit calls | 489 | 804 |
| … failing with `FileEditError` | 46 (9%) | 586 (73%) |
| … stale repeats | 4 | 436 |
| Task runs with a source edit that applied | 151 (69%) | 83 (38%) |
| Task runs that tried and never applied one | 2 | 15 |

- **The broken call that loops differs by size.** The 31B loops on calls whose arguments don't fit the tool (26 malformed runs). The 12B writes well-formed calls that can't succeed. In both, the error comes back at every call and the copy goes on. This fits G2's finding that the results' content isn't the cue ([ablation_G2.md](ablation_G2.md)), but here it is observed, not manipulated.

### 5.5 Which tasks loop

| Pair | Loop tasks (each) | In both | Chance | Chance within split | Jaccard |
|---|---|---|---|---|---|
| ctrl pass 1, pass 2 | 27, 27 | 15 | 6.7 | – | 0.38 |
| ctrl_12b pass 1, pass 2 | 48, 49 | 28 | 21.6 | – | 0.41 |
| ctrl and ctrl_12b (4 pairs) | 27 and 48–49 | 16–19 | 11.9–12.1 | 13.4–14.2 | 0.27–0.34 |

- **"Chance within split"** keeps each run's dev and rich loop counts, since both sizes loop more on rich: the 12B on 26–27 of the 39 rich tasks, the 31B on 14–15.
- **A permutation test** shuffles the 12B's per-task loop counts within split, 10,000 times. The sum over tasks of (31B loop passes × 12B loop passes) is 67, reached by chance with p = 0.015.
- **Tasks by (31B loop passes, 12B loop passes):** (0, 0) 31; (0, 1) 26; (0, 2) 13; (1, 0) 8; (1, 1) 9; (1, 2) 7; (2, 0) 1; (2, 1) 6; (2, 2) 8.
- **The 12B's loop tasks contain most of the 31B's:** 14 of the 15 tasks that loop in both 31B passes loop in at least one 12B pass. But 13 tasks loop in both 12B passes and in neither 31B pass.
- **So loop-proneness carries across sizes only in part.** Within a size, the overlap is as strong as the 31B's (Jaccard 0.41 against 0.38).

### 5.6 The model's own thoughts

Clean steps after a well-formed call, from the proxy's `calls.jsonl` (the 31B's control has it for pass 2 only).

| Run | Opens a thought at k = 1 | At k ≥ 2 | Continues the run, with / without a thought | Thoughts with text |
|---|---|---|---|---|
| ctrl pass 2 | 43.8% (2,529) | 6.3% (790) | 78.0% (50) / 98.6% (740) | 0 |
| ctrl_12b pass 1 | 85.7% (2,275) | 27.2% (1,479) | 88.6% (403) / 100.0% (1,076) | 0 |
| ctrl_12b pass 2 | 87.2% (2,188) | 32.3% (1,443) | 88.2% (466) / 99.9% (977) | 0 |

- **Every opened thought is empty,** `<|channel>thought\n<channel|>` and then the call, as for the 31B (ablation_G4.md §5.5).
- **The 12B writes it about twice as often.** Inside runs it writes it less often than after a fresh call, as the 31B does.
- **Steps with it continue less often** in both sizes (88% against about 100%). This is an association, not a lever: under dedup the 31B wrote it at most steps and went on copying (G4).

### 5.7 The analyzer sub-agent

- **The pinned agent gives the main agent a `code_analyzer_agent` tool,** a sub-agent with its own tools.
  - The 31B never calls it in a closed-loop run: 0 calls in the traces of `ctrl`'s two passes, `ctrl_v30` and `dedup`'s pass 1.
  - The 12B calls it 68 times in 50 tasks (pass 1) and 75 times in 54 tasks (pass 2). Its sub-agent makes 1,434 and 1,558 requests.
- **Many invocations fail.** 26 of 68 and 30 of 75 end on `ContextWindowExceededError`: the sub-agent's prompt grows past 16,385 tokens, the 32,768-token context less 16,384 for the output.
- **The sub-agent loops harder than the main agent.** Labelled by the main agent's rules:
  - 70% and 65% of its 1,324 and 1,441 calls are stale, 58% and 54% copies;
  - its longest runs are 241 and 229 identical calls.
- **Its calls count toward the 60-call cap.** In 4 tasks per pass the main agent made one call, to the analyzer, and the sub-agent used the whole budget. Of the 12B's 86 tasks at the cap, 49 had an analyzer call. In the other 37, the main agent made all 60 calls itself.
- **The late empty-thought prompts come from it, not from the template.** After each analyzer result the harness adds a user turn ("Please continue your work using the available tools, or call submit_patch when you have completed and verified your changes."). The template then ends the next prompt with the empty thought.
  - That accounts for all 47 of pass 1's prompts after call 1 that end with the empty thought, and 58 of pass 2's 60: the report's 4.6–7.6% at calls 2–5. `ctrl` pass 2 has 1 such prompt in 4,053.
- **The loop metrics count only the main agent's calls,** so the sub-agent's calls and requests are outside every table above. Its time and its share of the call budget are not.
- **Task runs with an analyzer call** resolve 8 of 104, those without 16 of 114; loop tasks 41 of 104 and 56 of 114. This is observed, not controlled: the tasks that send the 12B to the analyzer may differ.

### 5.8 Resolution

| | ctrl, 2 passes | ctrl_12b, 2 passes |
|---|---|---|
| Resolved | 72 of 218 | 24 of 218 |
| Loop task runs: resolved | 11 of 54 (20%) | 8 of 97 (8%) |
| Other task runs: resolved | 61 of 164 (37%) | 16 of 121 (13%) |
| Other − loop | **+16.8 (+1.3 to +30.4)** | +5.0 (−3.0 to +13.3) |

- **Solves per task** (31B, 12B; out of 2 each): 68 tasks 0 and 0; 19 tasks 2 and 0; 8 tasks 2 and 2; 6 (1, 0); 5 (2, 1); 2 (1, 1); 1 (0, 1).
- **The 12B solves 16 tasks at least once, and 15 of them are also the 31B's.** It isn't solving different tasks; it solves fewer of the same ones.
- **The size gap is mostly not loops.** At its non-loop rate, the 12B's 97 loop task runs would resolve about 13 instead of 8: about 5 of the 48 tasks it trails by. This is a rough bound in the loops' favour: a loop is no cause, and un-looping solved nothing at 31B (G3, G4).
- **Where the rest goes, in the records:**
  - fewer task runs try an edit, and most of the 12B's edits fail (§5.4);
  - 86 task runs end at the call cap and 60 at the timeout, and none of those resolve (§2);
  - the analyzer spends calls and time (§5.7).
- **Floor effect:** with 13% of non-loop runs resolved, a loop can't mark a lost run as clearly as at 31B.

## 6. The predictions, against the result

Written in README.md §3 before the run (`ctrl_12b` against `ctrl`, passes pooled, all 109 tasks, well-formed calls for the hazards).

| Prediction | Result | Verdict |
|---|---|---|
| Checks: provenance `ctrl`'s except the zip and the model; all 109 tasks per pass; call 1's prompt ends with the empty thought 100% | All three (§2) | Holds |
| Resolution falls below `ctrl`'s 72 of 218 | 24 of 218, −22.0 (−29.8 to −15.1) | Holds |
| Loops at least as common: stale ≥ 38.6%, loop tasks ≥ 24.8% | 53.4% and 44.5% | Holds |
| The hazard rises with k; on clean steps ≥ 0.95 at k = 5–9 and at 10+ | 0.058 → 0.687 → 0.950 → 0.990 → 0.997 | Holds |
| At most 10% of well-formed runs of 10+ end on their own (`ctrl`: 1 of 72) | 4 of 119 (3%); the 31B's well-formed: 0 of 58 (its 1 of 72 counts malformed runs) | Holds |
| Loop task runs resolve less often than other task runs | 8% against 13%: +5.0 (−3.0 to +13.3) | Holds in direction; the interval includes 0 |
| The 12B's loop tasks overlap the 31B's more than chance | 16–19 shared against 13.4–14.2 within split; p = 0.015 | Holds, modestly |
| No prediction: malformed calls | 0.2% against 5.5% | – |
| No prediction: the empty thought the model writes itself | 86–87% at k = 1 against 44%; always empty | – |
| No prediction: timeouts and the budget warning | timeouts 60 against 23; the 60-call cap 86 against 1 | – |

By the reading written before the run, the shape replicates: absorbing runs, exits from outside, and loops in lost runs (in direction). So the descriptive claims can be stated for two sizes.

## 7. What it shows

**1. The lock-in is Gemma 4's, not the 31B's.** Once a copy has been sent, the 12B continues at the 31B's rates, within a point or two at every k (§5.1). Its long runs are as absorbing: 4 of 119 runs of 10+ end on their own, and 11 of 2,377 clean steps at k ≥ 5 (§5.2).

**2. Size changes the onset, not the lock-in.**
- The 12B copies 2.4 times as often after a fresh call, and after a clean call its next call is bad 1.5–1.8 times as often.
- It then has more runs and longer stretches inside them: +0.22 runs of 10+ and +6.8 calls in runs per task.
- With G4's reading (the curve's height is selection of states where the model is near certain of the call): the 12B reaches such states more often and is as certain in them. That is an interpretation; selection wasn't measured at 12B.

**3. Broken calls loop in both sizes, whatever error comes back.** The 31B loops on malformed calls, the 12B on edits that can't apply and on truncated reads. The calls differ with the model's errors; the copy goes on through the error in both (§5.4).

**4. The outside exits are weaker at 12B.** The budget warning ends 63% of its warned long runs (the 31B: 98%), and the 12B returns to a call it left more often after every exit (§5.2–5.3). The levers that end the 31B's loops from outside may not transfer at their 31B sizes.

**5. "Loops mark lost runs" is weaker at the smaller size, and loops don't explain the size gap.** The 12B loses most runs with or without a loop. At most about a tenth of its gap would close if its loop runs resolved like its other runs (§5.8).

**6. Which tasks loop carries over in part.** The 12B loops on almost every task the 31B loops on, and on many more. Within each size, two passes agree as much as the 31B's two passes do (§5.5).

**7. The empty thought is still empty.** The 12B writes it about twice as often, and it carries no text (§5.6).

**8. The same bytes are not the same agent.** The zip differs in two `model:` lines, but the 12B uses a tool the 31B never touches. Its sub-agent loops harder than the main agent and spends the shared call budget (§5.7). A size control changes what the agent does, not only how well.

## 8. What it doesn't show

- *The mechanism claims at 12B.* Selection (G4), the copies' reinforcement (G1), the pattern against the content (G2) and the forced thought (T-open) were measured on the 31B only. No 12B replay states or dedup arm exist.
- *Other models.* Two sizes of one family (int4, w4a16), thinking off, one harness, four repositories and 109 tasks. No E4B, E2B or other family.
- *Thinking on against off at 12B.* The main track's thinking-on `budgetfit_12b` (15 and 18 of 70 dev, against the 31B's 31 and 30) used another zip and budget, so its ratio to this run's isn't a measured effect of thinking.
- *What the analyzer does to the 12B.* Its effects are observed within one arm; a 12B run without the tool would be another agent.
- *Small effects.* Two passes of 109 tasks; at 12B the k = 2 hazard rests on about 120 steps per pass.
- *A submission.* The 12B solves a third as many tasks; this is a generality check, not an entry.

## 9. What it changes

- **The generality limit narrows.** Findings 1 (loops are common), 4 (long runs end from outside) and 8 (which tasks loop), and the lock-in curve's shape, now hold for two Gemma 4 sizes. Finding 1's "mark lost runs" holds at 12B in direction only. Findings 2, 3, 5, 6 and 7 stay the 31B's.
- **Not yet synced** (this record only):
  - [../findings.md](../findings.md) §5 "Generality", §7 "Open" (the 12B control is done), and the size notes in §2.1, §2.4 and §2.8;
  - [README.md](../loops/README.md) §3, "The 12B control": its status and result, and the line under §3's head that says it hasn't run;
  - the repo's root README index, which doesn't list this record yet.
- **No new arm follows from it.** The open questions it raises (selection at 12B, the analyzer's effect, the budget warning's weaker pull) would each need a design of their own, after the freeze.
- **For the paper:**
  - Show the two lock-in curves side by side (§5.1). The point is onset against lock-in.
  - Report own exits by run length for both sizes, with the 31B's well-formed and all-run counts (§5.2).
  - Note the 12B's loop calls (failing edits, truncated reads) beside the 31B's malformed ones, and the analyzer as a caveat of the size comparison.
