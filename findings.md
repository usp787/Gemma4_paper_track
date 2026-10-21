# findings.md: what the loop study found

<!-- written 2026-10-20, when the experiments were frozen after G4 (the user's call). It consolidates the records and
     does not replace them: every number below is quoted from the record named next to it, where its design, run,
     intervals and caveats are. The records are ablation/ablation_G1.md ... ablation_G4.md, loops/README.md (round 1
     and the round-2 plan), loops/loops_reason_10_13.md (the mechanisms and the review), gate1/README.md (Gate 1 and
     the replay) and loops/literature.md (prior work, with checked references). Intervals are 95%, from resampling
     tasks. The legacy explainers, loops/loop_illustration_legacy.html and ablation/ablation_recap_10_17_legacy.html,
     predate G3's record and G4 and keep their older wording. Updated 2026-10-21: the 12B control
     (ablation/ablation_control.md), added after the freeze as a generality check: finding 10, §2.10, and size notes
     in §1 and §2.1, §2.4, §2.8, §5–§8. With it every arm has run. -->

**The question:** why does Gemma 4 31B, with thinking off, send the same tool call again and again on a coding task, and what breaks the loop?

**The setting of every number below, unless it says otherwise:**
- Gemma 4 31B (int4, w4a16), thinking off, through the competition's harness (`adk_submission`) and the `off_args` agent's pinned zip;
- T = 0.2, top-p 0.95, top-k 64; 4 minutes, 60 tool calls and 100 turns per task;
- 109 tasks from four repositories (70 dev-sound: fastapi, requests, httpx; 39 rich-sound: rich);
- the closed-loop runs on wheelhouse v31, one A100, 4 tasks at once;
- the one-step replays score recorded v30 histories;
- the 12B control's numbers (§2.10 and the size notes) say so: the same setting with Gemma 4 12B (int4, w4a16) in the 31B's place.

## 1. The findings

1. **Loops are common, and they mark lost runs.** A quarter of task runs are loop tasks, with half or more of their calls stale or malformed. Loop task runs resolve 20% of the time, against 37% for the rest. At 12B loops are commoner still (44.5% of task runs), and they mark lost runs only in direction (8% against 13%).
2. **Loops start where the model is already close to certain of the call.** With the request held the same at every step of a run, the agent sends the call again 78% of the time after its first copy and 99% by the 10th. So most of the lock-in curve's height is selection.
3. **The copies in view reinforce the loop on their exact tokens, and close its last escapes.** At fixed histories, identical copies raise the next copy (0.002 → 0.43 for 1 → 4 copies) and re-spelled ones don't. In the closed loop, the copies take the hazard from 87–93% to 96–100% at 3–9 identical calls.
4. **At T = 0.2, long runs end only from outside.** After 5 identical calls, the control never stopped on a clean step (1,342 of 1,342). Its long runs ended at the budget warning, a context compaction, or the task's end. The 12B's do too: 4 of its 119 runs of 10+ end on their own.
5. **The pattern of the context drives the copy, not what the results say.** Varying IDs or adding a faithful "zero matches" note leaves the copy where it was. An instruction moves it, and so does breaking the pattern before the call.
6. **The model skips thinking where a thought would help.** Where it chose not to think, a forced thought escapes 58% of the time, against 11%, one step ahead.
7. **Every whole-task lever changes the loops, not the outcome.** T = 1 halves how many loops start; the always-block template fragments them; dedup lets short runs end. Resolution stays at 71–73 of 218 (38 of 109 for the block's single pass).
8. **At T = 0.2, which tasks loop is a property of the task,** with or without the copies in view. At T = 1 it isn't. Across sizes it carries over only in part.
9. **The self-teacher shares the loop,** which is why the study left OPSD: Gate 1's thinking teacher put less KL on the failure tokens than on routine ones (ratio 0.51).
10. **Size changes how often loops start, not how firmly they hold.** Once a copy has been sent, Gemma 4 12B continues at the 31B's rates. It starts copying 2.4 times as often after a fresh call, so it loops more and resolves less; loops explain little of that gap.

In one sentence: the loop is a symptom of a state the model is already sure about, and it is held in place by its own copies; removing the loop doesn't recover the task.

## 2. The evidence, claim by claim

Status: *measured* means a controlled manipulation; *observed* means a pattern in recorded runs; *interpretation* means a reading of both.

### 2.1 Loops are common, and they mark lost runs (observed)

- **How common** (the v31 control, two passes): 38.6% of calls are stale, 28.0% are copies of the call just before, and 24.8% of task runs are loop tasks ([ablation_G3.md](ablation/ablation_G3.md) §3).
- **Bad calls follow bad calls:** after a stale or malformed call the next is bad 83% of the time, against 9% after a clean one (dev).
- **Most loops are searches and commands:** of the control's runs of 3+, 39 are read-only searches, 50 other commands and 26 malformed (§3 there).
- **They mark lost runs:**
  - loop task runs resolve 20%, against 37% (§5.3 there);
  - within the 20 sometimes-solved tasks that have both kinds of run, the looping runs resolve 22.8 points less often (−41.1 to −4.3, §5.7 there);
  - in 23 of 24 un-looped tasks, the loop began before the first source edit (§5.3 there).
- **At 12B** ([ablation_control.md](ablation/ablation_control.md) §3, §5.8):
  - loops are commoner: 53.4% of calls are stale, 39.5% copies, and 44.5% of task runs are loop tasks;
  - the loops are on other calls: no malformed runs, but failing edits and truncated reads (search 74, command 58, read 34, edit 26);
  - they mark lost runs only in direction: 8% against 13% (+5.0, −3.0 to +13.3). The 12B resolves only 13% of the runs where it doesn't loop, so the contrast has little room.
- **Caveat:** this is a correlation. A run that is already lost may loop for that reason. §2.7 shows that removing loops doesn't recover runs.

### 2.2 Loops start where the model is already close to certain (measured, closed loop)

- **The test** (G4, [ablation_G4.md](ablation/ablation_G4.md) §5.1): a proxy collapses every run of identical tool turns to its first, so at every step of a run the model gets the request it got after the run's first call.
  - Each repeated request is a fresh draw from the same prompt. A state's chance to continue is then the same at every step, and the curve can rise with k only through which states keep looping.

| P(one more identical call), clean steps | k = 2 | k = 3–4 | k = 5–9 | k = 10+ |
|---|---|---|---|---|
| request held fixed (dedup, 2 passes) | 0.78 (104) | 0.87 (145) | 0.93 (257) | 0.993 (811) |
| copies in view (control, 2 passes) | 0.68 (115) | 0.96 (142) | 1.00 (290) | 1.00 (1,052) |

- **One step ahead, G1 and G2 pointed the same way.** Four forced copies give p(c) 0.43, while the closed loop continues 96% (ablation_G1.md). G2's own number for "the state where a run began" was wrong (§4 below).
- **Caveats:**
  - The fixed request includes the whole history so far, so "the state" and "the task" aren't separated (§2.8).
  - k = 2 rests on about 55 steps in about 40 tasks per pass; the two passes give 0.83 and 0.72.

### 2.3 The copies reinforce, on exact tokens, and close the last escapes (measured)

- **At fixed histories** (G1, [ablation_G1.md](ablation/ablation_G1.md)): after an empty search c, adding identical copies of c raises p(c) at the run's sampling.
  - 1, 2 and 4 copies give 0.002, 0.098 and 0.434; exact p(c) 0.006, 0.052 and 0.307.
  - The same search re-spelled (quotes, flag order, long flags) stays near 0.03: dose4 − para4 is +0.376 (0.258 to 0.496).
  - 59 of 60 histories rise monotonically.
- **In the closed loop** (G4 §5.1–5.2):
  - At k = 2 the copies add nothing measurable; the paired difference there is +7.6 points (−8.9 to +22.4) in dedup's favour.
  - From k = 3 they close the escapes: −7.0 (−13.3 to −0.8) at 3–4 and −4.1 (−7.3 to −1.1) at 5–9 for dedup.
  - Without the copies in view, the model ends 47% of its runs of 3+ itself, against 15% (+32, +20 to +45), almost all runs of 3–9 calls.
  - Runs of 10+ per task fall from 0.33 to 0.25 (−0.08, −0.16 to −0.01). About three quarters of the long runs persist with no copies in view.
- **Reading:** there are two kinds of long run. Most are near fixed points of the policy, and the rest are built by the copies.

### 2.4 At T = 0.2, long runs end only from outside (observed)

- On clean steps (no budget warning in the last result, no compaction before the next call), the control continued every run of 5+: 290 of 290 steps at 5–9 and 1,052 of 1,052 at 10+ (ablation_G3.md §5.1).
- The 58 well-formed runs of 10+ ended after the budget warning ("Only 10 tool call(s) remaining …") 43 times, at a compaction 14 times, at the task's end once, and on a clean step never (§5.2 there).
- **The exits barely help.** After the budget warning at most 10 calls are left. After a compaction, the agent sends the same call again 30–41% of the time (§5.6 there).
- **Under dedup** the long runs end the same way: 31 of 54 at the budget warning, 15 with the task (ablation_G4.md §5.2).
- **At 12B** too (ablation_control.md §5.1–5.2):
  - 4 of its 119 runs of 10+ end on their own, and 11 of its 2,377 clean steps at k ≥ 5 end a run (0.5%, against 0 of 1,342);
  - the budget warning stops fewer of them: 51 of 81 warned long runs end before the task, against 43 of 44. The other 30 run into the 60-call cap or the timeout.

### 2.5 The pattern of the context drives the copy, not the results' content (measured, one step ahead)

- **G2** ([ablation_G2.md](ablation/ablation_G2.md)), 101 repeat states in 23 tasks, a 2 × 2 on the repeated results:
  - varying IDs changes exact p(c) by −0.005 (−0.009 to −0.002);
  - a faithful note ("The search completed and found zero matches …") by +0.020 (+0.005 to +0.036), slightly *more* copying;
  - the added tokens alone by −0.004.
- **Instructions do move it.** Round 1's note and hint ("Do not run it again; …") took p(copy) at 3rd-or-later copies from 0.94 to 0.49 and 0.45 (gate1.md §2). They were run at other states, so description against instruction is a contrast between two designs, not a measured interaction.
- **Breaking the pattern moves it.** At copy states, ending the context with the empty thought (`block`) takes exact p(c) from 0.791 to 0.216. With the empty thought before every earlier call too (`think_all`), it is back at 0.818. So the block works as a pattern break, not as a switch of mode.
- **The empty thought the model writes itself is no lever.** With the copies in view it stops writing it (6% of steps inside runs). With them collapsed it writes it at 60–84% of steps, always empty, and goes on copying (96.6% continue; ablation_G4.md §5.5).
- **Caveats:** one wording per note; 86 of the 101 states are fastapi.

### 2.6 The model skips thinking where a thought would help (measured, one step ahead)

- Free to think (T-think, thinking on), the model wrote a thought at 72% of 2nd copies and 23% of 3rd-or-later copies (gate1/README.md).
- **T-open** forces the thought open for 32 tokens. On 213 looping states in 44 tasks ([loops_reason_10_13.md](loops/loops_reason_10_13.md) §5):

| Looping states | T-think escapes | T-open escapes | Paired difference |
|---|---|---|---|
| T-think skipped the thought (114, 36 tasks) | 10.5% | 57.9% | +47.4 points (33.0 to 61.1) |
| T-think thought (99, 34 tasks) | 61.6% | 62.6% | +1.0 (−6.7 to +9.1) |

- **"A stuck agent is a confident agent"** is the reading. The closed loop now supports it: from the same request, re-drawn, the agent repeats the call 78–99% of the time (§2.2). It is the probability of a fresh draw, not calibrated confidence.
- **Caveats:** an escape is one step ahead, a call that is neither stale nor malformed, not a useful one. The forced thought was never run over whole tasks (§5).

### 2.7 Whole-task levers change the loops, not the outcome (measured, closed loop)

| Lever (arm) | What it does to loops | Resolved | Record |
|---|---|---|---|
| none (`ctrl`, 2 passes) | stale 38.6%, loop tasks 24.8%, run length 19.8 | 72 of 218 | G3 |
| T = 1 (`ctrl_t1`, 2 passes) | stale −18.2 points (−24.8 to −11.0); loop tasks 24.8% → 9.2%; once under way a run still continues 87% at 3+ (0.95 at T = 0.2) | 71 of 218 | G3 |
| always-block template (`block`, 1 pass) | runs 9 calls shorter, more of them (+0.35 per task), re-entry +26 points; stale +1.6 (−4.5 to +7.6) | 38 of 109 (ctrl: 35, 37) | G3 |
| dedup proxy (`dedup`, 2 passes) | more own exits of short runs (+32 points); copies −2.6 (−8.2 to +2.8), stale −1.3, loop tasks +1.4 | 73 of 218 | G4 |

- **Un-looping doesn't solve tasks.**
  - T = 1 un-looped 46 of the control's loop task runs: the control solved 10 of those tasks, T = 1 solved 9. The freed budget went into new calls and source edits (ablation_G3.md §5.3).
  - Dedup un-looped 6 task runs of tasks that loop in both control passes, and solved 2; the control solves those tasks 4 times in 12 runs (ablation_G4.md §5.6).
- **A rescue ceiling from recorded runs:** 6 of 23 loop failures were solved by at least one of 9 other runs, about 2 of 70 tasks (loops/README.md). That is an observed cross-run estimate, not an upper bound.
- **Caveat:** two passes of 109 tasks can't detect a change of 1–2 tasks; one run varies by about ±2.2 tasks.

### 2.8 Which tasks loop is a property of the task (observed)

- At T = 0.2, two control passes share 15 of their 27 loop tasks, against about 7 by chance. A dedup pass and a control pass share as many (Jaccard 0.39–0.49 against 0.38; ablation_G4.md §5.4).
- They rarely loop on the same call (2 of 15), and two passes diverge after a median of 3 calls (ablation_G3.md §5.4).
- At T = 1 the two passes share no loop task.
- Loops are as common in sometimes-solved tasks as in never-solved ones (24–26% of task runs), and rare in tasks every run solves (6.5%; ablation_G3.md §5.7).
- **Across sizes, only in part** (ablation_control.md §5.5):
  - a 31B pass and a 12B pass share 16–19 loop tasks, against 13.4–14.2 by chance within split (p = 0.015);
  - 14 of the 15 tasks that loop in both 31B passes loop in at least one 12B pass;
  - 13 tasks loop in both 12B passes and in neither 31B pass.
  - Within the 12B, two passes agree as much as the 31B's do (Jaccard 0.41 against 0.38).

### 2.9 The OPSD link: the self-teacher shares the loop (measured, Gate 1)

- Gate 1 scored recorded thinking-off trajectories under candidate teachers ([gate1/README.md](gate1/README.md)).
  - The thinking teacher (T-think) failed the gate: ratio 0.51 (0.28 to 0.96) of KL on failure tokens against routine ones.
  - The forced-empty-thought teacher (C-none) passed (2.82, 1.08 to 9.41). Its effect came from the forced empty thought, not from the thinking-on template.
- At 3rd-or-later copies, T-think's copy probability is 0.80: a self-teacher that thinks only when it chooses shares the student's loop.
- **Why no training was run:** the trainer had cross-entropy only, RFT round 0 went from 23 to 8 of 70, and G4, the gate for exit distillation, gained no tasks ([opsd_assessment.md](opsd_assessment.md), loops/README.md §3).

### 2.10 Size changes the onset, not the lock-in (observed, closed loop, two sizes)

- **The test** ([ablation_control.md](ablation/ablation_control.md)): two passes of `ctrl` with only the zip's two `model:` lines changed to Gemma 4 12B, against the 31B's two passes. Same harness, vLLM build, sampling, budget and 109 tasks.

| P(one more identical call), clean steps | k = 1 | k = 2 | k = 3–4 | k = 5–9 | k = 10+ |
|---|---|---|---|---|---|
| Gemma 4 31B (`ctrl`, 2 passes) | 0.024 (4,985) | 0.678 (115) | 0.958 (142) | 1.000 (290) | 1.000 (1,052) |
| Gemma 4 12B (`ctrl_12b`, 2 passes) | 0.058 (4,487) | 0.687 (243) | 0.950 (302) | 0.990 (572) | 0.997 (1,805) |

- **Paired, all steps:** +3.2 points at k = 1 (+2.2 to +4.2). From k = 2 on, no difference's interval excludes 0 (k = 2: −4.2, −22.0 to +13.1).
- **So the 12B starts more runs and holds them as firmly:**
  - runs of 3+ per task 0.88 against 0.53, and of 10+ 0.55 against 0.33;
  - calls inside runs of 3+, +6.8 per task (+3.2 to +10.4), about all of its +6.3 extra calls;
  - run length (19.7 against 19.8) and lock-in at 3+ (95.6% against 94.8%) don't differ.
- **The broken call that loops differs by size.** The 31B loops on malformed calls (26 runs). The 12B writes almost none (0.2% of calls). It loops on edits that can't apply: 25 of its 26 edit runs return `FileEditError` at every call, and 586 of its 804 edit calls fail, against 46 of 489. In both sizes the error comes back and the copy goes on (observed; G2 measured the cue at 31B only).
- **Resolution:** 24 of 218 against 72 (−22.0 points, −29.8 to −15.1). The 12B solves 16 tasks at least once, 15 of them also the 31B's. At its non-loop rate, its loop task runs would add about 5 of the 48: the gap is mostly elsewhere (fewer edits tried, most of them failing, 86 task runs at the call cap).
- **Reading, with §2.2:** if the curve's height is selection of states where the model is near certain of the call, the 12B reaches such states more often and is as certain in them. Selection wasn't measured at 12B.
- **Caveats:**
  - **The same bytes are not the same agent.** The 12B calls the analyzer sub-agent (68 and 75 times; the 31B never does). The sub-agent's calls count toward the 60-call cap, and 26 of 68 and 30 of 75 of its invocations end on its own context window. The loop metrics count the main agent's calls only.
  - Closed loop only: no 12B replay states and no 12B dedup arm, so §2.2, §2.3, §2.5 and §2.6 stay the 31B's.

## 3. Every lever, one step ahead and over whole tasks

| Lever | One step ahead (replay) | Over whole tasks (closed loop) |
|---|---|---|
| A forced thought (T-open) | escape 57.9% against 10.5% where the model skipped thinking | not run |
| An instruction (note, hint) | p(copy) at 3rd+ copies 0.94 → 0.49, 0.45 | not run as such; the harness's budget warning, an instruction, ends 43 of 58 long runs |
| A description (faithful note), varying IDs | no change (+0.020, −0.005) | not run |
| The always-block template | p(c) at copy states 0.791 → 0.216; copies turn into other stale calls | fragments runs, stale share unchanged, resolution unchanged |
| The empty thought before every call (`think_all`) | restores the copy (0.818) | – |
| Deleting the copies | G2's dedup 0.791 → 0.074, but it also deleted calls sent before the run (§4) | dedup of the run only: no fewer loops, resolution unchanged |
| A compaction (uncontrolled) | – | ends 14 of 58 long runs; re-entry follows 30–41% of the time |
| T = 1 | replay p(copy) 0.73 at 3rd+ copies | stale share halves, lock-in under way stays; resolution unchanged |

## 4. Claims withdrawn or narrowed on the way

| Claim as first made | What replaced it | When, where |
|---|---|---|
| The always-block template lowers the stale share 5–6 points | It fragments runs; the stale share doesn't fall on v31 (+1.6, −4.5 to +7.6). The drop held on v30 only, and even there its interval included 0 | 2026-10-16/19, ablation_G3.md §9 |
| Editing the context beats thinking | Self-selection: T-think skipped the thought where it would have helped | 2026-10-13, loops_reason_10_13.md §5 |
| Lock-in is self-reinforcement (each copy makes the next likelier) | Partly: the copies reinforce (G1) and close the last escapes, but most of the curve is selection (G4) | 2026-10-14 (G1), 2026-10-20 (G4) |
| The repeated tool result is the cue for the copy | The results' content isn't the cue; the call and the boundary before it are (G2) | 2026-10-15, ablation_G2.md §6.2 |
| Dedup is the strongest one-step lever: G2 0.79 → 0.07 at copy states, Gate 1 0.94 → 0.14 at 3rd+ copies | G2's dedup also deleted the calls sent before the run, at 44 of its 56 copy states: histories the agent never had. At the 12 clean states, exact p(c) 0.40 → 0.08 (6 tasks). Gate 1's replay uses the same function (`gate1.dedup`) and wasn't re-checked | 2026-10-20, ablation_G2.md §6.6 |
| G2's "states where a run began" have p(c) 0.074 (and G4's prior of 0.29 at k = 2) | The closed loop measures them directly: 0.78 at k = 2 | 2026-10-20, ablation_G4.md §5.1 |
| v31 doubles malformed calls | Noise: 261 in one pass, 174 in the next, 138 on v30 | 2026-10-16, ablation_G3.md §5.5 |

## 5. What the study can't claim

- **Generality.** Two sizes of one family (Gemma 4 31B and 12B, int4), thinking off, one harness, four repositories and 109 tasks.
  - The 12B tests the descriptive claims only (§2.1, §2.4, §2.8, §2.10). The measured mechanisms (§2.2, §2.3, §2.5, §2.6) and the levers (§2.7) are the 31B's.
  - No other family, no smaller Gemma 4 (E4B, E2B), and no thinking-on comparison at either size.
- **Small effects on resolution.** Changes of 1–2 tasks can't be detected, and an inconclusive difference is not equivalence.
- **The forced thought over whole tasks.** T-open's 47 points are one step ahead; no closed-loop arm forced thoughts.
- **Calibrated confidence.** The replay measures the probability of the recorded call; the closed loop measures re-draws. Neither is calibration.
- **That escapes are progress.** A one-step escape can be a new search that also fails.
- **A submission.** A Kaggle zip can change neither the chat template nor the requests, so the block and dedup test mechanisms, not entries.
- **Where in the model.** Nothing here looks inside the model. "Match" and "pattern" describe behaviour.
- **Replay validity.** The one-step replays score v30 histories in v31's serving env (their p(c) reproduces the recorded copy rate: 0.639 against 0.634 in G2). The closed-loop runs are the test.

## 6. Against prior work

From [loops/literature.md](loops/literature.md), which has the checked references:
- **Known:**
  - loops in coding agents (SWE-Gym, SWE-smith), and for this model in its non-thinking mode (Yuan et al. 2026, and two Kaggle threads in this competition);
  - the mechanisms in neighbouring settings: self-reinforcing repetition (Xu 2022), self-conditioning on errors (Sinha 2026), conversational inertia (Wan 2026), copied surface forms of failed calls (Gumaan 2026, models up to 1.7B), agents not linking actions to outcomes (Liu 2026);
  - Fork-and-Flush (arXiv 2610.07447): flushing the history helps autoresearch agents out of "idea basins". It is the nearest analogue of dedup, at a much larger scale of idea.
- **New here:**
  - the lock-in hazard of a coding agent by run length, in the closed loop, split into selection and reinforcement by holding the request fixed (G4);
  - how loops end: exogenous at T = 0.2 (G3);
  - levers compared at the same states one step ahead, and over whole tasks;
  - temperature in the closed loop, at Google's T = 1 (G3);
  - "loops mark lost runs, and un-looping doesn't recover them", with three levers;
  - the same lock-in curve at two sizes of one family, with only the onset moving (§2.10). SWE-smith reports a weaker model looping far more (runs of 10+ in over 25% of its 32B model's trajectories, under 4% for Claude 3.7); here the weaker model's excess is in how often runs start, not in how firmly they hold.
- **Liu's p(3rd identical action | 2)** is 37% pooled; ours (the k = 2 hazard) is 68% in the control. Report ours under Liu's definition, and note the different models, environments and screening.

## 7. Open

- **Done:** the 12B control (generality), two passes of `ctrl` on Gemma 4 12B, run on 2026-10-20/21 ([ablation/ablation_control.md](ablation/ablation_control.md), §2.10). With it every arm has run, and the experiment round ended on 2026-10-21.
- **Not planned, as limitations:**
  - a forced-thought closed-loop arm; Gate 1's dedup re-split; G2's missing thought cell; other wordings and temperatures;
  - from the 12B control: selection at 12B (a 12B dedup arm), the analyzer sub-agent's effect, and why the budget warning stops fewer of the 12B's runs.
- **Docs not yet synced with G3/G4:**
  - loops_reason_10_13.md's principles table and §3;
  - literature.md's novelty row (c), which quotes dedup 0.14;
  - the two legacy HTML explainers.
- **The paper:** 3,000 words, due 2026-11-12 23:59 UTC. Figures should come from the report files (§8), not from this summary. The outline and figure list come first, in a planning session started from [paper/outline_prompt.md](paper/outline_prompt.md).

## 8. Where everything is

| What | Where |
|---|---|
| Round 1 (T-open, always-block on v30) | [loops/README.md](loops/README.md) §1–2; gate1.md §4 in Gate 1's work dir; `$R/paper/loops/report_p1_1012-1727/` |
| G1, the dose test | [ablation/ablation_G1.md](ablation/ablation_G1.md); `$R/paper/loops/dose_baseline_nothink_args_v30_zip/` |
| G2, the feedback factorial | [ablation/ablation_G2.md](ablation/ablation_G2.md); `$R/paper/loops/fb_baseline_nothink_args_v30_zip/` |
| G3, temperature and replication | [ablation/ablation_G3.md](ablation/ablation_G3.md); `$R/paper/loops/report_p2_1016-1640/` |
| G4, dedup over whole tasks | [ablation/ablation_G4.md](ablation/ablation_G4.md); `$R/paper/loops/report_g4_p2_1019-2337/` |
| The 12B control (generality) | [ablation/ablation_control.md](ablation/ablation_control.md); `$R/paper/loops/report_12b_p2_1020-2212/` (pools both 12B passes with every arm) |
| The mechanisms, the math and the review | [loops/loops_reason_10_13.md](loops/loops_reason_10_13.md) |
| Prior work | [loops/literature.md](loops/literature.md) |
| The design of round 2, its predictions and tests | [loops/README.md](loops/README.md) §3 and the "Tested" sections |
| Gate 1 and the OPSD assessment | [gate1/README.md](gate1/README.md), [opsd_assessment.md](opsd_assessment.md) |
| The runs | `$R/results/loops_*` on Explorer, with $R = /scratch/zha.j/gemma4 |
