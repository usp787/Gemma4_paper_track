# ablation/ablation_G2.md: the feedback factorial (round 2's G2)

<!-- moved from loops/ on 2026-10-17. File names without a folder below (README.md, loops_reason_10_13.md, feedback.py,
     loop_report.py, ...) are in ../loops/ or in the work dir; the links point there. The recap of every ablation is
     ablation_recap_10_17.html. -->

<!-- written 2026-10-15 from $R/paper/loops/fb_baseline_nothink_args_v30_zip (feedback.md, feedback.json, prep.md,
     states.jsonl, score.jsonl, samples.jsonl). The tables of §3 and §4 are feedback.md's. The split into copies and
     returns (§5), the exact probabilities and sampled shares by group, their intervals and the comparison with G1 (§6)
     come from a local analysis of those files and of G1's in a session scratchpad, not in this repo. It resamples tasks
     10,000 times (seed 0), as feedback.md does. The design is in README.md §3 (G2), the code in feedback.py. G1's record
     is ablation_G1.md. -->

**The question:** is the copy cued by the repeated result's surface form, or by what the result means? G1 re-spelled the call and kept the result. G2 keeps every call and changes how the results are rendered, at fixed histories, one step ahead.

**The answer: neither.**
- **Surface:** varying IDs on the repeated results change exact p(c) by −0.005 (−0.009 to −0.002).
- **Meaning:** a faithful note, "The search completed and found zero matches for this pattern in these paths.", changes it by +0.020 (+0.005 to +0.036). That is a little *more* copying, and 5 points fewer sampled escapes.
- **The added tokens by themselves** (placebo − orig): −0.004 (−0.012 to +0.003).
- **What moves the copy is in the history.**
  - Deleting c's earlier copies (`dedup`) takes p(c) from 0.469 to 0.063.
  - Ending the context with the empty thought (`block`) takes it to 0.152.
  - With the empty thought before every earlier call as well (`think_all`), p(c) is back at 0.472.
- **Nearly all of it happens at the 56 copy states,** where c was also the call before. The 45 returns, where c comes back after other calls, cycle over several stale calls, and p(c) there barely moves.
- **G4's condition,** a clear semantic effect beyond placebo, isn't met.

## 1. The run

- **Jobs 11091741–45 at `ed1b419`,** queued at 00:20 EDT on 2026-10-15. The report was written at 03:13. Wheelhouse 31 (vLLM 0.19.1), one A100-SXM4-80GB.
  - prep took 3 min 42 s, and all 17 template checks passed.
  - The first GPU job hit its 2-h limit at 1,512 of 1,589 scored requests, as the chain allows.
  - The second scored the last 77 and drew all 994 sample requests (3 samples each), in 30 min. The third found nothing left.
  - About 2.5 A100-hours, as estimated. Scoring ran at ~2,020 prompt tokens/s.
- **Outputs** in `$R/paper/loops/fb_baseline_nothink_args_v30_zip/`: `prep.md`, `feedback.md`, `feedback.json`, and `states.jsonl`, `score.jsonl` and `samples.jsonl`, each under 1.1 MB.
- **One quirk, as in G1:** two rebuilt p(c) at the run's sampling sum above 1: `fastapi_14099` orig 1.047 and `fastapi_15280` dedup 1.273.
  - The two paths are scored in separate requests, and T = 0.2 amplifies small differences between requests near a tie.
  - No exact sum is above 1.

## 2. The states

- **101 repeat states in 23 tasks** (33 task runs), and **32 first failures as controls** (26 tasks).
  - The repeat states come from Gate 1's run (34), its replicate `_r2` (25) and round 1's `ctrl` (42).
- **Concentration:** 19 of the 23 tasks are fastapi and 4 rich (86 and 15 states).
  - Three tasks give a third of the states: `fastapi_11355` 12, `fastapi_14485` 11, `fastapi_15800` 10.
  - 6 of the 101 come from task runs that resolved.
- **Results and copies:** 74 states are `err_silent` (exit 1) and 27 `ok_empty` (exit 0). c has 2 copies in the context in 45 states, 3–4 in 9, and 5 or more in 47.
- **Copies and returns:**
  - **copies** (c was also the call before): 56 states in 18 tasks, 44 of them with 5 or more copies. The student copied next in 52.
  - **returns** (c comes back after other calls): 45 states in 21 tasks, 38 of them with 2 copies. The student's next call was another stale call in 23, a copy in 12 and a fresh call in 10.
  - 16 tasks have both kinds. `feedback.md` splits them at the run's sampling only; §5 has the rest.

## 3. Repeat states, by arm (from feedback.md)

| Arm | States | p(c), run's sampling | p(c), exact | Sampled: exact c | Stale | Escape | Changes approach |
|---|---|---|---|---|---|---|---|
| orig | 101 | 0.639 | 0.469 | 0.644 | 0.898 | 0.102 | 0.056 |
| dedup | 101 | 0.236 | 0.063 | 0.244 | 0.558 | 0.442 | 0.205 |
| placebo | 101 | 0.631 | 0.465 | 0.640 | 0.891 | 0.109 | 0.056 |
| semantic | 101 | 0.637 | 0.486 | 0.653 | 0.931 | 0.069 | 0.050 |
| break | 101 | 0.641 | 0.461 | 0.637 | 0.878 | 0.122 | 0.053 |
| both | 101 | 0.635 | 0.480 | 0.657 | 0.937 | 0.063 | 0.040 |
| block | 101 | 0.381 | 0.152 | 0.370 | 0.766 | 0.234 | 0.096 |
| think_all | 101 | 0.535 | 0.472 | 0.545 | 0.762 | 0.238 | 0.089 |

- No sample was malformed, and every sample had a call.
- **The controls stay near 0 in every arm.**
  - Exact p(c) is 0.009 in orig, 0.006–0.010 in the four cells and `block`, and 0.001 in `think_all`.
  - Sampled escapes are 0.80–0.98.

## 4. The factorial and the anchors (from feedback.md)

Each value is the mean over all 101 repeat states (23 tasks) of a paired contrast. The 95% intervals come from 10,000 resamples of tasks.

| Contrast | p(c), run's sampling | p(c), exact | Sampled: exact c | Sampled: escape | Sampled: changes approach |
|---|---|---|---|---|---|
| surface: varying IDs (main effect) | +0.004 (−0.007 to +0.017) | −0.005 (−0.009 to −0.002) | −0.000 (−0.014 to +0.016) | +0.003 (−0.008 to +0.015) | −0.007 (−0.016 to +0.002) |
| meaning: the semantic note (main effect) | +0.000 (−0.034 to +0.033) | +0.020 (+0.005 to +0.036) | +0.017 (−0.022 to +0.058) | −0.050 (−0.102 to −0.012) | −0.010 (−0.025 to +0.000) |
| interaction: (both − semantic) − (break − placebo) | −0.013 (−0.033 to +0.007) | −0.002 (−0.008 to +0.004) | +0.007 (−0.024 to +0.041) | −0.020 (−0.047 to +0.000) | −0.007 (−0.037 to +0.018) |
| break − placebo | +0.010 (−0.005 to +0.029) | −0.004 (−0.009 to +0.000) | −0.003 (−0.018 to +0.011) | +0.013 (−0.003 to +0.031) | −0.003 (−0.022 to +0.014) |
| semantic − placebo | +0.007 (−0.028 to +0.041) | +0.021 (+0.005 to +0.039) | +0.013 (−0.026 to +0.056) | −0.040 (−0.092 to −0.003) | −0.007 (−0.025 to +0.007) |
| both − placebo | +0.004 (−0.031 to +0.039) | +0.015 (−0.001 to +0.030) | +0.017 (−0.022 to +0.061) | −0.046 (−0.100 to −0.007) | −0.017 (−0.038 to +0.000) |
| placebo − orig (the added tokens themselves) | −0.009 (−0.068 to +0.044) | −0.004 (−0.012 to +0.003) | −0.003 (−0.064 to +0.046) | +0.007 (−0.031 to +0.063) | −0.000 (−0.028 to +0.022) |
| dedup − orig | −0.403 (−0.572 to −0.246) | −0.407 (−0.477 to −0.329) | −0.399 (−0.581 to −0.236) | +0.340 (+0.195 to +0.500) | +0.149 (+0.044 to +0.282) |
| both − dedup | +0.398 (+0.243 to +0.565) | +0.417 (+0.337 to +0.488) | +0.413 (+0.251 to +0.586) | −0.380 (−0.526 to −0.246) | −0.165 (−0.297 to −0.064) |
| block − orig | −0.258 (−0.394 to −0.127) | −0.317 (−0.380 to −0.250) | −0.274 (−0.425 to −0.128) | +0.132 (+0.064 to +0.210) | +0.040 (+0.007 to +0.084) |
| think_all − block | +0.154 (+0.025 to +0.280) | +0.320 (+0.246 to +0.388) | +0.175 (+0.048 to +0.298) | +0.003 (−0.086 to +0.080) | −0.007 (−0.057 to +0.036) |
| think_all − orig | −0.104 (−0.156 to −0.064) | +0.002 (−0.016 to +0.018) | −0.099 (−0.157 to −0.051) | +0.135 (+0.080 to +0.186) | +0.033 (+0.000 to +0.070) |

`feedback.md` also has both main effects by group, in exact p(c). Surface lies between −0.006 and −0.002 in every group. Meaning lies between +0.014 and +0.027, and its interval excludes 0 in all four groups: 2 copies, 3 or more, exit 1 and exit 0.

## 5. Copies and returns (local split)

"Other stale" is a stale call other than c, that is, a repeat of another earlier call.

| Arm | Copies: p(c) exact | Run's sampling | Sampled: c / other stale / escape | Returns: p(c) exact | Run's sampling | Sampled: c / other stale / escape |
|---|---|---|---|---|---|---|
| orig | 0.791 | 0.909 | 0.92 / 0.05 / 0.03 | 0.069 | 0.303 | 0.30 / 0.50 / 0.19 |
| dedup | 0.074 | 0.291 | 0.29 / 0.28 / 0.43 | 0.048 | 0.168 | 0.19 / 0.36 / 0.45 |
| placebo | 0.788 | 0.942 | 0.96 / 0.04 / 0.00 | 0.064 | 0.243 | 0.24 / 0.51 / 0.24 |
| semantic | 0.803 | 0.909 | 0.92 / 0.08 / 0.00 | 0.091 | 0.299 | 0.32 / 0.53 / 0.16 |
| break | 0.781 | 0.944 | 0.95 / 0.04 / 0.01 | 0.063 | 0.264 | 0.24 / 0.49 / 0.27 |
| both | 0.794 | 0.907 | 0.92 / 0.08 / 0.00 | 0.088 | 0.295 | 0.33 / 0.53 / 0.14 |
| block | 0.216 | 0.484 | 0.49 / 0.38 / 0.12 | 0.072 | 0.253 | 0.21 / 0.41 / 0.37 |
| think_all | 0.818 | 0.858 | 0.86 / 0.03 / 0.11 | 0.041 | 0.134 | 0.16 / 0.45 / 0.39 |

Paired contrasts, with 95% intervals over tasks (18 for copies, 21 for returns):

| Contrast | Copies | Returns |
|---|---|---|
| dedup − orig, p(c) exact | −0.717 (−0.816 to −0.633) | −0.021 (−0.055 to +0.018) |
| dedup − orig, sampled escape | +0.405 (+0.211 to +0.635) | +0.259 (+0.109 to +0.404) |
| block − orig, p(c) exact | −0.574 (−0.661 to −0.492) | +0.003 (−0.029 to +0.032) |
| block − orig, sampled other stale | +0.327 (+0.149 to +0.531) | −0.089 (−0.197 to +0.031) |
| block − orig, sampled escape | +0.095 (+0.007 to +0.190) | +0.178 (+0.074 to +0.316) |
| dedup − block, sampled escape | +0.310 (+0.061 to +0.577) | +0.081 (−0.077 to +0.227) |
| think_all − block, p(c) exact | +0.601 (+0.504 to +0.697) | −0.031 (−0.062 to −0.001) |
| think_all − orig, p(c) exact | +0.027 (−0.002 to +0.049) | −0.028 (−0.049 to −0.011) |
| think_all − orig, p(c) at the run's sampling | −0.051 (−0.117 to −0.005) | −0.169 (−0.277 to −0.090) |
| semantic − placebo, p(c) exact | +0.016 (−0.000 to +0.031) | +0.028 (+0.006 to +0.057) |
| semantic − placebo, sampled escape | 0 (no escapes in either) | −0.089 (−0.219 to −0.007) |
| break − placebo, p(c) exact | −0.007 (−0.012 to −0.002) | −0.001 (−0.007 to +0.005) |
| dedup − orig, sampled: opens the empty thought itself | +0.464 (+0.259 to +0.646) | +0.059 (+0.000 to +0.135) |

## 6. What it shows

**1. The results' content doesn't cue the copy.**
- **Surface:** one ID or varying IDs makes no practical difference: −0.005 over all states, −0.007 at copies.
- **Meaning:** the faithful note doesn't help.
  - It raises p(c) a little, in 73% of states on the log-odds scale. In exact p(c), 17 states rise by more than 0.05 and 4 fall by more than 0.05.
  - At returns it costs escapes (−0.089).
  - An example, `fastapi_14099` call 18, a return to `grep -n "StreamingResponse" fastapi/routing.py`. In `orig` and `placebo` all 3 samples broaden the search to `grep -r "StreamingResponse" .`; in `semantic` all 3 re-send c.
  - It holds for exit 1, which the harness renders as `"status": "error"`, `"error_type": "CommandError"`: meaning is +0.021 (+0.004 to +0.045) there.
- **The added tokens** do nothing by themselves (placebo − orig −0.004).
- **Description against instruction:** round 1's note and hint tell the model what to do ("Do not run it again; …"), and they did move the copy. At 3rd-or-later copies they took p(copy) from 0.94 to 0.49 and 0.45 (gate1.md §2). So a description of the failure doesn't move the copy, and an instruction does. Those texts went on the last result only and ran at other states, so this contrasts two designs; it isn't a measured interaction.

**2. With G1, the cue is the call and the boundary before it.**
- **G1:** the copy builds up only when the identical call recurs; re-spelled copies stay flat.
- **G2:** the results can differ (IDs) or state what they mean (the note) without changing it.
- The tokens just before each call, the end of the tool response, are the same after every result, IDs included. What is left to match is the earlier calls and the boundary each of them followed.

**3. The block is a pattern break, not a mode switch.**
- **At copy states, `think_all` restores the copy in full.**
  - Exact p(c) is 0.818, against 0.791 with no thought and 0.216 with the thought only at the end.
  - At the run's sampling, 88% comes back: 0.858, against 0.909 and 0.484.
- **By README §3's reading,** the block works by breaking the match between the end of the context and what preceded the earlier copies. The thinking-mode marker isn't needed to explain it.
- **Where there is no run to match,** at returns and controls, `think_all` lowers p(c) a little: −0.028 at returns, and 0.009 → 0.001 at controls.

**4. Dedup and the block do different things with the copies.**
- **The block** turns many copies into other stale calls (+0.33), with few escapes (+0.10). That is the one-step version of the always-block run's fragmentation (loops_reason_10_13.md §1): runs broke up, more started, and resolution didn't move.
- **Dedup** turns them into escapes: +0.41, and +0.31 over the block.

**5. Returns are another regime.**
- At returns the model cycles over several stale calls: in orig's samples, c 0.30 and other stale calls 0.50.
- p(c) is low and barely moves. Still, `dedup`, `block` and `think_all` each raise sampled escapes, by 0.18–0.26.
- Measure returns by the stale share, not by p(c).

**6. Selection, measured.**
- Dedup turns a copy state back into roughly the state where its run began: c once, after the same history.
- There p(c) is already 0.074 exact (0.291 at the run's sampling), against 0.009 (0.024) at first failures.
- The differences are +0.065 (+0.026 to +0.104) exact and +0.267 (+0.077 to +0.446) at the run's sampling, resampling tasks within each set.
- The same comparison with G1:

| | Without copies | With copies |
|---|---|---|
| First failures | 0.009 (G2's controls; G1's `dose1`: 0.006) | 0.307 (G1's `dose4`: three added copies) |
| States where a run began (G2's copy states) | 0.074 (`dedup`) | 0.791 (`orig`; 44 of 56 with 5 or more copies) |

- The values are exact p(c). At the run's sampling they are 0.024 (0.002), 0.434, 0.291 and 0.909.
- The populations and the numbers of copies differ, so the comparison is rough.
- Still, neither half reaches the closed-loop curve alone. Runs begin where the model already leans towards c, and the copies then lock it in. This is G1's "the curve also reflects which histories loop", now with a size.

**7. The copies also suppress the model's own empty thought.**
- At copy states the model opens it 11% of the time; with the copies deleted, 58% (+0.46). At returns the two rates are 43% and 49%.
- Round 1 found that spontaneous thinking skips looping states (loops_reason_10_13.md §5). Skipping may be part of the copied pattern, since the earlier copies went straight to the call.
- This is the thinking-off empty thought, while T-think ran with thinking on. So it is a lead, not a test.

**8. The replay reproduces the runs.**
- Mean p(c) at the run's sampling is 0.639, and the sampled copies come to 0.644. The recorded copy rate is 0.634 (64 of 101).
- Exact p(c) separates the states the student copied from the others with an AUC of 0.95.
- These are v30 histories scored in v31's serving env.

## 7. What it doesn't show

- *Other wordings.* Each note has one wording, by design: README §3 rules out sweeps. The semantic note leaves `"status": "error"` in place on exit-1 results.
- *A stronger surface break.* The IDs are 8 digits in each result. The rest of the result, and the closing tokens before each call, stay identical.
- *Whole tasks.* It is one step ahead.
- *Generality.* 23 tasks, 19 of them fastapi; three tasks give a third of the states.
- *Real thoughts.* `think_all` separates match from mode for the empty thought only. T-open's forced thoughts (loops_reason_10_13.md §5) are another lever.
- *Where in the model.* Nothing here looks inside the model; "match" describes the behaviour.

## 8. What it changes

- **loops_reason_10_13.md's "Copying" principle** names the repeated tool result as the cue, with format or mode conditioning as the rival.
  - The results' content isn't the cue, and mode conditioning isn't needed to explain the block. What remains is a match on the agent's own earlier calls and the boundary before them.
  - In §7's stylised model, $m_t(c)$ would be that boundary match, not the last result's.
  - Not yet edited there.
- **G4 doesn't run as written:** its condition fails.
  - Proposed for its slot, not decided: a closed-loop `dedup` arm against v31's `ctrl`, collapsing consecutive identical call/result pairs.
  - It is the strongest one-step lever, and the one that turns copies into escapes.
  - A chat template may be able to carry it, as it carried the always-block; check that first.
- **G3** is unchanged. Add the copy/return split to [loop_report.py](../loops/loop_report.py), and measure cycles by the stale share.
- **A cheap follow-up, not built:** the missing cell of the thought 2 × 2, with the empty thought before every past call but not at the end. If the match reading holds, the model writes the empty thought itself and copies through it.
