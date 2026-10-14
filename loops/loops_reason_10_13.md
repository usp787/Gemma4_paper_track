# loops/loops_reason_10_13.md: why the tool-call loops happen

<!-- written 2026-10-13, and revised the same day after the review at the end of this file (GPT 6.1 Sol). Its
     corrections are folded into sections 1-8, its rollout argument is section 9, and its ablation plan is round 2 in
     README.md section 3. Numbers come from:
     - the closed-loop control and always-block arms, $R/paper/loops/report_p1_1012-1727 (loops.md, loops.json,
       baseline_dev.md, baseline_rich.md);
     - Gate 1's work dir, gate1_baseline_nothink_args_v30_zip (gate1.md sections 1-4);
     - ../gate1/README.md, and the main track's agentic_workflow/README.md for the submit_patch loops.
     Re-derived on 2026-10-13 from the local copies in ../../Gemma4_kaggle_repo/logs/loop_audit_20261013/: section 5's
     paired intervals (20,000 task resamples), the always-block table, the seed count and the fastapi_12942 traces. That
     script sits in a session scratchpad, not in this repo; so do the 10-12 checks behind the stuck-stretch counts.
     The citations were checked against the primary texts on 2026-10-13.
     loop_illustration.html's "Why loops happen" section still has the wording from before this revision. -->

**The question:** why does Gemma 4 31B with thinking off lock onto one tool call, and why do only some levers break the lock?

**The short answer:** five principles, one for each stage of the loop. One is a causal effect we measured; the others are observed patterns, mechanisms known from neighbouring settings, or hypotheses.

| Stage | Principle | Status |
|---|---|---|
| Lock-in | The longer a run of identical calls, the likelier one more copy. | Observed here. Self-reinforcement, established in text, is the candidate cause; not isolated here |
| Copying | The repeated tool result is the cue for the copy (induction). | Hypothesis, with a rival: format or mode conditioning |
| Hardening | Low temperature concentrates each token's choice. | Established; large at our replay states; its share of closed-loop lock-in is untested |
| Onset | Loops start at searches that come back empty, then get re-sent unchanged. | Seen in our traces |
| No escape | Spontaneous thinking often skips looping states where a forced thought helps. | Measured here (T-open), one step ahead |

**Terms** follow [loop_illustration.html](loop_illustration.html), "Terms used on this page", with lock-in and the replay's probability stated more exactly:
- a *copy* repeats the call just before;
- a *stale repeat* is any call identical to an earlier one, with no successful `edit_file` or `write_file` in between. It stands in for a repeat that can't teach anything, and it misses edits made through `run_command` (§4);
- *lock-in after k* is P(the next call is one more copy | the current run of identical calls has length k). At k = 1 the last call differs from the one before it; k = 3+ pools every run of 3 or more;
- *p(copy)* is the replay's probability of the student's recorded call at the run's sampling (T = 0.2, top-p 0.95, top-k 64). It is rebuilt from the top 20 logprobs, along the call's recorded tokens (§3);
- an *escape* is a sampled call that parses and is neither stale nor malformed (`escaped()` in [../gate1/gate1.py](../gate1/gate1.py)). It is never executed, so it may still be useless, such as a new search that also finds nothing.

## 1. Lock-in: persistence rises with run length

*Status: observed here. Self-reinforcement is established in text generation; our curve is consistent with it but doesn't isolate it.*

The control arm's numbers over 109 tasks, dev and rich pooled (`loops.json`):

| Current run of identical calls | P(next call is a copy) | Cases | Odds |
|---|---|---|---|
| k = 1 (the last call differs from the one before it) | 2.7% | 2,649 | 0.03 |
| k = 2 | 77.5% | 71 | 3.4 |
| k = 3+ | 95.7% | 1,176 | 22.5 |

- **What the curve shows:** the chance of one more copy rises sharply with the observed run length, as self-reinforcement predicts.
- **What it can't show:** that each copy makes the next one likelier.
  - It compares different histories met during rollouts. Long runs can be enriched for sticky tasks or commands even if no copy changes anything: survivor selection.
  - The k = 3+ value is weighted by transitions, so long runs contribute many cases.
- **Where self-reinforcement is shown:** in text, by feeding a model a constructed input in which one phrase or sentence repeats a chosen number of times.
  - Holtzman et al.'s Fig. 4 repeats "I don't know." 200 times: the phrase's probability rises with each repetition.
  - Xu et al. repeat sentences 100 times: the probability of repetition rises "almost monotonically", then "stabilizes around a certain ceiling value".
  - A Pólya urn, where every ball drawn adds another of its colour, is a picture of this, not evidence.
- **The interventional evidence here:** at the same replay states, deleting the earlier copies (dedup) takes p(copy) at 3rd-or-later copies from 0.94 to 0.14 (gate1.md §2). Dedup also shortens the context and moves positions, so it isn't a clean dose test. Round 2's G1 adds 1, 2 or 4 copies to fixed histories ([README.md](README.md) §3).
- **Run length, as an illustration.** If each call continued the run with a constant probability $q$, the number of further copies $N$ before the first call that isn't one would be geometric:

$$P(N = n) = q^{n}(1-q), \qquad \mathbb{E}[N] = \frac{q}{1-q}$$

  The $1/(1-q)$ used here before also counts the call that ends the run.

| Arm | q at k = 3+ | Expected further copies, if q stayed constant |
|---|---|---|
| Control | 0.957 (dev 0.957, rich 0.958) | ≈ 22 |
| Always-block | 0.890 (dev 0.844, rich 0.920) | ≈ 8 (dev 5.4, rich 11.5) |

  - Histories differ, and warnings, compactions and task ends interrupt runs, so a constant $q$ is an illustration, not a fit.
  - Near $q = 1$ the mean is very sensitive to $q$, since $d\,\mathbb{E}[N]/dq = 1/(1-q)^2$. A modest drop in $q$ shortens runs sharply.
- **How runs of 3+ end.** In the control arm, 48 of 55 end under an outside event and 7 on their own.
  - `end_of()` in [loop_report.py](loop_report.py) checks, in this order: the task ends inside the run; a budget warning arrived in any of its results; the next prompt is shorter (a compaction).
  - A task end can cut off a loop that was still going, and a warning can arrive well before the run changes. So this is a classification, not a cause.
- **The always-block template over whole tasks,** control → block:

| Metric | Dev, 70 tasks | Rich, 39 tasks |
|---|---|---|
| Resolved | 25 → 24 (Δ −1; 95% CI −6 to +3) | 12 → 13 (Δ +1; −2 to +5) |
| Stale-call share | 32.3% → 26.3% | 53.6% → 48.3% |
| Copy starts (share of calls) | 1.5% → 3.1% | 2.3% → 4.4% |
| Lock-in at k = 3+ | 95.7% → 84.4% | 95.8% → 92.0% |
| Runs of 3+ identical calls | 28 → 50 | 27 → 37 |
| Runs of 5+ identical calls | 23 → 36 | 20 → 31 |
| Tasks with at least half their calls bad | 15 → 6 | 19 → 17 |
| Malformed-call share | 4.0% → 1.6% | 2.5% → 5.5% |
| Mean agent seconds per task | 120 → 117 | 98 → 116 |

  - Both arms resolve 37 of 109. The block lowers the stale share and fragments runs of identical calls, but more runs start. On rich it also costs time and malformed calls.
  - A broader measure, stretches of 3+ bad calls in a row, counts any stale or malformed calls, not only identical ones. Their number barely changed (36 → 41 on dev, 36 → 29 on rich), and they got shorter (18.9 → 11.0 calls on dev, 22.0 → 19.9 on rich). The two measures count different things; don't mix them.

## 2. The repeated result as the cue for the copy

*Status: hypothesis. It fits our observations, but every lever we tried changes more than the cue, and format conditioning explains some of them too.*

- **The mechanism.** Induction heads (Olsson et al.) find an earlier occurrence of the current token and copy what came after it: [A][B] … [A] → [B].
  - The paper's evidence is causal in small models, from ablations. For large models with MLPs it is "mainly correlational": "we do not have ablations for our full-scale models".
  - No one has intervened on Gemma's heads here.
- **How it would apply to a tool loop:**
  - 91% of stale repeats got exactly the result their earlier copy got ([../gate1/README.md](../gate1/README.md)).
  - So the context ends with a result $r$ that appeared before, and the previous time, $r$ was followed by call $c$.
  - On this reading, the repeated result doesn't tell the model to stop. It is the cue to send $c$.
- **Four observations fit, each with another explanation:**
  1. **p(copy) is higher at 3rd-or-later copies than at 2nd copies.** At T = 1 with top-p 0.95 it is 0.26 at 2nd copies and 0.73 at 3rd-or-later ones (gate1.md §2). At the first repeat, $r$ has never been followed by $c$, so the cue doesn't exist yet. *But* each class is defined by the call the student made, so the two are different, outcome-selected sets of states, not one history with one more repetition. The T = 1 values are also approximate (§3).
  2. **Changing the last result weakens the copy.** The note and the hindsight hint add text to the last tool result. At 3rd-or-later copies they take p(copy) from 0.94 to 0.49 and to 0.45 (gate1.md §2). *But* both change what the result says, and both tell the model what to do ("Do not run it again"), so they don't isolate the surface match.
  3. **The empty thought puts four tokens after the result.** The template strips past empty thoughts, so those tokens never appear in the history, and the end of the context no longer matches the earlier "$r$, then $c$". p(copy) falls from 0.95 to 0.50 on the 40 third-or-later copies where it applies (gate1.md §3). *But* the same four tokens mark the thinking mode. Google's formatting notes say to "consider adding an empty thinking token" for edge cases where the larger models open a thought with thinking off, and HF PR #147 proposes the same edit after tool responses. So the result fits mode conditioning as well as a broken cue.
  4. **The always-block run on `fastapi_12942` alternates two failing greps.**
     - A is `grep -r "get_endpoint_params" .`: exit 1, no output. B is the same search in `fastapi/`, piped to `head`: exit 0 and no output, because `head` hides grep's exit code. These are the two kinds of empty result in §6.
     - The run goes A, B, A, B (calls 5–8), then A ×9. Call 18 is a different grep, which finds something. Calls 19–56 are A ×5, B, A ×6, B, A ×5, B, A ×5, B, A ×4, B, A ×8. Then the agent submits.
     - Whenever B comes back empty, the history already holds "B's result, then A", first at calls 6–7.
     - The alternation is real, but the trace doesn't show that induction causes it.
     - The control run of the same task ends differently: 35 copies of one `edit_file`, each answered "No changes to apply. old_string and new_string are identical."
- **The Bayesian version is an analogy.** Xie et al. derive in-context learning as implicit Bayesian inference over a latent concept. They assume pretraining data drawn from a mixture of hidden Markov models, and a model that fits that distribution exactly. Read the agent's past calls as examples of "what this agent does here", and each copy is more evidence for "it sends $c$". But self-generated, correlated actions aren't independent evidence, so this doesn't derive §1's curve.
- **Agents show the same behaviour, without a settled mechanism.**
  - Sinha et al. show self-conditioning by injecting errors into the history at chosen rates.
  - Wan et al. find that agents "mimic their own previous responses as few-shot examples", and tie it to diagonal attention. They "do not claim definitive causal identification".
- **The test is round 2's G2** ([README.md](README.md) §3), a 2 × 2 replay factorial:
  - the repeated results either keep their surface form or get varying arbitrary IDs;
  - they carry either neutral text or a faithful reading of the result.
  - One more context keeps the empty thought in the history, which separates a broken cue from mode conditioning.

## 3. Low temperature concentrates each choice

*Status: established at the token level, and large at our replay states. How much of closed-loop lock-in it explains is untested.*

Sampling at temperature $T$ draws each token from the renormalised

$$p_T(x_t \mid x_{<t}) = \frac{p(x_t \mid x_{<t})^{1/T}}{Z_T(x_{<t})}, \qquad Z_T(x_{<t}) = \sum_{v} p(v \mid x_{<t})^{1/T},$$

then truncates to the top-k and the top-p nucleus.
- Two tokens at the same position get odds $\big(p(a)/p(b)\big)^{1/T}$, before truncation or after it when both survive. At T = 0.2, 2 : 1 becomes 32 : 1.
- A whole call is a product over its tokens:

$$p_T(c \mid h) = \prod_{t} \frac{p(c_t \mid h, c_{<t})^{1/T}}{Z_T(h, c_{<t})}$$

- Once two calls diverge, their normalisers differ, and truncation can keep different tokens. So the odds between whole calls are not in general raised to $1/T$.

The replay scores the student's recorded call at the same states under both samplers (gate1.md §2):

| State | p(recorded call), T = 1, top-p 0.95 | At the run's T = 0.2 |
|---|---|---|
| 2nd copy | 0.26 | 0.77 |
| 3rd-or-later copy | 0.73 | 0.94 |
| Routine call, for reference | 0.52 | 0.78 |

- **These are approximations.**
  - They are rebuilt from the top 20 logprobs, but the run samples from the top 64. Dropping the mass outside the top 20 can overstate the kept tokens' probabilities, especially at T = 1, and can shift the rebuilt nucleus. The error hasn't been bounded.
  - Each is the probability of the recorded token sequence, not of every way to write the same call.
  - Before splitting lock-in between the model and the sampler, check the cutoff's coverage, or rescore with at least the configured top-k.
- **What the table supports:** at these replay states the sampler sharpens a lot. At 2nd copies, the recorded copy goes from 0.26 at T = 1 to 0.77 at T = 0.2.
- **What it doesn't support:**
  - *That most closed-loop lock-in comes from the sampler.* Closed-loop k = 2 (77.5%, §1) is P(a 3rd identical call | two in a row), over every history. A replay 2nd-copy state is the 2nd identical call, at states picked because the student did copy. They are different estimands; their similar values don't apportion anything.
  - *That in-context reinforcement raised the model's own preference from 0.26 to 0.73.* Those are different outcome-selected classes (§2).
- **The classic result.** Holtzman et al. found that maximization-based decoding gives degenerate text that "gets stuck in repetitive loops".
  - In their Table 1, greedy output is 73.66% repetition and beam search 28.94%, against 0.28% for human text.
  - Stochastic methods repeat too when their parameters are set so low that they mimic greedy search.
  - They show this empirically; they don't derive that decoding settles into a cycle.
  - Pipis et al. find that reasoning models loop most at low temperature: "temperature is a reasonable fix when errors are small, but a stopgap when they are large".
- **The limit of the fix.** At 3rd-or-later copies the recorded copy keeps 0.73 at T = 1, so a higher temperature would likely weaken lock-in, not remove it.
  - The only closed-loop data at T = 1 is `agentic_v2` on v28, a different prompt. It still locked in at 73%.
  - Round 2's G3 adds the matched `off_args` arm at Google's T = 1.0 ([README.md](README.md) §3; [literature.md](literature.md) §6).

## 4. Loops start at searches that come back empty

*Status: seen in our traces. The "can't teach anything" argument holds only for unchanged, read-only calls, and our stale label approximates them.*

- **When a repeat is worthless.** In a deterministic environment that hasn't changed, an identical read-only call returns what it returned before, so it adds no evidence.
  - A rational searcher wouldn't make it.
  - Next-token and imitation objectives have no explicit term for information gain. The model picks what is likely given the context.
- **Our stale label approximates that set; it isn't the set.**
  - [trigger_replay.py](../adaptive_inference/trigger_replay.py) clears staleness only after a successful `edit_file` or `write_file`, so it misses edits made through `run_command`. The replay's states include repeated shell writes such as `cat <<EOF > tests/test_ref_handling.py`.
  - A repeat can become useful after the environment changes. A re-read after compaction can restore what the visible history lost.
  - A rough check on §5's skipped states: dropping the 10 whose recorded call writes leaves 11/104 escapes for T-think and 59/104 for T-open (all 114: 12 and 66). That isn't a full adjudication.
- **The onset in our traces** ([../gate1/README.md](../gate1/README.md), "How a loop starts"):
  1. The model searches for something it strongly expects, such as a plausible function name that isn't in the file (`get_openapi_route`).
  2. The empty result contradicts that expectation, but gives little text to condition on. In `off_args`'s first v30 run, 270 of the 580 repeated `run_command` calls got nothing back.
  3. After an empty search, the next call is a variant 78% of the time and a copy 1%. When the variants fail too, the model returns to its first and most likely query: 114 of 131 chains of bad calls start by re-sending a call from 2 or more calls back.
- **Agents use action–outcome pairing only modestly.** Liu et al. shuffle which past action goes with which observation.
  - Task success falls by only 2.55 points overall (46.39% → 43.84%; 0.30–5.40 points by model). That suggests limited use of the pairings, not none.
  - Their two fixes differ. Outcome labels on past actions ("Causal Transition") raise success to 48.25%. Generated experience notes cut next-action repetition the most, but lower success to 45.01%.
- **Compounding errors, as an analogy.** Take a policy trained by plain imitation that errs with probability ε on the expert's own states. Over $T$ steps it can incur up to $T^2\varepsilon$ extra cost (Ross & Bagnell 2010, restated as Theorem 2.1 by Ross, Gordon & Bagnell). DAgger's interactive training brings the bound down to linear in $T$ when the expert can recover from the learner's mistakes.
  - The intuition, in their words: after a mistake the learner "may encounter completely different observations than those under expert demonstration".
  - The bound is tight on some problems, not a law. It doesn't describe how Gemma was trained, and not every mistake leads to an unfamiliar state.
  - A made-up function name may lead to such a state, and copying its own recent behaviour is one fallback there.
- **The reasoning-model parallel.** Pipis et al. show, on a synthetic graph task with a small Transformer, that when the progress-making action is hard to learn but an easy cyclic action exists, the model puts relatively more probability on the cyclic one.
  - A distilled student can loop much more than its teacher: 30% against 4% in their example pair, at greedy decoding on AIME.
  - Their loops are in reasoning text, and "cyclic" means returning to a visited state, not an identical call. The mapping to SWE tool calls is loose.

## 5. Spontaneous thinking often skips the states where a thought helps

*Status: measured here (T-open), one step ahead.*

- **The normative frame.** Metareasoning (Russell & Wefald) says to deliberate when the expected gain from thinking exceeds its cost, the value of computation:

$$\mathrm{VOC}(h) = \mathbb{E}\big[U(a_{\text{after thought}})\big] - \mathbb{E}\big[U(a_{\text{now}})\big] - c_{\text{thought}}$$

  The replay measures no utility and no thought cost, so VOC frames the question; it isn't a result.
- **The pattern.** Free to think (T-think), the model writes a real thought at 72% of 2nd copies and 23% of 3rd-or-later copies, while the recorded copy's probability rises from 0.77 to 0.94 (gate1.md §2 and §4).
  - It thinks less where its recorded call is more probable.
  - That fits a trigger driven by confidence, but the replay doesn't measure calibrated confidence or entropy.
- **T-open forces the thought.** On the 213 looping states in 44 tasks, paired by state, with 95% intervals from 20,000 resamples of tasks:

| Looping states | T-think escapes | T-open escapes | Paired difference (95% interval) |
|---|---|---|---|
| T-think skipped the thought: 114 states, 36 tasks | 12/114 = 10.5% | 66/114 = 57.9% | +47.4 points (33.0 to 61.1) |
| T-think thought: 99 states, 34 tasks | 61/99 = 61.6% | 62/99 = 62.6% | +1.0 points (−6.7 to 9.1) |
| All looping states: 213, in 44 tasks | 73/213 = 34.3% | 128/213 = 60.1% | +25.8 points (16.0 to 36.0) |

  - Where T-think thought, the similar rates hide churn: 15 states gain an escape under T-open and 14 lose one.
  - T-think's samples on the first run shared seeds within tasks: 581 states had 75 seeds. T-open has one seed per state. Resampling whole tasks respects that clustering, but a replication should use independent seeds.
  - These are one-step escapes in a selected replay population, not solved tasks.
- **What it supports:** spontaneous thinking often skips states where a forced thought raises one-step escape a lot, by 47 points.
- **What it doesn't support:**
  - *That the trigger is blind to the benefit.* Where the model chose to think, a forced thought escapes as often as its own (63% against 62%, same states), so a forced thought is no worse than a chosen one. But the benefit is escape with a thought minus escape without one, and the replay has no thought-free sample where T-think thought. Round 2 bounds it from `score_S.jsonl` ([README.md](README.md) §3, the bonus).
  - *That forcing adds nothing where the model thinks.* +1.0 (−6.7 to 9.1) is a failure to find a difference, not an equivalence test.
  - *That VOC is highest where uncertainty is lowest.* Neither was measured.
  - *That an escape is progress.* It can be a new search that also fails.
- **Confidence-based triggers may miss these states, but no one has shown that they do.**
  - HRBench uses single-turn benchmarks. One case note there is headed "Confident wrong answers": on AIME, an entropy-based re-think trigger fired, but "too late—after the model has already committed to a wrong approach". The paper leaves agentic switching "unexplored".
  - CogRouter has the agent pick its own thinking level, trained on a task-success × format reward.
    - Action confidence only splits a successful trajectory's credit across levels, for a fixed action. Failed trajectories get no such reweighting.
    - In its ablation, applying the reweighting to failed trajectories too made the model collapse to the shallowest level (83.1%) "to evade 'confident mistakes'".
    - It never tests loops.
- **What a useful trigger could read.** A failure signal in the history, such as "my last call returned the same result as before", is motivated by the table above. Its benefit over whole tasks is untested.
  - A harness sees such signals, so a controller outside the model could decide when to think.
  - A Kaggle agent can't force a thought at runtime (it has no reply prefix and no callbacks), so there the trigger would have to live in the weights.

## 6. What is specific to tool calls and code

- **The environment echoes.** Each cycle adds two identical pieces to the context, the call and its identical result. In free text, only the model's own words repeat.
- **Calls are short, rigid strings.** Copying a whole call is cheap and usually gives valid syntax. Malformed calls can repeat too: the replay's `malformed` class.
- **Empty results say the least.** A grep that exits 1 with no output is scoped negative evidence: this pattern isn't in these files.
  - A first such search can be good exploration; repeating it unchanged adds little.
  - An `ok` with empty output is more ambiguous: `grep … | head` exits 0 whatever grep found.
- **Interpretable failures, in a neighbouring study.** Sethi et al. study a different failure: final answers that assert values a tool didn't return.
  - Under their deployment prompt, dishonest answers were 0% when the tool returned `status: error` (HTTP 500, timeout), and up to 45.3% when it returned `status: ok` with an unusable payload.
  - That 45.3% is the redacted-permission case. An empty result gave 8.1%.
  - The lesson is that interpretable failure signals mislead less than ambiguous ones, not a rate for our greps.
  - Explicit errors can still loop: see `fastapi_12942`'s control run (§2).
- **Contexts grow.** The repeated block grows while the task statement recedes. Compaction cuts both ways: it deletes the copies (a natural dedup) and the evidence of failure. 44% of chains of bad calls re-send a call that compaction had hidden.
- **Not every loop is a search loop.** The main track's thinking-on `budgetfit_contract` agent ends 21 of its 179 task runs in 21–65 identical `submit_patch` calls, while its thoughts say "I'll stop." Inside those runs, 77–79% of the thoughts repeat the one before word for word.
  - `submit_patch` is free, and a session ends only when the model answers without a tool call. This prompt, unlike `budgetfit`'s, never says how to end.
  - So it looks like a protocol loop. An ablation of that one prompt line would confirm it ([the main-track analysis](../../Gemma4_kaggle_repo/agentic_workflow/README.md#result-2026-10-13-a-tie-keep-a)).
  - Real thoughts can coexist with tool loops. Count search loops, malformed-call loops, termination loops and repetitive thoughts separately.

## 7. One stylised model

An organising hypothesis, not a fitted or identified model:

$$P(a_t = c \mid h_t) \;\propto\; \exp\!\Big(\tfrac{1}{T}\big[\, s(c) + \lambda\, n_t(c) + \mu\, m_t(c) \,\big]\Big)$$

- $s(c)$: the model's prior for call $c$, given the task.
- $n_t(c)$: copies of $c$ already in the context.
- $m_t(c)$: 1 when the last result matches what came before $c$ the previous time, otherwise 0.
- $\lambda$, $\mu$: how strongly the copies and the cue pull.
- $T$: the sampling temperature. Applying it to a whole call's score is a simplification, since the sampler acts per token (§3).

Each lever we tested is meant to change one part of it, and changes more. The numbers are one step ahead, at 3rd-or-later copies (gate1.md §2–§4):

| Lever | Meant to change | Also changes | p(copy) |
|---|---|---|---|
| Copies deleted (dedup) | $n$ | context length and positions | 0.94 → 0.14 |
| Empty thought | $m$ | the thinking-mode marker | 0.95 → 0.50 (40 states) |
| Note; hindsight hint | $m$ | what the result says, plus an instruction | 0.94 → 0.49; 0.45 |
| T = 1 instead of 0.2 | every weight, divided by 5 | nothing else, but the value is approximate (§3) | 0.94 → 0.73 |
| Forced thought (T-open) | the context the terms are computed on | the prompt's mode, the decoding constraints, and the computation available | 0.94 → 0.12 (escape 70%) |

- The table organises the levers; it doesn't isolate one term per lever.
- **What the model has no term for:** information gain, and which call produced which result.
  - That is a choice made in writing the model, not evidence that Gemma represents neither.
  - In this picture, new information would act there: an informative empty result, file locations, or a restart that keeps the earlier diff.
- **What it is consistent with,** without being fitted to it:
  - the large jump in continuation at the first repeat (§1);
  - run length's sensitivity to $q$;
  - fragmentation under the empty thought;
  - why deleting the copies or forcing a thought brings p(copy) furthest down.

## 8. What is established and what is ours

- **Established elsewhere** (text generation, or other agents):
  - in text, repetition in a constructed input raises the chance of more repetition, up to a ceiling (Holtzman et al.; Xu et al.);
  - maximization-based decoding degenerates into repetition (Holtzman et al.), and reasoning models loop most at low temperature (Pipis et al.);
  - imitation's errors can compound quadratically with the horizon, on some problems (Ross et al.);
  - models condition on their own past errors (Sinha et al.) and imitate their own past responses (Wan et al.);
  - an entropy trigger can fail on confident errors, in one single-turn case (HRBench).
- **Measured in our data:**
  - persistence rises sharply with the observed run length (§1);
  - at the same replay states, T = 0.2 against T = 1 changes the recorded copy's probability a lot, approximately (§3);
  - where the model skipped thinking, a forced thought raises one-step escape by 47 points (33 to 61) (§5);
  - the always-block template's mixed effects over whole tasks, with resolution unchanged (§1);
  - loops start at empty searches (§4);
  - termination loops in a thinking-on agent (§6).
- **Fits our data, untested:**
  - that copies cause further copies in our setting (round 2's G1);
  - the repeated result as the copy cue, against format conditioning (G2);
  - the A/B alternation as a copied pattern;
  - conversational inertia behind the drift of the self-written empty thought (the compaction test in loop_illustration.html, open question 2).
- **Not measured yet:**
  - what escapes lead to;
  - any lever's benefit over whole tasks;
  - whether the model's thinking trigger tracks the benefit (§5).
- **The "about 2 tasks" estimate** ([../gate1/README.md](../gate1/README.md), "Spread and ceiling") comes from rescues across runs. 6 of 23 loop failures were solved by at least one of 9 heterogeneous runs.
  - It is not an upper bound on what a new intervention could solve.
  - An escape can also still end in a wrong patch or a timeout.

## 9. What loops mean for rollout data

*Status: an argument, with a cheap test planned (round 2's bonus, [README.md](README.md) §3). No training has been run.*

- **The argument.** Failed rollouts need not be discarded.
  - Distinct failures with interpretable, locally attributable feedback can supply corrective supervision, even without task success.
  - Repeated failures that add no new evidence instead concentrate the rollout budget on redundant states, where a self-teacher may agree with the same ineffective action.
  - So measure feedback quality and corrective teacher disagreement, rather than treating all failed rollouts alike.
  - Whether these differences improve post-training is untested.
- **What our data shows so far:**
  - redundancy is large: stale repeats are 32% of the control arm's dev calls and 54% of its rich calls (§1);
  - explicit feedback moves the action: on the same 47 states, the hindsight hint takes the 3rd-or-later copy from 0.94 to 0.45;
  - yet the averaged signal is weak. T-hint's KL on the failure tokens is 0.020 nats, and its mean log-ratio on the student's tokens −0.010. T-think's failure/routine KL ratio is 0.51 (gate1.md §1). Failures plus a self-teacher don't guarantee strong, well-aimed supervision.
- **Qualifications:**
  - Penalising every failed search would discourage exploration. Target repeats made after the evidence was already in the context.
  - Plain SFT on failed actions reinforces them. Self-distillation can correct them only where the feedback-conditioned teacher moves mass to better actions.
  - Under forward KL, the gradient on the student's logits is $p_S - p_T$, so a large KL alone doesn't show that the correction points the right way.
  - Feedback should be attributable to the local state and action. A privileged hint that relies on information unavailable at deployment may not transfer.
  - Keep the evidence of failure through compaction. Keep failure prefixes with a diagnosis or a corrected continuation, not every copy.
- **Gate 1's top-20 KL is not a lower bound on the full KL.** The floor for missing student tokens lowers the kept terms, but the dropped tail terms can be negative.
  - Take one kept token with mass 0.9 under the teacher and 0.8 under the student, and tails of 0.1 and 0.2.
  - The kept term is $0.9 \ln(0.9/0.8) \approx 0.106$ and the tail's is $0.1 \ln(0.1/0.2) \approx -0.069$, so the full KL is about 0.037.
  - Read the gate's KLs as approximate.
- **Related methods:**
  - HERO turns each next observation into a turn-level diagnosis and learns from failed trajectories too, for instance to avoid redundant API calls.
  - SDPO distils the feedback-conditioned model's predictions back into the policy, with dense, logit-level credit.
  - OPSD matches the student to a teacher with privileged context, by per-token forward KL on the student's own rollouts.
  - They motivate the argument. None of our work shows that distilling the measured differences helps the student.

## References

**New here** (not in [literature.md](literature.md)):

- Russell, S. & Wefald, E. *Principles of metareasoning.* Artificial Intelligence 49 (1991), 361–395. DOI 10.1016/0004-3702(91)90015-C.
  🔗 [**Access Options**](https://onesearch.library.northeastern.edu/openurl/01NEU_INST/01NEU_INST:NUL?genre=article&aulast=Russell&issn=0004-3702&title=Artificial%20Intelligence&atitle=Principles%20of%20metareasoning&volume=49&issue=1-3&spage=361&epage=395&date=1991-05-01&rft_id=info:doi/10.1016%2F0004-3702(91)90015-C&rft_id=info:pmid/&sid=LibKey)
- Ross, S., Gordon, G. J. & Bagnell, J. A. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS 2011. Full text not available through your institution. Open copy: https://arxiv.org/abs/1011.0686
- Xie, S. M., Raghunathan, A., Liang, P. & Ma, T. *An Explanation of In-context Learning as Implicit Bayesian Inference.* ICLR 2022. Full text not available through your institution. Open copy: https://arxiv.org/abs/2111.02080

**Also cited** (details and links in [literature.md](literature.md) §4; the versions read on 2026-10-13):
- Holtzman et al. 2020 (1904.09751);
- Xu et al. 2022 (2206.02369);
- Olsson et al. 2022 (2209.11895);
- Pipis et al., ICML 2026 (2512.12895v3);
- Sinha et al., ICLR 2026 (2509.09677v3);
- Wan et al., ICML 2026 (2602.03664v3);
- Liu, J. et al. 2026 (2610.02769v1);
- Sethi et al. 2026 (2609.14758v1), an anonymised workshop-format preprint;
- HRBench, Ning et al. 2026 (2605.28398v1);
- CogRouter, Yang et al. 2026 (2602.12662v1);
- HERO, Liu, H. et al. 2026 (2606.11559v1); SDPO, Hübotter et al. 2026 (2601.20802v2); OPSD, Zhao et al. 2026 (2601.18734v3);
- Google, *Gemma 4 Prompt Formatting* (updated 2026-06-03) and the *Gemma 4 model card* (updated 2026-07-30);
- the Hugging Face PRs google/gemma-4-31B-it #147 and google/gemma-4-26B-A4B-it #67, both open, from one community author. #147 reports no measurements of its own; #67's are on the 26B model.

## Review Comments,  from GPT 6.1 Sol.

> **Status (2026-10-13):** folded into sections 1–9 above, after checks against the local audit copies, the code and the primary texts. Its ablation plan (its section 10) is round 2 in [README.md](README.md) §3. It stays here as the record. What the checks found:
> - **Numbers:** all reproduced. That covers its section 2's paired intervals (within resampling noise), the always-block table, the 75 seeds, and the write-like sensitivity check (10 write-like states).
> - **Code readings, math and citations:** confirmed, with four refinements:
>   - Google's formatting page says to "consider adding" an empty thinking token, for edge cases in the larger models. It doesn't recommend it in general.
>   - PR #147 is an open community PR. Its measurements come from the 26B model (#67).
>   - HRBench's "Confident wrong answers" is one appendix case note, in which the entropy trigger did fire, too late.
>   - Pipis et al.'s 30% against 4% is one example pair, at greedy decoding.
> - **One argument refined:** "forced thinking adds about 47 points in skipped states and about one point in states where a thought already occurred" compares forcing with the model's own choice. So it doesn't show the trigger carries information about the benefit, either. Both readings need the escape rate without a thought, at the states where the model thought (section 5 above).

*Review recorded from the October 13, 2026 discussion. These comments preserve the original document above and distinguish results, interpretations, corrections, and proposed work. The ablation plan below is the latest evaluated proposal and supersedes the earlier broader suggestions. Appending these comments does not implement experiments, authorize training, or submit jobs.*

### 1. Overall assessment and proposed central claim

This is a strong hypothesis document, but several statements need narrowing before it becomes a defensible causal explanation of why loops happen. The strongest contribution is the mismatch between spontaneous thinking and recoverable failure states. The induction mechanism and unified mathematical model remain hypotheses.

A defensible central claim is:

> In this thinking-off configuration, repeated tool calls become highly persistent. Context interventions and forced thinking change the next-action distribution, but spontaneous thinking frequently skips recoverable states, and reducing exact repetition does not necessarily improve task completion.

Keep four quantities distinct: suppression of the exact recorded copy, escape from all stale or malformed actions, useful progress after execution, and eventual task resolution. We agree that what happens after a replay escape has not yet been fully investigated.

The review used local artifacts, saved Explorer reports and metadata, relevant baseline and October 13 main-track experiments, and primary references. Evidence paths, with \(R=/scratch/zha.j/gemma4\):

- \(R/paper/loops/report_p1_1012-1727/\): loops.md, loops.json, baseline_dev.md, and baseline_rich.md.
- \(R/paper/gate1/gate1_baseline_nothink_args_v30_zip/\): gate1.md, states, generated continuations, and scores.
- \(R/results/ab_budgetfit_contract_vs_budgetfit_v30/compare.md\).
- Local copies: ../../Gemma4_kaggle_repo/logs/loop_audit_20261013/.
- Code definitions: [loop_report.py](loop_report.py), [gate1.py](../gate1/gate1.py), and [trigger_replay.py](../adaptive_inference/trigger_replay.py).

### 2. What the current measurements establish

| Finding | Verified result | Supported interpretation |
|---|---|---|
| Exact-copy persistence, control | 77.5% continuation after two identical calls; 95.7% after three or more | High persistence among observed surviving loops |
| Forced thinking where optional thinking was skipped | 12/114 to 66/114 one-step escapes | Many skipped states are recoverable under the replay's escape definition |
| Forced thinking where optional thinking occurred | 61/99 to 62/99 escapes | Little aggregate additional benefit in this selected group |
| Always-block over complete tasks | Dev 25 to 24 solved; rich 12 to 13; 37/109 for each arm | Behavioral change without demonstrated solve-rate improvement |

Independent paired task-cluster resampling gives:

| Selected looping states | T-think escape | T-open escape | Paired difference, 95% interval |
|---|---|---|---|
| T-think skipped: 114 states, 36 tasks | 12/114 = 10.5% | 66/114 = 57.9% | +47.4 percentage points [33.0, 61.1] |
| T-think thought: 99 states, 34 tasks | 61/99 = 61.6% | 62/99 = 62.6% | +1.0 points [-6.7, 9.1] |
| All looping: 213 states, 44 tasks | 73/213 = 34.3% | 128/213 = 60.1% | +25.8 points [16.0, 36.0] |

Method: pair states.jsonl, gen.jsonl, and gen_T-open.jsonl; reproduce escaped() and the greater-than-four-token thought split; resample tasks with replacement 20,000 times, preserving paired outcomes and all states within each task; compute pooled state differences and percentile intervals. The audit used deterministic LCG seed 20261013. These intervals describe the selected replay population, not whole-task success. The selected-thinking group has 15 gains and 14 losses: similar aggregate rates do not mean identical outcomes.

A rough sensitivity check excluding clearly write-like recorded calls gives 11/104 optional escapes versus 59/104 forced escapes, compared with 12/114 versus 66/114 in the full skipped subset. This is not complete adjudication of environment changes.

These analyses need repository scripts and explicit provenance before publication. The original first-run samples also shared seeds within tasks: 581 states had 75 seeds. Later T-open samples used the corrected state-specific seed scheme. Replication should use independent continuation seeds and retain task clustering.

### 3. Persistence is observed; self-reinforcement is not yet isolated

The lock-in curve does not by itself identify the causal effect of each copy. Longer surviving runs can be enriched for intrinsically sticky tasks or commands even if their continuation probabilities never change. The pooled \(k \geq 3\) estimate is transition-weighted, so long runs contribute many observations.

[Holtzman et al.](https://arxiv.org/html/1904.09751v2) and [Xu et al.](https://arxiv.org/html/2206.02369v2) manipulate repetitions within fixed contexts. Our current curve compares different histories encountered during rollouts. Safer wording is:

> Continuation probability rises sharply with observed run length, consistent with self-reinforcement.

The replay and closed-loop quantities differ:

- Gate 1 labels a state using the student's realized next action. A copy_2nd state evaluates the second identical call, after one adjacent predecessor, and is selected because the student actually copied.
- Closed-loop k=2 predicts the third call after two identical predecessors across all observed eligible histories.

Replay probability 0.77 and closed-loop continuation 77.5% are different estimands. Their numerical similarity does not show that most lock-in comes from the sampler. Likewise, 0.26 to 0.73 compares different outcome-selected states; it is not a controlled estimate of adding one repetition.

The table's “k=1 (any call)” label needs correction: the implementation uses a current run length of exactly one, not every call. The Pólya urn is a qualitative analogy, not evidence for a fixed odds multiplier per copy.

### 4. Arithmetic, sampler probabilities, and endpoint classification

For a constant continuation probability \(q\), the expected number of further copies before the first noncopy is:

\[
\mathbb{E}[\text{further copies}] = \frac{q}{1-q}.
\]

The original \(1/(1-q)\) counts the terminating noncopy as well. At \(q=0.957\), the corrected illustrative expectation is about 22.3 further copies; at \(q=0.890\), about 8.1. Heterogeneous histories, warnings, compaction, and task termination mean that a constant-hazard geometric model is illustrative rather than fitted.

Temperature raises the odds between two next-token alternatives sharing a prefix to \(1/T\), before truncation or when both survive the same truncation. It does not generally raise whole-call odds to \(1/T\): after calls diverge, subsequent normalization constants differ, and top-p/top-k support can change.

Gate 1 reconstructs sampling probabilities from top-20 logprobs while configured sampling uses top-k 64. Especially at T=1, omitted mass can overstate retained-token probabilities and alter the reconstructed nucleus. The current numbers are approximate sampler probabilities, not the unmodified model distribution; their error has not been bounded. Whole-call probability is the product along the exact recorded token serialization, not the total mass of all semantically equivalent tool calls. Validate cutoff coverage or rescore with at least the configured support before making a quantitative sampler-versus-model decomposition.

For the early OPSD diagnostic, the top-20 partial KL sum is not generally a lower bound on full KL. Floor imputation lowers retained-token terms, but omitted tail terms can be negative. For example, teacher/student top-set masses 0.9/0.8 give a retained contribution \(0.9\log(0.9/0.8)\approx0.1060\), while tail masses 0.1/0.2 contribute about -0.0693, leaving full KL about 0.0367. Treat the KL gates as approximate diagnostics.

The finding that 48/55 control runs of three or more calls end under an outside-event category is descriptive. end_of() prioritizes task end, then a budget warning anywhere within the stretch, then a shorter next prompt as compaction. Task end can censor an ongoing loop; a warning can precede the actual change. Say endings coincide with or are classified under outside events, rather than treating this classification as causal identification.

### 5. Induction, Bayesian inference, and alternative template explanations

The repeated-result cue is plausible, but current interventions change several features:

- Dedup removes earlier calls and results, changing length and positions as well as repetition count.
- Notes and hindsight hints change meaning and can give behavioral instructions.
- Empty-thought tokens alter learned channel or mode conditioning.
- T-open changes the prefix, decoding constraints, and available computation.

The lever table does not causally isolate one \(n\), \(m\), or temperature term per intervention. The stylized model is an organizing hypothesis, not an identified or fitted explanation. Omitting information gain and action-outcome tracking from that equation does not prove the actual model has no representation of either.

[Olsson et al.](https://arxiv.org/abs/2209.11895) supports induction as a candidate mechanism, with causal evidence in small attention-only models and more limited evidence for larger models with MLPs. No Gemma head intervention has been performed here. Four channel tokens can change mode conditioning without abolishing a copying circuit.

The Bayesian interpretation from [Xie et al.](https://arxiv.org/html/2111.02080) relies on particular distributional assumptions. Correlated self-generated actions are not automatically independent posterior evidence. Treat this as an analogy, not a derivation of the observed odds curve.

An important alternative is format or mode conditioning. [Google's formatting documentation](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4) recommends an empty thinking channel to stabilize larger Gemma models when thinking is disabled. [HF discussion/PR #147](https://huggingface.co/google/gemma-4-31B-it/discussions/147) proposes the corresponding post-tool-response template change. The block result is compatible with this explanation as well as result-cue disruption.

The fastapi_12942 A/B example is schematic. The actual block trace includes A x9, then a different C call (grep get_model_fields) with nonempty output, then A x5, before another B. Alternating stale searches are real, but this is not a pure A/B cycle and does not identify induction as its cause.

### 6. Thinking allocation, uncertainty, and value of computation

The supported claim is:

> Spontaneous thinking frequently skips states where forced thinking substantially improves one-step escape.

The replay does not measure latent uncertainty, calibrated confidence, entropy, downstream utility, or thought cost. “A stuck agent is a confident agent” is an interpretation; “VOC is highest exactly where uncertainty is lowest” is not established.

The trigger cannot be said to carry no information about benefit. Against optional thinking, forced thinking adds about 47 points in skipped states and about one point in states where a thought already occurred. Similar final forced-escape rates do not imply equal incremental benefit. Failure to find a difference is not an equivalence test.

escaped() means a parsed call that is neither stale nor malformed under the replay's checks. It is not an executed, productive action; an escape can be a new but useless search. This supports misallocated recovery effort, not measured optimal metareasoning.

Do not generalize to every confidence- or entropy-based trigger. [HRBench](https://arxiv.org/html/2605.28398) supports a particular confident-error failure mode, not a universal impossibility. A controller using history-grounded failure signals is motivated, but whole-task benefit remains to be tested.

### 7. Staleness, information gain, and the rollout-learning discussion

The stale-repeat label is a proxy, not exactly the set of deterministic read-only zero-information actions. trigger_replay.py clears stale history only after successful edit_file/write_file calls; it misses mutations inside run_command. The replay includes repeated shell writes such as cat <<EOF > tests/test_ref_handling.py. Repeated commands can become useful after environment changes.

A reread after compaction may restore information missing from visible history even if the environment has not changed. Separate repetition from adjudicated useless repetition. Copying rigid tool strings does not guarantee valid syntax: malformed calls can themselves repeat.

The user's rollout-sampling argument is persuasive after distinguishing task success, environmental evidence, and training signal:

> Distinct failures with interpretable, locally attributable feedback can be valuable training data. Repeated failures that add little new evidence can concentrate sampling on redundant states without producing useful corrective supervision.

A failed rollout can reveal a local mistake and teach an alternative despite ending unsuccessfully. A loop can spend the same rollout budget reproducing nearly the same failure. Sampling and retention should consider diversity of failure states, redundancy, and actionable corrective feedback, rather than successful-versus-unsuccessful outcome alone.

Qualifications:

- Empty stdout with a grep exit code 1 is scoped negative evidence. The first failed search can be reasonable exploration; repeating it unchanged usually adds little fresh evidence. An ok with no output can be more ambiguous.
- Penalizing every unsuccessful search would discourage useful exploration. Target redundant or unjustified repeats after relevant evidence is available.
- Failure is not automatically negative supervision. Plain SFT on failed action tokens reinforces them. OPSD can correct the student when the feedback-conditioned teacher favors better alternatives or reduces mass on the ineffective decision.
- Under full forward KL, the student-logit gradient is \(p_S-p_T\). Larger disagreement or KL alone does not establish appropriately directed correction.
- Teacher feedback should be attributable to the local action and state. Privileged feedback is allowed in distillation, but prescribing behavior that depends on unavailable deployment information creates a transfer risk.
- Preserve evidence of failure through compaction and retain failure prefixes with diagnoses or corrected continuations where justified. Indiscriminately retaining all copies can overweight redundant histories.

T-hint reduces third-or-later copy probability from approximately 0.94 to 0.45 on the same 47 states. That shows explicit feedback can change the local action distribution. Yet aggregate failure KL is 0.020 nats and mean sampled-token log-ratio is -0.010, while T-think's failure/routine KL ratio is 0.51. Having failures and a self-teacher does not guarantee strong, appropriately targeted supervision.

[HERO](https://arxiv.org/html/2606.11559) studies locally aligned observations and turn diagnoses, including learning from failed trajectories. [SDPO](https://arxiv.org/html/2601.20802) connects feedback-conditioned predictions to token-level corrective credit. [OPSD](https://arxiv.org/html/2601.18734v3) supplies the distribution-matching framework. These support the motivation, but our work has not demonstrated that distilling the measured differences improves the student.

Proposed documentation statement:

> Failed rollouts need not be discarded: distinct failures with interpretable, locally attributable feedback can supply corrective supervision even without task success. Repeated failures that add no new environmental evidence may instead concentrate sampling on redundant states where a self-teacher agrees with the same ineffective action. Our results motivate measuring feedback quality and corrective teacher disagreement rather than treating all failed rollouts as equally useful. Whether these differences improve post-training remains untested.

### 8. Whole-task evidence, recoverability, and protocol loops

The always-block experiment has mixed effects that should accompany favorable replay results:

| Metric | Dev control to block | Rich control to block |
|---|---|---|
| Resolved | 25 to 24 / 70 | 12 to 13 / 39 |
| Stale-call share | 32.3% to 26.3% | 53.6% to 48.3% |
| Repeat starts | 1.5% to 3.1% | 2.3% to 4.4% |
| Runs of at least 3 identical calls | 28 to 50 | 27 to 37 |
| Runs of at least 5 identical calls | 23 to 36 | 20 to 31 |
| Tasks with at least half their calls bad | 15 to 6 | 19 to 17 |
| Malformed-call share | 4.0% to 1.6% | 2.5% to 5.5% |
| Mean agent seconds/task | 120 to 117 | 98 to 116 |

Both arms resolve 37/109. The dev difference interval is -6 to +3 tasks, and rich -2 to +5. The intervention reduces stale-call share and fragments long runs while increasing repeat starts and run counts. On rich it also increases time and malformed calls. Exact-copy runs and broader “stuck stretches” use different definitions and must not be conflated.

The “about two-task ceiling” in companion materials is not a counterfactual upper bound. Six of 23 loop failures were solved by the union of nine heterogeneous tested runs, but those runs cannot bound what a successful new intervention might solve. Conversely, escape can still lead to a wrong patch or timeout. Publish task IDs and the averaging rule, and call this an observed cross-run rescue estimate.

October 13 main-track results add a protocol/termination failure. The thinking-on budgetfit_contract agent repeatedly calls submit_patch while thoughts say “I'll stop.” The recorded analysis reports 21/179 task runs with 21-65 repeated submissions and 77-79% verbatim repetition of thoughts inside these stretches.

The review directly verified fastapi_15030 on Explorer: 80 tool calls, including 66 submissions; final calls return identical patch metadata while thoughts still say “I'll stop.” submit_patch is free and does not consume the counted-call budget. A session ends when the model answers without a tool call. The contract prompt omits the explicit text-only termination instruction present in the original budgetfit prompt. This strongly suggests a protocol explanation; an isolated prompt-line ablation would establish its effect.

Repeated submissions leave the working tree unchanged and waste time even when they do not change the graded patch. Real thoughts can coexist with tool loops. Distinguish search loops, malformed-call loops, termination loops, and repetitive thoughts.

The accompanying budgetfit/contract A/B ties on dev: 31.5 solved per run for both arms; rich is 15 versus 16. These bundles differ in several features and provide context, not a clean control for thinking-off search loops. The main-track audit reports that v31 did not change the 31B agent phase; nevertheless, freeze and record wheelhouse, serving, template, zip, sampling, and budget provenance for new comparisons.

See [the main-track analysis](../../Gemma4_kaggle_repo/agentic_workflow/README.md#result-2026-10-13-a-tie-keep-a).

### 9. Reference corrections and limits

- **Sethi:** Table 3 gives 8.1% dishonesty for EMPTY_RESULT under the deployment prompt. The 45.3% figure belongs to REDACTED_PERMISSION; HTTP_500 and TIMEOUT are 0%. Dishonesty is broader than fabricated answers. The supported lesson concerns interpretable versus ambiguous failure signals. [Primary text](https://arxiv.org/html/2609.14758).
- **CogRouter:** reward combines task success and format correctness; confidence redistributes credit among cognitive levels for a fixed action from a successful trajectory. It is not simply trained to maximize action confidence. “Would learn to skip exactly these states” is untested. [Section 3.4](https://arxiv.org/html/2602.12662).
- **DAgger:** a policy can incur quadratic error accumulation under the stated worst-case assumptions. This is not an established description of Gemma's training or fallback behavior; every mistake need not lead to an unseen state. [Ross et al.](https://arxiv.org/html/1011.0686).
- **Liu:** the 2.55-point shuffle loss supports modest sensitivity to action-outcome pairing, not zero tracking. Causal Transition outcome labels improve task success; generated Experience reduces repeats but lowers success. Do not conflate the interventions. [Sections 4-5](https://arxiv.org/html/2610.02769).
- **Pipis:** the 30% versus 4% student/teacher looping comparison and temperature as a partial remedy are supported. Preserve the limits of mapping synthetic reasoning loops to this SWE configuration. [Primary text](https://arxiv.org/html/2512.12895v2).
- **Sinha and Wan:** support harmful self-conditioning and conversational inertia, but do not establish the same internal mechanism for Gemma tool loops. [Sinha](https://arxiv.org/html/2509.09677v3), [Wan](https://arxiv.org/html/2602.03664v3).
- **Metareasoning:** use Russell and Wefald as the normative framework for benefit minus computational cost, not evidence that our replay measured that quantity. [Paper](https://www.sciencedirect.com/science/article/abs/pii/000437029190015C).

### 10. Latest evaluated ablation plan and measured cost basis

We have time for a focused causal study before the deadline. A comprehensive internal-mechanism study would require substantially more work. The recommended scope is feedback interpretation, controlled history reinforcement, and sampling/replication, with one conditional whole-task validation.

The authenticated Kaggle CLI check during the discussion confirmed November 12, 2026, 23:59 UTC (3:59 p.m. Pacific), about 30 days from October 13. [Competition timeline](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview/timeline). This is a dated check, not an assertion that the schedule cannot change.

Actual Explorer accounting, replacing the earlier README-based estimates:

| Completed experiment | Measured single-A100 allocation time | Job |
|---|---|---|
| Control, 109 tasks, four concurrent tasks | 1h19m14s | 11012947 |
| Always-block, same setting | 1h28m04s | 11012949 |
| T-open, 581 continuations plus scoring | 1h19m41s | 11012220 |
| Original Gate 1 core | 3h16m16s | 10991391 |
| Additional block scoring and sampled continuations | 50m55s | 10999083 |

The subsequent control/block resume jobs took 32/44 seconds and did no further evaluation. Compute is manageable; implementation, validation, queue delay, and writing are the larger constraints. At the queue check, the main track occupied the two L40S lanes, so the proposed package uses A100.

| Priority | Experiment and question | Proposed scope | Additional compute estimate |
|---|---|---|---|
| Required | Existing-data audit: which failures provide useful feedback? | Separate verified search misses, explicit errors, repeated nonempty results, and ambiguous empty successes; inspect directed teacher disagreement | CPU only |
| Required | Does meaningful feedback help beyond changing the result's surface form? | Approximately 120-160 curated states; four controlled renderings plus original/dedup anchors; three continuation samples per arm | 4-6 A100-hours |
| Required | Does adding repeated history itself increase copying? | Approximately 60 fixed histories; 1, 2, and 4 call/result copies, with a context-length control | 1-2 A100-hours |
| Required | Do findings survive recommended sampling and replication? | Control T=0.2, always-block T=0.2, and control T=1; two runs each over 109 tasks | 9-15 A100-hours |
| Conditional | Does the strongest feedback intervention help across whole tasks? | Winning feedback treatment versus its control; two runs each | 6-10 A100-hours |

The core totals 14-23 A100-hours; reserve 20-30 hours for startup, scoring corrections, and reruns. Including conditional validation, reserve approximately 30-45 hours. These are planning ranges, not allocation or queue guarantees.

Assumptions: existing model and serving environment, four evaluation lanes on one A100, the same task budgets, no Kaggle-style pacing, and valid pinned comparisons. Higher-temperature trajectories may take longer. Rescoring with at least the configured sampling support may increase cost. Queue delay is additional. With two A100 allocations, independent arms can overlap; GPU-hours are not elapsed wall time.

#### Feedback interpretation versus superficial cue disruption

Use four controlled conditions, plus the untouched original and existing dedup as diagnostic anchors:

| | Neutral text | Faithful failure interpretation |
|---|---|---|
| Repeated results retain matching surface forms | Placebo control | Semantic feedback |
| Repeated results receive varying arbitrary IDs | Surface-match disruption | Both changes |

Semantic feedback could say: “The search completed and found zero matches for this pattern in this file.” Initially avoid instructions such as “change approach,” which introduce another explanation.

Preserve call strings, match added token lengths and positions, and apply each rendering consistently to relevant historical results. Fixed versus varying arbitrary IDs form the surface-match factor. The semantic rendering should be stable across identical payloads; a changing repeat counter would alter both meaning and matching and belongs outside the initial factorial.

Select states by preceding history rather than requiring the original next action to have repeated. The primary subset should contain verified deterministic, read-only searches and unchanged relevant environment state. An ok with no stdout is not automatically a failed search. Spread states across tasks and include nonloop/search-miss controls where feasible.

Report exact copying, valid nonstale proposed calls, malformed calls, and approach change. If teacher scores are added, assess direction at decision tokens and disclose probability approximation; larger KL is not the definition of better feedback. Three continuations improve sampling estimates but do not triple independent task count. Use paired comparisons and task-cluster intervals.

Interpretations:

- IDs help: sensitivity to identical result surface forms contributes to persistence.
- Semantic interpretation helps beyond neutral text: understandable feedback contributes.
- Both interact: report the interaction rather than forcing a single cause.
- No reliable effect: limit the claim rather than adding an open-ended wording sweep.

This diagnoses behavior and available corrective supervision, not training improvement or induction heads.

#### Controlled history repetition

Manipulate repetition count within the same fixed history, using 1, 2, or 4 copies of an observed call/result block. Include a length-matched control and preserve task instructions and other evidence as far as possible. This addresses survivor selection in the original hazard curve. Keep it separate from the feedback factorial.

A positive dose effect strengthens evidence for context reinforcement in the tested histories. A null or heterogeneous effect leaves observed persistence intact but weakens a universal reinforcement-law claim.

#### Whole-task sampling and replication

Run three arms: original control at T=0.2, always-block at T=0.2, and original control at T=1; two runs each on the same 109 tasks. This replicates the template finding and adds the missing matched temperature comparison without a full template-by-temperature grid.

Keep zip, prompt, tools, task IDs, harness, serving stack, budgets, hardware, concurrency, and pacing fixed wherever applicable. Google recommends T=1, top-p 0.95, top-k 64. [Model card](https://ai.google.dev/gemma/docs/core/model_card_4).

Measure repeat fragmentation and re-entry, per-task and pooled stale rates, malformed calls, source edits, resolution, time, and tokens. Stratify loop types and identify what escapes lead to. Do not claim a solve-rate benefit from small noisy differences.

#### Conditional whole-task feedback validation

If semantic feedback has a clear, reproducible replay effect beyond placebo, evaluate it against its matched control over complete tasks. Budget runtime integration and smoke checks before full runs.

Measure executed-call usefulness, loop re-entry, time, and resolution, not just exact-copy suppression. A runtime trigger must depend on already observed history. If replay evidence is weak or integration threatens writing time, keep this conditional and limit the paper to one-step effects.

A failure-triggered forced-thought controller remains a useful alternative follow-up if that becomes the central question, but it is not an additional required arm of this deadline package.

#### Scope limits, statistical expectations, and schedule

Defer attention-level mechanism localization, multiple-model coverage, broad temperature or thinking-budget grids, every note wording, and a new OPSD training campaign. These diagnostics connect loops to rollout quality without making successful training a dependency of the paper.

The sample sizes target large behavioral effects like the observed 47-point escape gain. They are not expected to establish improvements of one or two resolved tasks. Report uncertainty and negative results; an inconclusive comparison is not equivalence.

| Dates, 2026 | Proposed work |
|---|---|
| October 14-16 | Definitions, existing-data audit, implementation, and validation |
| October 17-22 | Core runs and paired analysis |
| October 23-26 | Conditional whole-task validation if replay evidence warrants it |
| October 27 | Freeze experimental scope |
| October 27-November 5 | Write the paper and prepare reproducible figures/code |
| November 6-10 | Review, corrections, and early submission |

This is roughly one to two working weeks of experimental work with overlap, followed by a writing buffer. It assumes timely allocations; unexpected engineering or queue delays should reduce scope rather than consume the writing period.

The final proposed scope from the discussion is: test feedback meaning, controlled repetition, and temperature/replication; reserve one conditional whole-task validation. Keep the OPSD connection as a tested supervision hypothesis, with training effectiveness explicitly open. Implementation, commits, and Explorer/Kaggle submissions remain separate actions.

