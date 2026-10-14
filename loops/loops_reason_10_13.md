# loops/loops_reason_10_13.md: why the tool-call loops happen

<!-- written 2026-10-13. Numbers come from three places:
     - the closed-loop control and always-block arms, $R/paper/loops/report_p1_1012-1727 (loops.md, loops.json);
     - Gate 1's work dir, gate1_baseline_nothink_args_v30_zip (gate1.md sections 2-4);
     - ../gate1/README.md.
     The 95% intervals and the stuck-stretch counts come from read-only checks of copied outputs on 2026-10-12. Those scripts sit in a session scratchpad, not in this repo.
     The same content, with figures, is the "Why loops happen" section of loop_illustration.html. -->

**The question:** why does Gemma 4 31B with thinking off lock onto one tool call, and why do only some levers break the lock?

**The short answer:** five principles, one for each stage of the loop.

| Stage | Principle | Status |
|---|---|---|
| Lock-in | Each copy multiplies the odds of the next (self-reinforcement). | Established; measured here |
| Copying | The repeated tool result is the cue for the copy (induction). | Hypothesis |
| Hardening | Low temperature turns a preference into a certainty. | Established; measured here |
| Onset | The model repeats a call that can't teach it anything, under a strong prior. | Established principle; seen in our traces |
| No escape | The model thinks when unsure, and a loop is where it is most sure. | Measured here (T-open) |

**Terms** follow [loop_illustration.html](loop_illustration.html), "Terms used on this page":
- a *copy* repeats the call just before;
- a *stale repeat* is any call identical to an earlier one, with no successful edit in between;
- *lock-in after k* is P(the next call is one more copy | the last k calls were identical);
- *p(copy)* is the replay's probability of the copy at the run's sampling (T = 0.2, top-p 0.95, top-k 64).

## 1. Lock-in is a self-reinforcing process

*Status: established in text generation; measured here.*

Each copy multiplies the odds of the next one. These are the control arm's numbers over 109 tasks, dev and rich pooled (`loops.json`):

| The last k calls were identical | P(next call is a copy) | Cases | Odds | Odds multiplied by |
|---|---|---|---|---|
| k = 1 (any call) | 2.7% | 2,649 | 0.03 | – |
| k = 2 | 77.5% | 71 | 3.4 | ×120 |
| k = 3+ | 95.7% | 1,176 | 22.5 | ×6.5 |

- **The same law as repetition in text:**
  - A repeated phrase becomes more likely with each repetition (Holtzman et al., Fig. 4), and the effect then saturates (Xu et al.).
  - A Pólya urn behaves this way: every ball drawn adds another ball of its colour.
- **Run length.** If each call continues the loop with probability $q$, the rest of the run is geometric:

$$\mathbb{E}[\text{further copies}] = \frac{1}{1-q}$$

| Arm | q at k = 3+ | Expected further copies |
|---|---|---|
| Control | 0.957 (dev 0.957, rich 0.958) | ≈ 23 |
| Always-block | 0.890 (dev 0.844, rich 0.920) | ≈ 9 (dev 6, rich 12) |

- **Why loops end from outside.** In the control arm, 23 further calls is more than most loops have left before the budget warning. So 48 of the 55 runs of 3+ end at an outside event (the budget warning, a compaction or the end of the task), and only 7 end on their own.
- **Why the always-block template fragments loops.** Near $q = 1$ the mean is very sensitive to $q$, so a modest drop cuts runs sharply. That matches what the always-block template did over whole tasks:
  - the number of stuck stretches barely moved: 36 → 41 on dev, 36 → 29 on rich;
  - they got shorter: 18.9 → 11.0 calls on dev, 22.0 → 19.9 on rich.

## 2. The repeated result is the cue for the copy

*Status: hypothesis. It fits four observations but is untested.*

- **The mechanism.** Induction heads (Olsson et al.) find where the current context appeared before and copy what followed: A B … A → B.
- **How it applies to a tool loop:**
  - 91% of stale repeats got exactly the result their earlier copy got ([../gate1/README.md](../gate1/README.md)).
  - So the context ends with a result $r$ that appeared before, and the previous time, $r$ was followed by call $c$.
  - The repeated result doesn't signal "stop" to the model. It is the cue for "send $c$".
- **Four observations fit:**
  1. **The model's own p(copy) jumps between the 2nd and the 3rd copy.** At the first repeat, $r$ has never been followed by $c$, so the cue doesn't exist yet; from the third call on, it does. At T = 1 with top-p 0.95, p(copy) is 0.26 at 2nd copies and 0.73 at 3rd-or-later ones (gate1.md §2).
  2. **Changing the last result weakens the copy.** The note and the hindsight hint add text to the last tool result. At 3rd-or-later copies they take p(copy) from 0.94 to 0.49 and to 0.45 (gate1.md §2).
  3. **The empty thought puts four tokens between the result and the call.** Those tokens never appear in the history, because the template strips past empty thoughts, so the result-to-call adjacency is broken. p(copy) falls from 0.95 to 0.50 on the 40 third-or-later copies where it applies (gate1.md §3).
  4. **The always-block run on `fastapi_12942` alternates two failing greps:** A ×9, B, A ×5, B, and so on. After B's empty result, the history already holds "B's result, then A" (calls 7 and 8). The cycle is itself a copied pattern.
- **The statistical version.** In-context learning behaves like Bayesian inference over a hidden concept (Xie et al.).
  - Read the agent's own past calls as examples of a hidden "what this agent does here".
  - Each copy is more evidence for "it sends $c$", so the posterior odds multiply, as in §1.
  - Sinha et al.'s self-conditioning and Wan et al.'s conversational inertia are this effect seen in agents.
- **Tests, for later** (one replay context each):
  - Make each repeated result differ in a way that carries no meaning, such as a request id. If lock-in falls sharply, the exact match matters.
  - Separately, add a counter ("this command has returned nothing 3 times"). That adds information as well as breaking the match, so it tests something different.

## 3. Low temperature turns a preference into a certainty

*Status: established; measured here.*

Sampling at temperature $T$ draws from the renormalised

$$p_T(x) \propto p(x)^{1/T}, \qquad \frac{p_T(a)}{p_T(b)} = \left(\frac{p(a)}{p(b)}\right)^{1/T}$$

At T = 0.2 the odds between two calls are raised to the fifth power, so 2 : 1 becomes 32 : 1. For a whole call, the sharpening applies at every token. The replay separates this from the model's own preference at the same states (gate1.md §2):

| State | Model's own p(call), T = 1, top-p 0.95 | At the run's T = 0.2 |
|---|---|---|
| 2nd copy | 0.26 | 0.77 |
| 3rd-or-later copy | 0.73 | 0.94 |
| Routine call, for reference | 0.52 | 0.78 |

- **Lock-in has two factors:**
  - in-context reinforcement raises the model's own preference for the copy (0.26 → 0.73 from 2nd to 3rd-or-later copies);
  - low temperature makes the most likely call nearly certain. At the 2nd copy, most of the 78% lock-in comes from the sampler.
- **The classic degeneration result.** Near-greedy decoding of a self-reinforcing process settles into a cycle (Holtzman et al.). Pipis et al. call temperature "a stopgap"; in these terms, it removes the second factor and leaves the first.
- **The limit of the fix.** At 3rd-or-later copies the model's own preference is already 0.73, so T = 1 would weaken lock-in, not remove it.
  - The only closed-loop data at T = 1 is `agentic_v2` on v28, a different prompt. It still locked in at 73%.
  - An `off_args` run at Google's prescribed T = 1.0 is still missing ([literature.md](literature.md) §6).

## 4. Loops start with a call that can't teach anything

*Status: established principle; seen in our traces.*

- **Why a repeat is worthless, and why the model makes it anyway:**
  - In a deterministic environment, an identical read-only call returns what it returned before, so its information gain is zero.
  - A rational searcher never makes one, and our stale repeat is exactly this set of calls.
  - Next-token training has no term for information gain. The model picks what is likely given the context.
- **The onset in our traces** ([../gate1/README.md](../gate1/README.md), "How a loop starts"):
  1. The model searches for something it strongly believes exists, such as a plausible function name that isn't in the file (`get_openapi_route`).
  2. The empty result contradicts that belief but carries almost no text to condition on. In `off_args`'s first v30 run, 270 of the 580 repeated `run_command` calls got nothing back.
  3. After an empty search, the next call is a variant 78% of the time and a copy 1%. When the variants also fail, the model returns to its first and most likely query: 114 of 131 chains of bad calls start by re-sending a call from 2 or more calls back.
- **The disconnect, measured directly.** Liu et al. find that shuffling an agent's past actions costs it only 2.55 points. It isn't tracking which call produced which result.
- **Why it happens once the agent is off track (learning theory).**
  - A policy trained to imitate experts errs at rate $\varepsilon$ per step, and each error takes it to states the experts never visited, where it has nothing to imitate.
  - Over a horizon of $H$ steps its errors therefore grow as $O(\varepsilon H^2)$ rather than $O(\varepsilon H)$.
  - This is the analysis behind DAgger (Ross, Gordon & Bagnell). A made-up function name is such a state, and copying its own recent behaviour is the policy's fallback there.
  - Pipis et al. find the same in reasoning models: when progress is hard to learn, an easy repeating action keeps its probability, and distilled students loop more than their teachers (30% vs 4%).

## 5. The model thinks when unsure, and a loop is where it is most sure

*Status: measured here (T-open).*

- **The right rule.** Metareasoning (Russell & Wefald) says to deliberate when the expected gain from thinking exceeds its cost, the value of computation:

$$\mathrm{VOC}(h) = \mathbb{E}\big[U(a_{\text{after thought}})\big] - \mathbb{E}\big[U(a_{\text{now}})\big] - c_{\text{thought}}$$

- **The rule the model follows instead.** It behaves as if it thinks when uncertain. Free to think (T-think), it writes a real thought at 72% of 2nd copies and 23% of 3rd-or-later copies, while p(copy) rises from 0.77 to 0.94 (gate1.md §2 and §4).
- **Why that is backwards here.** At a loop state the current action is worth nothing (it can't teach anything), and a thought improves it 58–69% of the time. So VOC is highest exactly where uncertainty is lowest.
- **T-open measures this.** On the looping states (213 in 44 tasks):
  - where T-think skipped the thought, a forced thought escapes 58% of the time (95% CI 46–69), against 11% (5–16);
  - where T-think chose to think, the two are the same: 63% and 62%.
  - So thinking helps as much where the model skipped it as where it picked it, and the model's trigger carries no information about the benefit.
- **Any trigger built on confidence or entropy has this blind spot.** HRBench calls the failures "confident wrong answers". A router trained to maximise action confidence (CogRouter) would learn to skip exactly these states.
- **What a useful trigger reads.** It needs a signal tied to the benefit, such as "my last call returned the same result as before".
  - A harness sees that easily, so a controller outside the model can decide when to think.
  - A Kaggle agent can't force a thought at runtime (it has no reply prefix and no callbacks), so there the trigger would have to live in the weights.

## 6. What is specific to tool calls and code

- **The environment echoes.** Each cycle adds two identical pieces to the context, the call and its identical result, and nothing outside disturbs them. In free text, only the model's own words repeat.
- **Calls are short, rigid strings.** Copying a whole call is cheap and always gives valid syntax.
- **The worst results are the emptiest.** A grep that exits 1 with no output, or an `ok` with empty output, gives the copy cue a clean match and gives the model almost nothing to reason from. Sethi et al. found that an `ok` status with an empty payload produced 45% fabricated answers, against 0% with an explicit error.
- **Contexts grow.** The repeated block grows while the task statement recedes. Compaction cuts both ways: it deletes the copies (a natural dedup) and the evidence of failure. 44% of chains of bad calls re-send a call that compaction had hidden.

## 7. One model for all of it

This is a stylised model, not a fitted one:

$$P(a_t = c \mid h_t) \;\propto\; \exp\!\Big(\tfrac{1}{T}\big[\, s(c) + \lambda\, n_t(c) + \mu\, m_t(c) \,\big]\Big)$$

- $s(c)$: the model's prior for call $c$, given the task.
- $n_t(c)$: copies of $c$ already in the context.
- $m_t(c)$: 1 when the last result matches what came before $c$ the previous time, otherwise 0.
- $\lambda$, $\mu$: how strongly the copies and the cue pull.
- $T$: the sampling temperature.

Each lever we tested changes one part of it. The numbers are one step ahead, at 3rd-or-later copies (gate1.md §2–§4):

| Lever | What it changes | p(copy) |
|---|---|---|
| Copies deleted (dedup) | fewer copies, $n$ | 0.94 → 0.14 |
| Empty thought | breaks the result-to-call cue, $m$ | 0.95 → 0.50 (40 states) |
| Note; hindsight hint | change the last result, $m$, and restate it | 0.94 → 0.49; 0.45 |
| T = 1 instead of 0.2 | softens every term fivefold | 0.94 → 0.73 |
| Forced thought (T-open) | rewrites the context the terms are computed on | 0.94 → 0.12 (escape 70%) |

- **What the model has no term for:** information gain, and which call produced which result. That gap is where new information would act: an informative empty result, file locations, or a restart that keeps the earlier diff.
- **The model reproduces the shape of the data:**
  - the large jump at the first repeat (§1);
  - run length's sensitivity to $q$;
  - fragmentation under the empty thought;
  - why only deleting the copies or forcing a thought brings p(copy) near zero.

## 8. What is established and what is ours

- **Established elsewhere** (text generation, or other agents): self-reinforcement and its saturation; degeneration at low temperature; compounding errors in imitation; confidence-based triggers failing on confident mistakes.
- **Measured in our data:** the lock-in curve; the split between in-context reinforcement and temperature; the thinking trigger's failure (T-open); the onset at calls that can't teach anything.
- **Fits our data, untested:**
  - the repeated result as the copy cue (§2's tests would check it);
  - the A/B cycle as a copied pattern;
  - conversational inertia behind the drift of the self-written empty thought (the compaction test in loop_illustration.html, open question 2).

## References

**New here** (not in [literature.md](literature.md)):

- Russell, S. & Wefald, E. *Principles of metareasoning.* Artificial Intelligence 49 (1991), 361–395. DOI 10.1016/0004-3702(91)90015-C.
  🔗 [**Access Options**](https://onesearch.library.northeastern.edu/openurl/01NEU_INST/01NEU_INST:NUL?genre=article&aulast=Russell&issn=0004-3702&title=Artificial%20Intelligence&atitle=Principles%20of%20metareasoning&volume=49&issue=1-3&spage=361&epage=395&date=1991-05-01&rft_id=info:doi/10.1016%2F0004-3702(91)90015-C&rft_id=info:pmid/&sid=LibKey)
- Ross, S., Gordon, G. J. & Bagnell, J. A. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS 2011. Full text not available through your institution. Open copy: https://arxiv.org/abs/1011.0686
- Xie, S. M., Raghunathan, A., Liang, P. & Ma, T. *An Explanation of In-context Learning as Implicit Bayesian Inference.* ICLR 2022. Full text not available through your institution. Open copy: https://arxiv.org/abs/2111.02080

**Also cited** (details and links in [literature.md](literature.md) §4):
- Holtzman et al. 2020 (1904.09751);
- Xu et al. 2022 (2206.02369);
- Olsson et al. 2022 (2209.11895);
- Pipis et al. 2026 (2512.12895);
- Sinha et al. 2026 (2509.09677);
- Wan et al. 2026 (2602.03664);
- Liu et al. 2026 (2610.02769);
- Sethi et al. 2026 (2609.14758);
- HRBench, Ning et al. 2026 (2605.28398);
- CogRouter, Yang et al. 2026 (2602.12662).


## Review Comments,  from GPT 6.1 Sol.

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

