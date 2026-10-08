# OPSD: worth it, what it costs, what it gives the main track

<!-- research session 2026-10-07/08; numbers as of 2026-10-08 -->

This file answers three questions about the [primary direction](README.md#primary-direction-opsd-for-a-thinking-off-agent):
1. Is on-policy self-distillation (OPSD) worth it as this track's direction?
2. What schedule and compute does it need, and what should we ask NEU faculty for?
3. Does the work also help the main track?

The evidence:
- the OPSD paper and its code;
- the main-track repo's measurements;
- Explorer itself: `sinfo -a`, `sacctmgr`, `squeue`, our `sacct`, and two read-only passes over our call logs;
- the Kaggle CLI;
- AICR's and NEU RC's own pages;
- about 20 preprints from 2026.

See [Sources](#sources).

## Bottom line

- **Keep the direction, a thinking-off agent trained by self-distillation, but change the teacher.**
  - **The README's kind of teacher, the same model given privileged task information, has already failed this test.** PSP ([2609.29051](https://arxiv.org/abs/2609.29051), 2026-09-24) ran OPSD on thinking-off agents on SWE-bench Verified. It fell *below the untrained base*: 3.56 → 2.40 for Qwen3-4B and 4.72 → 2.88 for Qwen3-8B, while GRPO and PSP gained.
    - PSP's privileged information was an analyzer's per-task guidance, not the gold patch.
    - The authors' diagnosis, that OPSD "teaches the student to act with confidence but without the information behind it", applies even more to a gold patch, which tells the teacher exactly where to edit.
  - **Make the main teacher the model's own thinking.** For each student call, the same weights with thinking on write a thought, then score the student's tool call after that thought. The student learns what the thinking-on agent would do, from information it already has, so PSP's failure mode doesn't apply. The closest prior work (SeOPD, ThinkOPD, Masked Self-Distillation) is single-turn math or code. We found nothing on multi-turn agents under a clock.
  - **Keep two more teachers as arms of the study:**
    - the gold-patch teacher, which tests whether PSP's failure holds at 31B;
    - a hindsight teacher that sees a mechanical diagnosis of each repeated or malformed call (the HERO/SDPO idea).
- **Five other changes** ([1.4](#14-what-to-change)):
  - a go/no-go on **Oct 13**, on a diagnostic that needs no training;
  - **full-vocabulary forward KL** computed on the H200;
  - **packed trajectories** (8.8× fewer tokens, measured);
  - **masked thought tokens**;
  - **two cheap baselines** that the judges will ask about.
- **Worth it: yes, as this reframed study.** Plain gold-patch OPSD on its own is not worth it.
  - The downside is bounded: the fallback paper (the thinking-budget study plus the diagnostic) reuses the same runs.
  - The question "which self-teacher fixes a thinking-off SWE agent?" is open, relevant, and answerable in 35 days. Our odds are in [1.6](#16-verdict).
- **Compute: about 180 GPU-hours** (A100 ~33, H200 ~70, L40S ~75). That fits Explorer's current limits.
  - What binds is latency: 2 L40S eval slots shared with the main track, an H200 queue of 96 jobs, and 8-h job limits.
  - The real upgrade is **AICR**: 248 B200s, 24-h jobs, up to 32 GPUs per user. It's on a separate cluster, and only a faculty PI can apply. Multi-GPU jobs and B200s aren't *needed*, but AICR would remove every bottleneck at once.
- **The faculty conversation is also about compliance.** RC policy: "Do not use the Explorer Cluster for any purpose other than for Northeastern University-sponsored research or course work", and "students must have a faculty sponsor". Both tracks currently run on course accounts. The ranked asks: [2.7](#27-option-plan-for-the-faculty-negotiation).
- **Main track:** the indirect gains are likely; a direct score gain from the adapter is possible but uncertain ([3](#3-does-the-paper-work-help-the-main-track)).
- **Blocker found during this session:** `/scratch/zha.j/gemma4/models/gemma-4-31b-it-qat-w4a16-ct` has been empty since 2026-10-07 13:19 EDT (0 bytes). Data, venvs and adapters are intact. Every GPU job of either track fails until `sbatch slurm/fetch_model.slurm` re-downloads it (CPU, ≤ 2 h).

## 1. Is OPSD worth it?

### 1.1 What the OPSD paper shows, and what it doesn't

Read from [Reference/opsd.pdf](Reference/opsd.pdf) (v3, 2026-03-20) and the [released code](https://github.com/siyan-zhao/OPSD).

| | OPSD paper | Us |
|---|---|---|
| Model | Qwen3 1.7B / 4B / 8B, instruct | Gemma 4 31B, int4 QAT, served by vLLM |
| Task | single-turn math; 1,024-token student rollouts | multi-turn tool use: ~56 calls per task, 32k context with compaction, a 4-min clock |
| Teacher sees | a reference chain-of-thought solution | (planned) the gold patch, or a location-only hint |
| Rollouts | 1 per problem per step | ~4 per task per round |
| Training | LoRA r64, lr 5e-6, batch 32, 100 steps, 8× A100/H100 | LoRA r16, 1 H200, round-based |
| Thinking at **evaluation** | **on** (Table 8: up to 38,912 tokens) | **off** |
| Gain over base (avg of 3 benchmarks) | +6.3 (1.7B), +2.4 (4B), +3.0 (8B) | unknown |

- **The thinking-off claim is ours, not OPSD's.**
  - OPSD samples the student with thinking off only during training, then evaluates it with thinking on. Its best pairing, a thinking-off student with a thinking-on teacher, is a choice of training signal.
  - Nothing in it shows a *deployed* thinking-off student matching a thinking-on model.
  - The README line "OPSD's best setting in its paper was exactly this pairing" is true for training only.
- **Forward KL is what worked.**
  - Table 3, Qwen3-1.7B on AIME25 (base 36.7): forward KL reaches 43.9 at step 50. Reverse KL gives 37.5, then 35.0 at step 100. JSD gives 36.9, then 39.0.
  - The sampled-token objective (Thinking Machines' on-policy distillation) is a policy gradient with advantage log p_T − log p_S on the sampled token. That is a one-sample estimate of the **reverse** KL.
  - Table 4's 82.1 vs 84.1 (pass@8, Qwen3-4B) compares it with full-vocabulary forward KL and reports no base. So it doesn't show the cheap variant beating the base. The code's flag for it (`use_thinking_machines_loss`) has no clipping and warns that it "could be unstable".
  - Our README's line "per-token forward KL … start with the sampled-token variant" mixes two different objectives.
- **Clipping is under-specified.**
  - The code clamps each vocabulary entry's contribution before the vocabulary sum: `jsd = jsd.clamp(max=token_clip)`.
  - The paper says τ was not tuned. The example scripts use 1e-7 (8B) and 1e-6 (4B, 1.7B); the flag defaults to 0.05.
  - Gemma's vocabulary is 262k entries against Qwen3's 152k. Sweep τ, and check on a toy pair of distributions that the clipped loss still moves the student toward the teacher.
- **How the code builds a thinking-on teacher.** It renders the teacher prompt with `enable_thinking=True` (the default), then appends the student's non-thinking tokens directly.
  - At the start of every reply the teacher therefore expects a thought block that the student doesn't write.
  - In Gemma 4 that block is `<|channel>thought … <channel|>`. Our thinking-off student writes an empty 4-token one on about 40% of replies (ids 100, 45518, 107, 101).
  - Under forward KL the student gets pushed toward opening thought blocks. "On Repulsive and Attractive Teachers" ([2609.21561](https://arxiv.org/abs/2609.21561)) reports self-distillation tipping non-thinking models into their latent thinking mode. Mask these positions, and monitor thought-token rates during training.
  - The code also has `--reason_first`: the teacher analyses the reference first and scores the student after its analysis.
- **Limits the authors state:**
  - experiments only up to 8B; larger scales are "an open question";
  - the teacher can't help on problems beyond the model's comprehension, even with the reference. PSP's 4B/8B students solve only 3–5% of SWE-bench Verified, which is that regime. Ours solves 27% of our dev tasks with thinking off, so PSP's result may not transfer fully. We would be the test of that.

### 1.2 For

| Rubric criterion | What the reframed study gives us |
|---|---|
| Novelty | A thinking self-teacher for a multi-turn SWE agent that is deployed with thinking off and works under a clock, at 31B. Not found in the 2026 literature ([1.5](#15-novelty-check-2026-literature)). It also tests PSP's negative OPSD result at 31B. |
| Quality (generality) | Nothing in it is specific to this harness: any hybrid-thinking model (Gemma 4, Qwen3, DeepSeek). The own-thinking teacher needs no reference patch, so it applies to any task with an environment. Train on rich, evaluate on fastapi/requests (cross-repo), then on Kaggle's hidden private repos (out of distribution). |
| Relevance | Thinking takes 65–71% of our agent's time. A thinking-off agent that keeps the thinking-on agent's call hygiene costs less and finishes more within a clock. This is the host's first listed topic, "Tuning & Optimization". |
| Verifiability | Our token proxy logs the exact prompt and completion ids of every call. The local scorer replica passes 111 of 129 gold patches. Paired tests, trace metrics, scripts we can release. |
| Clarity | One question, one mechanism, one figure of where each teacher disagrees with the student. |

- **The target is measurable.**
  - With thinking on, 9% of calls repeat an earlier identical call and 2.2% are malformed. With thinking off, about 50% repeat and 24–28% are malformed (the agentic_workflow report and the Oct 5 audit count slightly differently).
  - Gaps that large dwarf run-to-run noise, while the resolve rate moves by ±2.2 tasks per run.
- **It fixes RFT's data problem.**
  - Round 0 trained on 52k loss tokens from 11 tasks.
  - One round of 236 rollouts gives about 0.7M loss tokens, failures included. The own-thinking and hindsight teachers need no reference at all.
- **Most of the tooling exists:**
  - the trainer with chunked, soft-capped logits;
  - the token proxy;
  - the scorer replica;
  - the eval driver;
  - `diagnose_run.py`'s call-hygiene table, which also yields the hindsight teacher's diagnoses.
- **It produces a main-track submission:** a rank-16 adapter with `thinking_budget: 0`.
- **A negative result is still a paper.** "Which privileged context helps a thinking-off SWE agent, and where does each teacher disagree with it" stands on the diagnostic alone, together with the thinking-budget runs (option A).

### 1.3 Against

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Privileged-information OPSD fails on thinking-off SWE agents** | PSP: OPSD below base on SWE-bench Verified with thinking off, with per-task guidance as the privileged information. A gold patch gives the student an even bigger information gap. HERO ([2606.11559](https://arxiv.org/abs/2606.11559)) also finds naive multi-turn OPSD degrades. Its explanation: an outcome-level reference doesn't match the state the student is in. | The own-thinking teacher is locally aligned by construction. The gold patch becomes a comparison arm, not the method. |
| **Thinking→non-thinking may compress exploration** | Self-distillation toward an "attractive" teacher "promotes shorter, more confident responses" (2609.21561). | Watch the exploration metrics: reads before the first edit, pytest runs. Keep the learning rate small; try 2 rounds rather than many steps. |
| **Mode switch** | The student may start writing thought blocks, or long visible reasoning. | Mask thought-channel tokens; track output tokens per call. |
| **Top-k teacher targets miss tool-call decisions** | "When Top-K Misses the Decision" ([2607.07050](https://arxiv.org/abs/2607.07050)): retained top-K mass looked like 1.000 yet omitted the tool-call entry token, which caused over-calling. | Full-vocabulary teacher logits on the H200 for training; top-k from vLLM only for the diagnostic. |
| **The cheap loss is the weak one** | See 1.1: sampled-token = reverse KL. | Forward KL. |
| **Compute and latency** | Per-call training spends 99% of its compute re-encoding prompts. The own-thinking teacher must score each call separately. vLLM and the trainer run as separate Slurm jobs, so "on-policy" means once per round. | Pack the student side; score the teacher per call only where it differs; 2–3 rounds. AICR would allow a true online loop. |
| **Statistical power** | One run varies by ±2.2 tasks. A +4-task gain needs 2–3 seeds per arm for p < 0.05 on paired tests. The dev set is 60/70 fastapi, which is async-heavy, unlike the hidden repos. | Trace metrics as the primary result, resolve rate secondary; ≥ 2 seeds; Kaggle's hidden set as the out-of-distribution check. |
| **Baseline burden** | If a prompt fix or plain SFT gets the same, judges will ask why on-policy. | The two baselines in change 5. |
| **A crowded field that moves fast** | About 30 OPSD-family preprints in 2026, six of them relevant to us from the last three weeks. A single-turn paper could add an agent experiment before Nov 12. | Post an arXiv preprint as soon as round 1 has a result; the track allows it ([745872](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745872)). Cite precisely. |
| **Timeline** | 35 days, shared with the main track. Per-call teacher scoring, packing and the round loop add about a week to the README's "about a week". | Gates on Oct 13 and Oct 22; writing starts Oct 26, in parallel. |
| **The host can change the harness** | v28 invalidated v25's rollouts overnight. | Every run records its wheelhouse version; `build_rft_data.py` refuses to mix them. |
| **Disclosure** | The paper is public after Nov 12; the main track runs to Dec 2. | Accept it: few teams can rebuild the pipeline in 20 days. |

### 1.4 What to change

1. **Diagnostic first; go/no-go on Oct 13.**
   - Score the 70 existing thinking-off dev trajectories (`baseline_nothink_v28_zip`) with every teacher. No new rollouts or training.
   - Teachers:

     | Teacher | Its privileged context | Packs per segment? |
     |---|---|---|
     | **T-think** (the proposed method) | its own thought for this call: the same weights with thinking on, budget ~512 tokens, generated by vLLM | no: one sequence per call |
     | **T-hint** (hindsight) | for a call that repeats an earlier one or has bad argument names, a mechanical diagnosis from `diagnose_run.py`'s rules ("this repeats call #k, which returned …"); nothing for clean calls | flagged calls only |
     | T-patch (PSP's test, with the gold patch) | the gold patch, thinking-off template (the README's design) | yes |
     | T-patch-on | the gold patch, thinking-on template (OPSD's default), thought tokens masked | yes |
     | T-loc | files and functions from the patch only | yes |
     | C-wrong | another task's patch (control, after [2608.18271](https://arxiv.org/abs/2608.18271)) | yes |
     | C-none | nothing; thinking-on template only (the template effect alone) | yes |

   - Measure, grouped by token class:
     - the exact log-ratio log p_T − log p_S on each sampled token;
     - top-k forward KL.
   - The token classes: argument names of malformed calls; tokens of calls that repeat an earlier identical call; file paths; `edit_file` code; the tool-name token; the rest.
   - **Gate:** some teacher must put clearly more KL on the failure tokens than on the rest. If none does, start the fallback paper on Oct 14, with this figure in it.
   - Cost: about 10 A100-hours. The per-call T-think scoring dominates, since vLLM re-prefills any request that asks for prompt logprobs ([V1 guide](https://docs.vllm.ai/usage/v1_guide.html)).
2. **Forward KL with full-vocabulary teacher logits, on the H200.**
   - Run the teacher under PEFT's `disable_adapter()`, as the README planned. Chunk both log-softmaxes over 1,024 positions, as `train_lora.py` already does for the loss.
   - Top-k targets from vLLM serve the diagnostic only, because of tool-call drift (1.3).
   - Keep OPSD's clipping, with a τ sweep and a toy-distribution unit test.
   - For T-think, the teacher forward runs per call: the call's prompt, the teacher's thought, then the student's tokens. That costs ~370k tokens per trajectory, without a grad.
   - Reusing a cache cropped at each call's prompt would cut this to ~50k. Gemma 4 mixes sliding-window and global layers, though, so check that HF's cache supports cropping before relying on it.
3. **Pack the student side.** Measured on our logs (read-only Explorer jobs, 2026-10-08):

   | Run | Calls / task (median) | Tokens per task, per-call training | Packed | Saving | Loss tokens / task |
   |---|---|---|---|---|---|
   | thinking off (`baseline_nothink_v28_zip`) | 56 | 449k | 30k | **8.8×** | 3.0k |
   | thinking on (`base_v28_zip`) | 24 | 214k | 32k | 6.9× | 6.4k |

   - With thinking off, 48% of consecutive calls extend the previous prompt exactly. Another 35% differ only by the 4-token empty thought block, which the next prompt drops. 9% restart after compaction. That leaves about 8% that need a new segment.
   - So plain causal packing works, with no tree-shaped mask: one sequence per compaction segment (median 3 per task), loss on each call's completion minus the thought tokens.
   - Thinking on is different: 92% of its transitions re-render the completion. That is why RFT's notes needed a tree mask.
4. **Mask thought-channel tokens** in every teacher.
5. **Two cheap baselines**, which are also the natural comparisons:
   - the prompt-fixed thinking-off agent (`agentic_v1`), which the main track needs to run anyway;
   - off-policy self-distillation: SFT on thinking-on successes re-rendered with the thoughts removed, i.e. context distillation ([post_train option 1](https://github.com/usp787/Gemma4_kaggle_repo/blob/main/post_train/README.md#options-ranked)). The difference between it and T-think is exactly "on-policy": Masked Self-Distillation found SFT generalizes worse.

### 1.5 Novelty check: 2026 literature

Read on arXiv on 2026-10-08, abstracts only unless noted. "Overlap" means overlap with the reframed study.

| Paper | Date | What it shows | Overlap |
|---|---|---|---|
| OPSD [2601.18734](https://arxiv.org/abs/2601.18734) | Jan (v3 Mar) | gold-solution self-teacher; single-turn math; ≤ 8B; evaluated with thinking on | the method we start from |
| SDPO [2601.20802](https://arxiv.org/abs/2601.20802) | Jan 28 | the model conditioned on feedback (e.g. runtime errors) as its own teacher; code and tool use | medium: our T-hint |
| Hindsight Hint Distillation [2605.11556](https://arxiv.org/abs/2605.11556) | May 12 | hints from failed rollouts scaffold on-policy rollouts; distilled without hints; SWE-bench Verified +8 points against ~2 for baselines | medium: SWE, hints |
| Patches-to-Trajectories [2605.21996](https://arxiv.org/abs/2605.21996) | May 21 | the reference patch as privileged information for curating SFT trajectories; SWE-bench Verified up to +10.8 | low: SFT, not on-policy |
| HERO [2606.11559](https://arxiv.org/abs/2606.11559) | Jun 10 | naive multi-turn OPSD degrades; next observations as locally aligned feedback; TauBench, WebShop | medium: our T-hint, not SWE |
| Masked Self-Distillation [2607.22629](https://arxiv.org/abs/2607.22629) | Jun 18 (v3 Sep 30) | copies of one model internalize the chain of thought; on-policy beats SFT; math and graph colouring; Qwen3 4B/8B | **high in idea**, single-turn |
| Rethinking OPSD for Thinking Models [2607.05184](https://arxiv.org/abs/2607.05184) | Jul 6 | privileged context hurts thinking models (up to −17% relative) but can help instruction models | supports a thinking-off student |
| Tool-Call Drift [2607.07050](https://arxiv.org/abs/2607.07050) | Jul 8 (v6 Sep 5) | top-K teacher logits omit decision-critical tool-call tokens | design: full vocabulary |
| State-Matched Routing [2608.05219](https://arxiv.org/abs/2608.05219) | Aug 5 | a reference trajectory misguides states the student reached on its own; ALFWorld, WebShop | medium: why T-patch may fail |
| AgentOPSD [2608.05987](https://arxiv.org/abs/2608.05987) | Aug 6 | turn-level credit from teacher-student log-prob gaps; ALFWorld, WebShop, Search-QA; Qwen2.5 3B/7B | low |
| Rethinking Privileged Information [2608.18271](https://arxiv.org/abs/2608.18271) | Aug 18 | the correct reference isn't consistently better; another problem's solution can win | the C-wrong control |
| Repulsive and Attractive Teachers [2609.21561](https://arxiv.org/abs/2609.21561) | Sep 18 | attraction shortens and sharpens; repulsion can switch a model into thinking mode | design: masks, monitoring |
| **PSP** [2609.29051](https://arxiv.org/abs/2609.29051) | Sep 24 | OPSD on thinking-off agents (AppWorld, SWE-bench Verified; Qwen3-4B/8B) falls below base. Its privileged information is an analyzer's per-task guidance; putting it in the sampler with GRPO instead wins (results table read) | **high: tests the README's kind of teacher** |
| **SeOPD** [2609.33181](https://arxiv.org/abs/2609.33181) | Sep 27 (v2 Oct 7) | the model's own thinking-mode chain of thought as privileged information for its non-thinking responses | **high in idea**; no agents or tools mentioned |
| **ThinkOPD** [2609.37044](https://arxiv.org/abs/2609.37044) | Sep 29 | a think-mode teacher for a non-thinking student, deployed without thinking; math and code | **high in idea**, single-turn |
| PivotOPD [2609.40285](https://arxiv.org/abs/2609.40285) | Sep 30 | OPD at pivotal mistakes, using a teacher's gold and recovery actions; SWE-bench Verified +3.2 (Nemotron-3.5) | medium: multi-turn SWE OPD, external teacher |

Seen in search results but not opened: Open-SWE-Traces (2606.16038), SOD (2605.07725), Thinking Collapse in OPSD (2607.10805), Privileged Solutions or Context-Induced Teacher Behavior? (2608.09228), DualOPSD (2608.26019), What Does Privileged Information Add to OPSD? (2609.20612), DART-SD (2608.18524), Know When to Stop, Where to Restart (2609.14636). Read their abstracts before writing the related work.

**Novelty verdict**
- **Taken:**
  - gold-patch OPSD for agents (PSP's baseline, which failed);
  - thinking→non-thinking self-distillation for single-turn reasoning (SeOPD, ThinkOPD, Masked Self-Distillation);
  - observation feedback as privileged context outside SWE (HERO, SDPO);
  - on-policy distillation from an external teacher on SWE (PivotOPD, Orchard).
- **Open, as far as we found:** a *multi-turn SWE agent*, deployed thinking-off, taught by *its own thinking* (and by hindsight), *under a wall clock*, at 31B. Plus a 31B test of PSP's negative result, and a token-level account of where each teacher disagrees (malformed arguments, verbatim repeats).
- **We must cite and set ourselves apart from:** PSP, SeOPD, ThinkOPD, Masked Self-Distillation, HERO, SDPO, PivotOPD, Rethinking Privileged Information, Rethinking OPSD for Thinking Models, Tool-Call Drift, and OPSD itself.

### 1.6 Verdict

**Pursue it, reframed:** "Which self-teacher makes a thinking-off SWE agent act like it thinks?" Gold-patch OPSD becomes one arm, not the method.

Our odds (judgement, not measurement):

| Milestone | Chance |
|---|---|
| Gate 1: some teacher's KL concentrates on the failure tokens. T-think is the favourite: thinking on cuts malformed calls about 10×. | ~65% |
| Gate 2: the trained student's repeat and malformed shares fall clearly (cumulative) | ~40% |
| Resolve rate +4/70 or more over the thinking-off base, 2 seeds | ~30% |
| T-think beats both cheap baselines on the resolve rate | ~20% |
| Any prize, this paper | ~5–10% |
| Any prize, the fallback paper alone | ~3–5% |

- **Why it beats the alternatives:**
  - Option A alone is measurement only.
  - A topic from scratch can't reach a result in 35 days.
  - The reframing costs about one extra week of engineering, but the gates keep the fallback intact.
- **What would make us drop it:**
  - Gate 1 fails on Oct 13;
  - or, by Oct 22, the trained student hasn't moved the repeat and malformed shares.

## 2. Schedule and resources

### 2.1 Host schedule (checked 2026-10-08 with the Kaggle CLI)

| Date | Event |
|---|---|
| **Thu 2026-11-12 23:59 UTC** (15:59 PST, 18:59 EST) | Paper deadline; also the last day to join or merge teams. 158 teams have joined. |
| Wed 2026-11-25 | Main track: entry and team-merger deadline |
| Wed 2026-12-02 23:59 UTC | Main track: final submissions. 1,989 teams; we are rank 328 at 0.13; the top is 0.24, then 0.18. |
| Dec 2026 | NeurIPS expo for the winning papers |

- US daylight saving time ends Nov 1, so the deadline is 15:59 Pacific time.
- The paper track has no milestone before the deadline. A tie goes to the paper "entered first", so submit a complete version early (Nov 8) and keep editing.
- **Maintenance windows inside the sprint:**
  - AICR: the third Tuesday of each month, 5 a.m.–5 p.m. That makes Oct 20 (our calculation from the stated rule).
  - Explorer: per RC, the first Tuesday. That makes Nov 3 (unverified).

### 2.2 Our schedule

| Dates | Work | Gate |
|---|---|---|
| Thu Oct 8 | Re-fetch the model. Faculty contact ([2.7](#27-option-plan-for-the-faculty-negotiation)). Freeze the student config: thinking off; `agentic_v1` if its local run is clean. Write the split: train on the 59 non-dev tasks, evaluate on the 70 dev tasks. Check that the 20 ungradable tasks' sandboxes still start. | |
| Oct 8–12 | Teacher-scoring script: T-think thoughts through vLLM; prompts for all 7 teachers built from `calls.jsonl` token ids. Round-0 rollouts on the train split (4 per task, A100). | |
| **Tue Oct 13** | Diagnostic over the 70 dev trajectories. | **Gate 1** ([1.4](#14-what-to-change), change 1). If it fails, the fallback paper starts Oct 14. |
| Oct 12–17 | The trainer: packed student samples, the full-vocabulary teacher under `disable_adapter()`, forward KL, clipping, masks; per-call teacher sequences for T-think. Loss check, smoke. | |
| Oct 17–21 | Round 1 with the best teacher (likely T-think): train at rank 16, then dev with 2 seeds. In parallel, the baselines: `agentic_v1` and SFT self-distillation. | |
| **Thu Oct 22** | | **Gate 2:** repeat and malformed shares down clearly, and resolved ≥ the thinking-off base on paired tests. If not, the fallback paper, centred on the diagnostic. |
| Oct 22–Nov 2 | Round 2 on fresh rollouts. Arms: T-patch (PSP's test at 31B, with the gold patch), T-hint, and C-wrong if budget allows. | |
| by Fri Oct 30 | One Kaggle submission of the best adapter. Scoring takes 10–15 h, and errors have needed resends. | |
| from Mon Oct 26 | Write the related work and method sections in parallel. Optional arXiv preprint once round 1 has a result. | |
| Nov 2–5 | Final seeds and figures. Numbers frozen on Nov 5. | |
| Fri Nov 6 | Create and save the Writeup on Kaggle, to check the editor's save bug. | |
| **Sun Nov 8** | Submit a complete version (tie-break). | |
| Nov 9–11 | Polish, count words, check links. Final submit by Nov 11. | |
| Thu Nov 12 | Buffer only. | |

### 2.3 Compute estimate

- **Measured inputs:**
  - thinking-off trajectories: 43k packed tokens per task (mean), ~370k per-call prompt tokens, 2.7 min per task with setup;
  - the trainer: ~400 tok/s on long sequences (315 at 16k, 450–515 at 5–15k);
  - A100 rollouts: 4 lanes;
  - a thinking-off dev eval: ~2.3 h on 2 L40S, about 4.5 L40S-hours.
- **Assumed:**
  - 236 rollouts per round (59 tasks × 4);
  - vLLM prefill at ~2,000 tok/s on an A100;
  - a no-grad teacher forward at ~1,500 tok/s on the H200;
  - T-think thoughts averaging ~150 tokens.

| Phase | A100-h | H200-h | L40S-h |
|---|---|---|---|
| Diagnostic: 7 teachers × 70 dev trajectories (T-think per call) | 10 | – | – |
| Round-0 rollouts (base student) | 3 | – | – |
| Round 1 (T-think): thoughts, 1 epoch with per-call teacher, dev × 2 seeds | 2 | 16 | 9 |
| Round 2: rollouts, thoughts, train, dev × 2 seeds | 5 | 16 | 9 |
| T-patch arm (packed teacher), 1 epoch, 2 seeds | – | 9 | 9 |
| T-hint arm, 1 epoch, 2 seeds | 2 | 12 | 9 |
| Baseline: SFT self-distillation (thinking-on rollouts, train, 2 seeds) | 4 | 3 | 9 |
| Baselines: `agentic_v1` × 2, thinking-on 3rd seed | – | – | 14 |
| Smokes, loss checks, failed jobs (+25%) | 7 | 14 | 15 |
| **Total (~180 GPU-h)** | **~33** | **~70** | **~75** |

- **Without packing**, the student side alone becomes ~450 H200-hours. That doesn't fit, so packing is a requirement.
- **A cropped teacher cache** would cut T-think's 16 H200-hours per round to ~9.
- **Against our usage so far:** about 60 GPU-hours since Sep 20 (51.7 in `sharing`, 8.2 in `gpu`). The plan triples the rate for five weeks, on top of the main track.
- **Wall clock per round,** if nothing goes wrong:
  - rollouts 3 h;
  - thoughts 1 h;
  - training 12–16 h in ≤ 8-h jobs;
  - 2 eval seeds 4.6 h.

  That is 20–25 h plus queue waits. Plan on 2–3 days per round with debugging.

### 2.4 What Explorer gives us

- **Our access:** two course accounts, `cs6120.202710` and `cs6140.202630`.
  - QOS `gpu`: 1 GPU per job, 4 running and 8 queued, 8 h.
  - QOS `sharing`: 1 h, 2 running and 4 queued.
  - `multigpu`, `gputest` and every lab partition are hidden from us.
- **Account risk:**
  - Course access lasts "for the duration of the course". After grades, "all student personal directories will be deleted" ([RC](https://rc.northeastern.edu/courses-on-explorer/)).
  - `cs6140.202630` is by its term code a Spring 2026 course, which has ended.
  - Fairshare is computed per account, so we share priority with whole classes.
- **What runs where:**
  - vLLM: A100 and L40S only. On the H200s it fails, because the host wheel's Marlin kernels ship sm_80 code plus CUDA 12.9 PTX, and the H200 driver (570.86.15) can only JIT up to 12.8 ([baseline/README.md](https://github.com/usp787/Gemma4_kaggle_repo/blob/main/baseline/README.md#run-order)). The GPUs themselves are fine: our Nemotron project served vLLM on these H200s in April 2026 from a CUDA 12.8 container.
  - Training: H200 only. An A100 runs out of memory at 16k context.
  - Scores: L40S, at about Kaggle's speed per request. An A100 serving 4 lanes is about as fast per request (33 tok/s).
- **GPU inventory** (`sinfo -a`, 2026-10-08):

  | GPU | Public (`gpu`, `multigpu`, `gputest`, …) | Lab partitions |
  |---|---|---|
  | H200 141 GB | 4 nodes × 8 (d4052–d4055) | `torresani-lab` 2×8, `lilaclab` 1×8, `LustigLab` 1×4, `gyorilab` 1×4 |
  | H100 | – | `ai-biodigital` 1×4 |
  | A100 80 GB | ~11 GPUs in `gpu` | `177huntington`, `usace`, `ai-jumpstart` and others, 8 per node |
  | L40S | only via `sharing` (1 h) | `jlab` 3×8, `robot_learning` 1×8, `redwood` 2×4 + 1×3, `d2r2` 1×4, `xz-group` 1×4 |
  | B200 / B300 / GB200 | **none** (Blackwell is on AICR, 2.5) | **none** |

- **Queue, 2026-10-08 01:15 EDT:**
  - `gpu` has 96 H200 jobs pending, mostly 8-h jobs from about 8 users each at their 8-job cap.
  - Our own H200 and A100 jobs so far waited 0.2–24 min, since short jobs backfill.
  - `multigpu` runs 1 × 8-H200 and 1 × 4-H200 job, with 2 × 8 and 3 × 4 pending.
  - In `sharing`, several L40S nodes were idle.

### 2.5 AICR: where NEU's Blackwell GPUs are

The Massachusetts AI Compute Resource sits at MGHPCC in Holyoke. NEU RC runs NEU's access to it. Verified on [docs.aicr.ai](https://docs.aicr.ai/system-description/) and [NEU RC](https://rc.northeastern.edu/2026/05/21/aicr-hpc-cluster-updates-from-the-avp/) on 2026-10-08.

- **Hardware:**
  - 248 B200: 31 nodes × 8, 180 GB HBM3e each, NVLink 5;
  - 152 RTX PRO 6000: 19 × 8, 96 GB each;
  - 128 cores and 2.25 TB RAM per GPU node.
- **Partitions:**

  | Partition | Nodes | Max time | Limit |
  |---|---|---|---|
  | `b200-batch` | 25 | 24 h | 32 GPUs per user |
  | `b200-devel` | 2 | 4 h | 2 GPUs at a time |
  | `b200-fullnode` | 4 | 24 h | whole nodes, multiples of 8 |
  | `rtx-batch` | 17 | 24 h | 32 GPUs per user |
  | `preemptable` | 46 | 24 h | no per-user cap; lowest priority; requeued |

  Fairshare runs institution → project → user.
- **Software:** Apptainer with `--nv`; NGC images built for CUDA 13. Storage: 100 GiB home, 10 TiB scratch (purged), and project work space.
- **Access:**
  - "AICR accounts are managed through your institution's research computing team."
  - At NEU, "faculty and PIs" submit a brief "Project Proposal for AICR HPC Cluster", accepted on a rolling basis since June 2026.
  - Students get in through a PI's project. The turnaround isn't published.
  - The AICR access page still says "initial alpha and beta-test users", while NEU's status page says "fully operational".
- **What it would do for us:**
  - **Capacity:** 32 GPUs and 24-h jobs per user, against Explorer's 4 GPUs and 8 h. One `b200-fullnode` job can hold vLLM rollouts and training on the same node: a true online on-policy loop instead of 2–3 rounds.
  - **Kaggle-faithful evals at scale:** the token proxy can hold each reply to a set per-call speed (`PACE_CALL_S`, `PACE_TOK_S`).
    - Its default is the old 0.6 s + tokens / 25.5. The public refit for adapter-free zips is 0.49 s + tokens / 34.5.
    - Paced that way, many agents can share one B200, and a 70-task eval could finish in under an hour instead of 2.3 h on two L40S.
    - This needs one validation run against an L40S score, and Kaggle's adapter tax measured first (`kaggle_probe`).
  - **The driver:** a CUDA 13 driver JITs the host wheel's CUDA 12.9 PTX, so vLLM should load. Check it on `b200-devel` first.
- **Costs:**
  - onboarding: 1–3 days for login certificates, rebuilding the envs as Apptainer images, and copying our 22 GB of data within our own project space (the data rules forbid sharing it, not moving it);
  - an unknown approval lead time.
  - Access by ~Oct 20 helps round 2, the arms and the final seeds. Later access still helps the main track through Dec 2.

### 2.6 Do we need multi-GPU jobs, or B200/B300?

**Not needed.** Training a 31B LoRA fits one H200 (88–91 GiB peak), and ~180 GPU-hours fit within 4 + 2 concurrent GPUs over five weeks.

What extra hardware would buy:

| Resource | Speeds up | By how much | Matters? |
|---|---|---|---|
| AICR B200 project | everything: evals (paced lanes), rollouts, training, an online loop | evals ~3× faster or more; training ~1.5–2× per GPU (estimate); 8× the concurrency | **Most, if approved in time** |
| More L40S slots, no 1-h cap (a lab node) | dev evals: 10 shards at once instead of 2 | ~35 min per eval instead of ~2.3 h; 3 seeds become affordable | **High:** evals sit on every round's critical path, and the main track shares our 2 slots |
| A guaranteed H200 (lab node) | training rounds | no queue risk, no 8-h job splitting | **High near deadlines** |
| `multigpu` (8 GPUs, 24 h) | DDP training | 12–16 h per round → ~3 h on 4–8 H200 | Moderate |
| vLLM on Explorer's H200 (driver ≥ 575, or CUDA forward-compat `libcuda`) | rollouts and teacher scoring on 32 public H200s | ~3× decode per lane over L40S | Moderate. We can test conda-forge `cuda-compat` ourselves; a driver upgrade is RC's call. |
| B300 specifically | – | bf16 about the same as B200 (estimate) | **No** |

**Speed per GPU** (H200 = 1.0×; datasheet values; the realistic columns are estimates, not measurements):

| GPU | Memory bandwidth | Decode (W4A16 31B, batch 1–8) | Dense bf16 | LoRA training |
|---|---|---|---|---|
| L40S | 0.86 TB/s | ~0.15–0.2× | 362 TF | doesn't fit (48 GB) |
| H200 | 4.8 TB/s | 1.0× | 989 TF | 1.0× |
| B200 | ~8 TB/s | ~1.3–1.6× | ~2.25 PF | ~1.5–2.0× |
| B300 | ~8 TB/s | ≈ B200 | ≈ B200 | ≈ B200 |

- A B200 makes each GPU about 1.5–2× faster.
- What matters more for us is AICR's limits: 8× the GPUs per user and 3× the job length.
- Our trainer runs at 6–9% of the H200's bf16 peak, so packing and attention kernels gain more than any newer GPU.

### 2.7 Option plan for the faculty negotiation

**Why:** compliance first, capacity second. Every option below except the cloud fallback needs a faculty sponsor.

| # | Ask | What the faculty member does | Lead time | What it gives us |
|---|---|---|---|---|
| 1 | **Sponsor this as NEU research:** add me to your lab's storage space and research account. RC: "All Explorer users must have membership in a PI/Staff-owned storage space." | add me with RC's `project` command, or approve my "Research Computing Access Request" (ServiceNow) as sponsor | "up to 24 hours" after sponsor approval ([RC](https://rc.northeastern.edu/getting-access/)) | policy compliance for both tracks; an account that survives the course; our own fairshare. Needed for #2 and #3. |
| 2 | **An AICR project proposal** (B200, Oct 15–Dec 2, me as user) | submit the brief "Project Proposal for AICR HPC Cluster" form; email rchelp@northeastern.edu with "AICR" in the subject, asking for onboarding before Nov 12 | unknown; ask today | 24-h jobs, up to 32 GPUs; paced evals at scale; an online loop ([2.5](#25-aicr-where-neus-blackwell-gpus-are)) |
| 3 | **`multigpu` access** on Explorer | endorse my request ([form](https://bit.ly/NURC-PartitionAccess), "Multigpu"). I run RC's scaling test on 1/2/4/8 GPUs; it needs efficiency > 0.5 ([RC docs](https://rc-docs.northeastern.edu/en/latest/gpus/multigpu-partition-access.html)). | 1–2 weeks (estimate) | 8 GPUs, 24-h jobs on the public H200s; DDP training rounds |
| 4 | **Guest access to a lab partition** through Nov 12 (better: Dec 2): L40S for evals or H200 for training, at low priority | ask the owning PI: yours, or a colleague's (`jlab`, `robot_learning`, `redwood`, `torresani-lab`, `lilaclab` …). RC then adds me. | days, if the owner agrees | evals without the 1-h cap or the 2-job limit; a guaranteed H200 |
| 5 | **Fallback funds:** $1.5–3k, only if #2 isn't live by Oct 21 | discretionary or grant funds | same day | on-demand H200 or B200 at roughly $3–7 per GPU-hour (unverified estimate) |
| – | Not asked: B300 specifically, a deadline reservation (RC has no documented process), hardware buy-in (months) | | | |

**What we offer in return:**
- acknowledgement or co-authorship, as the faculty member prefers;
- Apache-2.0 code;
- the winners' NeurIPS 2026 expo slot if the paper places;
- a non-archival paper that can grow into a workshop or conference submission;
- a weekly usage summary, and no idle GPUs (RC's IdleBot cancels idle jobs anyway).

**Peak need:** 6–8 concurrent GPUs from Oct 17 to Nov 5. About 180 GPU-hours to Nov 12, then whatever the main track uses to Dec 2.

## 3. Does the paper work help the main track?

### 3.1 Directly: an adapter as a submission

- **What would ship:** thinking off and a rank-16 adapter on the main agent, plus possibly a thinking-on advisor without the adapter. That is `agentic_v2`'s structure, since adapters and thinking are set per agent.
- **The bar is high.**
  - Thinking off scored 0.05 on Kaggle (3/58) against 0.12–0.13 with thinking on.
  - It dropped 5.2× from local to Kaggle, against 3.0× for thinking on: the private repos seem to need more reasoning.
  - Public thinking-off bundles with better prompts do reach 0.12–0.15, so the gap is partly our config.
- **The upside is time.**
  - With thinking off, the 12-h run projects to 6.4 h against 9.5 h with thinking on.
  - An agent with thinking-on hygiene but no thinking time could spend the difference on a longer clock. 8 min per task gave +5 tasks locally in v1.1, but didn't fit 12 h with thinking on.
- **The own-thinking teacher aims at exactly what the main track lacks:** the call hygiene of thinking on, at the speed of thinking off.
- **The LoRA tax barely matters with thinking off.** Rank 16 costs 8% on the L40S, and a replay at 26% slower decode kept 17 of 19 resolved tasks. Kaggle's own low-rank tax is still unmeasured (`kaggle_probe`).
- **Our estimate** (judgement):
  - 25–35% that an adapter bundle beats our 0.13 on Kaggle;
  - about 10% that it reaches 0.18, the score just below the top team.
  - Expected change: −1 to +3 tasks of 58, against ±2 of noise.
- **No adapter bundle has a confirmed Kaggle score yet** ([743213](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743213)). Ours would test that path too.

### 3.2 Indirectly (likely)

- **The diagnostic tells the main track what to fix without training.** If the teachers' disagreement concentrates on certain argument names or retry patterns, those become prompt rules or skills, with no LoRA tax.
- **The trainer gets 7–9× faster** from packing. That carries straight to RFT round 1 and hint distillation (`post_train` options 1 and 3).
- **Evaluation method:** 2-seed paired runs, with trace metrics as the primary measure, make every main-track config decision sounder.
- **The fallback study tunes the main-track config directly.** A thinking budget of 512–1024 saves an estimated 7–17% of time on the v28 traces (`agentic_workflow`), and nobody has measured what it costs in resolved tasks.
- **The runs serve both tracks:** `agentic_v1`'s local run is a paper baseline and a main-track decision at once.
- **AICR and the sponsorship** serve the main track through Dec 2, and keep both tracks within RC policy.

### 3.3 Costs and conflicts

- **GPU slots:** both tracks share one user's QOS (4 `gpu` + 2 `sharing`). Without more resources, paper evals delay main-track evals.
- **Kaggle submissions:** 1 per day for both tracks. The paper needs 1–2 (Oct 30, and a resend if Kaggle errors).
- **The rich holdout:** the main track kept the 39 sound rich tasks as its untouched holdout, and the paper trains on them. The main track's real holdout is Kaggle's hidden set anyway, but its local checks lose their only non-fastapi repo.
- **Disclosure:** the paper is public on Nov 12, and the main track ends Dec 2.
- **Our time:** one person on two tracks. The gates exist so that a failing direction stops costing time early.

### 3.4 Recommendation

Run both tracks in parallel, with these priorities:
1. **Shared work first:** re-fetch the model, run `agentic_v1` locally, run `kaggle_probe` (adapter tax), contact the faculty member.
2. **The paper's diagnostic before any main-track training:** it decides, for both tracks, whether self-distillation or prompt fixes are the better use of the next three weeks.
3. **After Nov 12**, the pipeline stays useful for the main track's last 20 days: more rounds on more tasks, against the final config, on AICR if it's live.

## Next 48 hours

1. Re-fetch the model (`sbatch slurm/fetch_model.slurm`), then check `ls -la` on the model directory before anything else runs.
2. Send the faculty request ([2.7](#27-option-plan-for-the-faculty-negotiation)). Asks #1 and #2 have the longest lead times.
3. Start `agentic_v1`'s local run, a baseline for both tracks.
4. Write the train/dev split file. Smoke-start the 20 ungradable tasks' sandboxes.
5. The teacher-scoring script: T-think thoughts through vLLM; prompts for all 7 teachers from `calls.jsonl` token ids; first on 5 trajectories.

## Sources

- **OPSD:** [Reference/opsd.pdf](Reference/opsd.pdf), Tables 2–6 and 8 and Appendix A. Code: [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD), `opsd_trainer.py` and `data_collator.py` (read 2026-10-08).
- **2026 literature:** the arXiv pages linked in [1.5](#15-novelty-check-2026-literature), read 2026-10-08. Only PSP's full HTML was read (its SWE-bench Verified table); for the others we read the abstracts. [vLLM V1 guide](https://docs.vllm.ai/usage/v1_guide.html) on prompt logprobs and the prefix cache.
- **Main track** ([Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo)):
  - `agentic_workflow/README.md`: thinking time, repeats, malformed calls;
  - `post_train/README.md` and `post_train/RFT/{README,round0,v28}.md`: trainer speed, LoRA tax, round 0;
  - `post_train/v28/README.md`: noise, local vs Kaggle;
  - `docs/{hpc_workflow,gpu_queue_log}.md`: QOS, queue waits, GPU roles;
  - `baseline/README.md`: why vLLM fails on H200;
  - `logs/explorer_audit_20261005/AUDIT.md`: repeat and malformed shares.
- **Explorer, 2026-10-08:**
  - `sinfo -a`, `sacctmgr show assoc/qos`, `squeue -p gpu,multigpu`, and our `sacct` since 2026-09-20;
  - packing and re-render analyses over `calls.jsonl` of `baseline_nothink_v28_zip`, `base_v28_zip` and `agentic_v1.1_think_v28_zip`, run as read-only `short` jobs.
- **NEU RC:**
  - [general policies](https://rc.northeastern.edu/research-computing-policies/general-policies/);
  - [courses on Explorer](https://rc.northeastern.edu/courses-on-explorer/);
  - [multigpu access](https://rc-docs.northeastern.edu/en/latest/gpus/multigpu-partition-access.html);
  - [AICR updates](https://rc.northeastern.edu/2026/05/21/aicr-hpc-cluster-updates-from-the-avp/);
  - [AICR status](https://rc.northeastern.edu/aicr/aicr-status-updates/).
- **AICR docs:** [system description](https://docs.aicr.ai/system-description/), [Slurm basics](https://docs.aicr.ai/running-jobs/slurm-basics/), [containers](https://docs.aicr.ai/software/containers/), [request access](https://docs.aicr.ai/request-access/).
- **NEU RC, access:** [getting access](https://rc.northeastern.edu/getting-access/): storage-space membership, sponsor approval, "up to 24 hours".
- **Not verified in this session:** cloud GPU prices, the B200/B300 speed estimates, Explorer's maintenance day, and the `multigpu` turnaround. These came from a research agent's memory or estimates.
- **Kaggle CLI 2.2.4, 2026-10-08:** `competitions list`, `leaderboard`, `submissions`.
