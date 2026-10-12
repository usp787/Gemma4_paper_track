# loops/literature.md: what is already known about our tool-call loops

<!-- written 2026-10-12 by the literature-review session (lit_review_prompt.md). Every source was fetched on that day. Our own numbers come from lit_review_prompt.md, ../gate1/README.md and ../baseline/README.md. -->

**Scope:** the loop behaviour of the thinking-off Gemma 4 31B agent's failed rollouts. This file:
- answers the user's five questions;
- checks claims (a)–(f) for novelty;
- checks the works we already cite;
- gives the must-cite list.

How the sources were read is in [Sources](#7-sources). In the tables, ✓ marks a number that was re-checked against the source a second time.

## 1. Bottom line

1. **Loops are not new.**
   - Agents repeating an identical action is a named failure mode since ReAct (2022), and a harness stop rule since Reflexion and WebArena.
   - It is a standard metric in SWE training papers. In SWE-Gym, 29–40% of zero-shot open-model runs repeat one action 3 times in a row. In SWE-smith, over 25% of the 32B model's runs hold a run of 10 or more repeats, against under 4% for Claude 3.7.
   - OpenHands, Gemini CLI, Cline, Roo Code and Qwen Code all ship a loop detector.
2. **They are not new for our model either.**
   - Gemma 4 31B in its non-thinking mode averages 108.8 turns, against 18.0 with thinking, with "failure-driven retry loops" (Yuan et al. 2026).
   - Kaggle entrants report the baseline agent re-running one command 69 times (thread 745774).
   - Another entrant writes that "Loops and malformed tool calls are a large share of failures for every agent here" (744844).
3. **The mechanisms are known in neighbouring settings:**
   - copies reinforce themselves (Xu 2022);
   - models condition on their own errors (Sinha 2026);
   - conversational inertia (Wan 2026);
   - the surface form of a failed call gets copied (Gumaan 2026, ≤1.7B models);
   - agents don't link actions to their outcomes (Liu 2026).
4. **What nobody has measured:**
   - the lock-in hazard of a coding agent;
   - four levers compared at the same loop states;
   - a count of how loops end.

   The nearest works are Liu 2026 (p(3rd identical action) is 37% pooled, 51% for Qwen3-8B), MIRAGE-Bench (agents repeat again 21–59% of the time at a 3rd or 6th repeat) and Gumaan (log-prob probes on ≤1.7B models).
5. **Verdicts:** (a) partly taken; (b) partly taken; (c) partly taken; (d) partly taken; (e) partly taken; (f) open. In short:
   - (a) The idea is known; our measurement is open.
   - (b) An open Hugging Face PR makes our exact template edit, but it measures stray thoughts, not loops.
   - (c) Taken by Gumaan, Liu and Wan, but none compares the levers with thinking.
   - (d) Taken for math OPD only.
   - (e) SWE-smith already calls repeats "symptoms"; our counterfactual ceiling is new.
   - (f) No one has run it as a controlled study.
6. **The study can carry the paper as a mechanism-and-levers study, not as a discovery.**
   - Build it around "repeated tool calls", lock-in, and which lever breaks it.
   - Position it against Gumaan, Liu, Wan, Sinha and SWE-smith.
7. **Temperature will be the first question a judge asks.**
   - Google prescribes T = 1.0 "across all use cases", yet most published loop rates are measured at T = 0. We found no agent study that sweeps temperature.
   - Our T = 1 replay (p(copy) 0.73 at 3rd+ copies) and `agentic_v2`'s lock-in of 73% at T = 1.0 answer it in part. One closed-loop T = 1.0 run would settle it.
8. **Generality needs a second model.**
   - Loops are reported for Qwen3, Qwen2.5-Coder, GLM, Gemini and every Gemma 4 size.
   - The cheapest second model is Gemma 4 12B or E4B through swegemma v30. E2B and E4B write no empty thought with thinking off, which tests (b) for free.
9. **Eight of our citations were wrong or too broad, and were fixed on 2026-10-12** in [../README.md](../README.md), [../adaptive_inference/README.md](../adaptive_inference/README.md), [../opsd_assessment.md](../opsd_assessment.md) and [../gate1/README.md](../gate1/README.md) ([§4.1](#41-the-works-we-already-cite-checked)). What was wrong:
   - Cognitive Companion's "52–62%" is −52% on one prompt;
   - our PSP quote is wrong;
   - "The Danger of Overthinking" does not show that more reasoning hurts;
   - NetConfArena's "invalid actions" are rejected ones;
   - TAB saves 20% and 15%, not 35% and 30%, on agent tasks;
   - 2609.14636 is about something else;
   - Sinha's "thinking removes" self-conditioning holds only for Qwen3's thinking models, on a synthetic task;
   - Real-Time Detection and Repair's loops are injected, and its gain doesn't come from repairing them.
10. **The biggest risk is being scooped.** Other entrants hold the same observations, an open HF PR targets the template, and the host is "investigating" it (744844, 2026-10-12). Submit early and cite the threads.

## 2. Answers to the five questions

### Which phenomenon is ours, and what to call it

| Phenomenon | Names in the literature | Ours |
|---|---|---|
| Token-level degeneration (text repetition) | "repetition loops" (Holtzman); "self-reinforcement effect" (Xu); "looping" (Pipis); "fast-commit loop" vs "doom looping" for Gemma 4 (Lazaridis) | Not our failure: thinking-off calls are short and well formed. A risk only inside T-open's forced thoughts, since Gemma 4 31B loops more at the token level with thinking on (Lazaridis). |
| Action-level repetition (identical tool calls) | "Stuck in Loop", the same action 3 times in a row (SWE-Gym, SWE-Bench Pro); "step repetition" (MAST, Terminal-Bench); "repeated tool call" (Otani); "exact repeat rate" (Gumaan); "next-action repetition" (Liu); "doom loop" (OPENDEV, practitioners) | Stale repeats: 45% (`off`), 28% (`off_args`), 7% (`on`). 88% of `off_args`'s are identical `run_command` calls. Lock-in is 83–94% after 2 copies. |
| Semantic loop (variants of the same failed search) | "search loop", 10+ search or read actions with no write (Shu); "non-informative search" (ReAct); "repetitions arise from unsuccessful prior searches" (Bouzenia & Pradel); duplicate queries (KV-compaction study) | After an empty search, the next call is a variant 78% of the time and a copy 1%. 25% of chains cycle back within 2–5 calls. |
| Copying of one's own errors | "self-conditioning" (Sinha); "conversational inertia" (Wan); "anchoring" (Verma); "self-correction blind spot" (Tsui); "context contamination" (Yang) | The format slip: 6% of ranged reads before the first one, 82% after. Copies in context: dedup takes p(copy) from 0.94 to 0.14. |

- **Suggested name: "tool-call loops"**, measured in stale repeats, in three stages:
  - **onset:** an empty search, then variants of it (a search loop);
  - **return:** back to an earlier call, often one that compaction hid;
  - **lock-in:** identical copies, which is self-reinforcement at the action level.
- **For findability:**
  - Put "repeated tool calls" and "loops" in the title and abstract. In arXiv abstracts from 2025–26 the most common phrases are "repeated actions" (14), "infinite loops" (13) and "repeated tool calls" (8).
  - Link to "self-conditioning" and "conversational inertia".
  - Use "doom loop" once, as the practitioners' word. Lazaridis uses it for a different, token-level failure.

### Q1. Has the loop behaviour been discovered? What progress, and what challenges remain?

**The literature**
- **Named and counted:**
  - ReAct: the model "repetitively generates the previous thoughts and actions". A "non-informative search, which counts for 23% of the error cases, derails the model reasoning".
  - SWE-agent: edit loops are 23.4% of unresolved runs.
  - SWE-Gym: 3 identical actions in a row in 29.4–39.6% of Qwen2.5-Coder runs on Verified. ✓
  - SWE-smith: repeat runs of 10 or more mean an 89% chance of failure. ✓
  - SWE-Bench Pro: "Stuck in Loop" is 33.9% of Claude Sonnet 4's no-submit failures and 19.8% of Qwen3-32B's.
- **Same model and mode:**
  - Yuan et al. 2026: Gemma 4 31B "Instant" (non-thinking) in Claude Code makes 2,757 edits with a 95.4% failure rate, "trapped in a repeated edit-failure pattern". ✓
  - The Kaggle forum and vLLM #40080 report the same. ✓
- **Mechanisms, all from neighbouring settings:** self-reinforcement (Xu), self-conditioning (Sinha), conversational inertia (Wan), copying of the failed call's surface form (Gumaan), weak action–outcome links (Liu), and risk aversion plus temporally correlated errors (Pipis).

**Progress**
- **Detection is standard.**
  - OpenHands stops at 4 identical action–observation pairs or 3 identical errors.
  - Gemini CLI flags 5 identical calls, plus an LLM judge from turn 30.
  - Cline warns at 3 and stops at 5. Roo Code blocks the 4th.
- **Context edits cut repeats:**
  - describing the failed call instead of repeating it removes 76% of Gumaan's effect;
  - outcome labels and experience notes (Liu);
  - clipping the history (Wan).
- **Decoding:** a ban on previously failed strings takes exact repeats from 31% to 8% (Gumaan).
- **Training:** SFT cuts SWE-Gym's loop rate by 4.6–18.6 points. RL papers penalise over-long, looping trajectories.
- **Thinking:** it cuts Qwen3-8B's repeated actions from 0.496 to 0.124 (NetConfArena). ✓

**What remains**
- **Suppressing repeats rarely raises success:** SWE-smith (40.2% → 38.4%), Liu, Gumaan.
- **Detectors fire late:** at 3–5 copies, while our lock-in starts at 2.
- **Models don't ask for help or think when stuck.**
  - Ambig-SWE: "Without explicit prompting, models almost never interact".
  - Our advisor was the next call after a bad call 1.1% of the time.
- **Most evidence is correlational,** measured at T = 0 on ≤32B or API models.
- **Missing entirely:** a lock-in hazard curve, and a count of how loops end.

**Ours, against it**
- **Matches the literature:**
  - the onset after an empty search (ReAct; Bouzenia & Pradel; Otani: "returns an empty result" → the identical call again);
  - re-sends after lost context (WebVoyager's clipping; duplicate searches under KV compaction rise from 11.7% to 21.3% ✓);
  - fewer repeats with thinking on (NetConfArena; Yuan).
- **New:**
  - the hazard curve, 83–94% after two copies;
  - compaction as the most common exit (8 of 23 runs), while it also restarts 44% of chains;
  - the four levers compared at the same states;
  - the model's own thinking switching off inside loops;
  - the counterfactual ceiling.

### Q2. Can it be reproduced stably in Gemma 4, and in other open models such as Qwen?

**From the literature: which models, sizes, modes and settings loop**

| Model | Mode | Setting | What was seen | Source |
|---|---|---|---|---|
| Gemma 4 31B int4 QAT | off | host harness, vLLM 0.19, 70 SWE tasks, T = 0.2 | stale repeats 45%; lock-in 83–94% | ours |
| Gemma 4 31B | "Instant" (off) vs thinking | Claude Code, vLLM 0.20, ADE-Bench | 108.8 ± 178.7 vs 18.0 ± 15.3 turns; 2,757 edits, 95.4% failed | Yuan 2026 ✓ |
| Gemma 4 31B QAT | the public bundle | Kaggle 4×L4, T = 0.2 and 0.7 | the same `python3 -c` 69×, `grep` 46×; worse at T = 0.7 | Kaggle 745774 ✓ |
| Gemma 4 31B and 26B A4B | – | vLLM 0.19–0.23, JSON-schema output, tool calls | repetition loops; one 26B sub-agent sent "the same tool call ~30 iterations" | vLLM #40080 ✓ |
| Gemma 4 26B A4B AWQ | – | vLLM 0.26, T = 1.0 | JSON strings inside arrays: the lookup fails, then "it retries the identical call" | vLLM #51284 ✓ |
| Gemma 4 26B A4B FP8 | off | vLLM 0.28, T = 0.2 and 1.0 | opens a thought itself after tool responses: 80/80 (short prompts), 229/300 (~14k-token tool chats) | HF PR #67 ✓ |
| Gemma 4 12B | – | a Codex prompt with 10 tools; T = 0, 0.7, 1.0; F16 too | a looping thought channel in 21 of 48 seeds | HF 12B #41 |
| Gemma 4 E2B–31B | on and off | enumeration prompts, T = 0.7 | token loops; 31B loops more with thinking on | Lazaridis 2026 ✓ |
| Gemma 4 E4B | not stated | free-text reasoning, T = 1.0 | step-to-step repetition | Cognitive Companion |
| Qwen3-8B / Qwen3-32B | off vs on | network configuration, ReAct, T = 0 | repeated actions 0.496 vs 0.124 (8B); 0.017 vs 0.020 (32B) | NetConfArena ✓ |
| Qwen3-8B / Qwen3-32B | not stated | ALFWorld, ScienceWorld, WebShop, AppWorld | p(3rd identical action) 51% / 19% | Liu 2026 ✓ |
| Qwen3-8B | off | 8 AgentGym environments, T = 0.8 | repeats its last action 22–28% of the time | Wan 2026 ✓ |
| Qwen3-32B | judge-labelled | SWE-Bench Pro | "Stuck in Loop" in 19.8% of no-submit failures | SWE-Bench Pro |
| Qwen2.5-Coder 7–32B | – | OpenHands, T = 0 | 3 identical in a row: 29.4–39.6% of Verified runs | SWE-Gym ✓ |
| SWE-agent-LM-32B | – | SWE-agent, T = 0 | runs of 10+ repeats in >25% of trajectories (Claude 3.7: <4%) | SWE-smith ✓ |
| Qwen3.5-35B-A3B, Qwen3-Coder-Next, Qwen3.6 | – | llama.cpp, vLLM, ~30k-token contexts | the same call "dozens of times" | llama.cpp #20164; NVIDIA forum |
| GLM-5.2 | – | OpenHands | 7 of 107 runs end STUCK (GPT: 0 of 85) | OpenHands issue #4331 |
| Gemini-3-Flash | medium reasoning | KBQA, HotpotQA, T = 0 | the identical call again after an empty result: 30–45% of instances | Otani 2026 ✓ |

- **Breadth:** loops are reported for every Gemma 4 size and every precision tested (int4 QAT, AWQ, FP8, NVFP4, F16), and for every Qwen generation from 2.5 to 3.6.
- **Size:** within a family, larger models loop less (NetConfArena; Liu; Pipis; Gumaan). Our 31B loops heavily with thinking off, so the mode matters as much as the size.
- **Mode:** thinking cuts repeated actions: Qwen3-8B 4× (NetConfArena), ours 45% → 7%, and Yuan's turn counts. It raises other failures, though:
  - token-level loops in Gemma 4 31B (Lazaridis);
  - unparseable actions in every NetConfArena model.
- **Triggers:** empty or no-effect results, long contexts (14–30k tokens), many tools, the template gap after tool responses, and low temperature.

**Feasibility of reproducing it ourselves (nothing was run)**
- **Gemma 4 12B, E4B and E2B:**
  - **Harness and scripts:**
    - swegemma 0.2.11, our v30, adds "support quantized e2b/e4b/12b models". Kaggle staff list 31B, 12B, E4B and E2B (QAT w4a16) as allowed (thread 748287).
    - These sizes share the call syntax and parsers, so [loop_report.py](loop_report.py) and [../adaptive_inference/trigger_replay.py](../adaptive_inference/trigger_replay.py) run unchanged: they read traces.
    - Gate 1 needs two checks: the empty thought's token ids (`EMPTY_IDS`) and its token roles in each size's tokenizer.
  - **What each size tests:**
    - 12B: its template prefills the empty thought as the 31B's does, so it is a pure size test.
    - E2B and E4B: they write no empty thought with thinking off (model card), so they test (b) for free.
  - Cost: below the 31B's per run. Not estimated here.
- **Qwen3 (8B first):**
  - **What a port needs:**
    - It isn't in the host's harness as far as public information shows. swegemma ships only in the wheelhouse dataset, and whether its model registry accepts non-Gemma models is unverified.
    - Locally, it needs Qwen3's tool and reasoning parsers in vLLM, its own chat template, and `enable_thinking=false` passed through ADK.
    - The trace metrics carry over. Gate 1's token-level edits don't, since they are written for Gemma's tokens.
  - **Why Qwen3 is worth it:**
    - With thinking off, Qwen3's template appends `<think>\n\n</think>\n\n` to every generation prompt, after tool results too (Qwen3-8B template ✓). So Qwen3's no-think mode is "always-block" by construction, which makes it the contrast for (b).
    - The literature already shows Qwen3-8B looping with thinking off (NetConfArena, Liu, Wan). Expect it to reproduce at 8B, but not necessarily at 32B (0.017 in NetConfArena).
- **Stability in our own data:**
  - The trace numbers held from v28 to v30: stale repeats 47.2% → 45.0%, the chain 88.2% → 86.5%.
  - `agentic_v2` at T = 1.0 still locks in at 73%.
  - The resolve rate is the noisy readout (±2.2 tasks per run), so lead with trace metrics.

### Q3. Does it occur outside coding?

Yes, in every agent domain we checked.

| Domain | Evidence | Models |
|---|---|---|
| Web | WebArena halts when "the same action is repeated more than three times on the same observation"; WebVoyager's "Navigation Stuck" is 44.4% of failures, made worse "due to the input clipping"; at the 3rd or 6th repeat, agents repeat again 21–59% of the time (MIRAGE-Bench ✓); identical consecutive actions appear in 58.4% of clean runs, up to 54 in a row (StressWeb) | GPT-4(V), Llama-3.1, DeepSeek, Qwen2.5 |
| Computer use | OSWorld: agents "tend to predict repetitive actions"; OpenCUA "loops indefinitely" after an action with no visible effect; feedback-blind repetition is 29.5% of genuine failures (Dong 2026) | GPT-4V, Claude 4 Sonnet, OpenCUA |
| Embodied and games | Reflexion's ALFWorld rule; SwiftSage's fast model "has a tendency to repeat meaningless actions"; loops in ~80% of trials of an embodied puzzle (Zenkri & Brock); BALROG agents revisit rooms; Minecraft "craft retry storms" | GPT-4o, o1, Flan-T5 |
| Multi-turn tool use | the identical call again after an empty result (Otani ✓); BFCL ends a turn after 20 steps, which "usually happens when the model gets stuck in a loop"; Loop@3 is 15.1% under vanilla distillation (Tool-Call Drift); ToolSandbox; NetConfArena ✓ | Gemini-3-Flash, GPT-5-mini, Qwen3, Qwen3.5 |
| Terminal | Terminal-Bench's failure taxonomy includes "Step Repetition" (shares are in figures only); Gemini CLI and Qwen Code ship detectors | 21 models |
| Long horizon | Vending-Bench's "meltdown" loops; TheAgentCompany ("got stuck in a loop"); AgentBench's Task Limit Exceeded, which it ties to repeated generations, is 23.9–82.5% of runs by domain; Sinha's self-conditioning | Claude 3.5, o3-mini, Gemini, 29 LLMs |
| Multi-agent | MAST: "step repetition" in 15.7% | GPT-4o, Claude 3.7, Qwen2.5-Coder |

- **For the Quality score:** the phenomenon is general, so the anatomy and the copying levers should transfer. The empty-thought lever is specific to hybrid-thinking templates.

### Q4. What are the reasons, and what resolves it?

**Causes the literature names, next to ours**

| Cause | Literature | Ours |
|---|---|---|
| Copies reinforce themselves | Holtzman Fig. 4 ("a positive feedback loop"); Xu (the chance of a repeat rises with the number of copies, then saturates); Pipis (the model "becomes increasingly confident… a catalyst" ✓); repetition neurons; next-token entropy falls cycle by cycle (Mahaut & Franzon) | lock-in 83–94% after 2 copies; p(copy) 0.94 at 3rd+ copies |
| Conditioning on one's own context | Sinha; Wan; Gumaan (the call's surface form is 83% of the effect ✓); Verma (33–68% near-identical retries when the model sees its own failed program); Laban | dedup: 0.94 → 0.14; the format slip: 6% → 82% |
| Uninformative results | ReAct (non-informative search, 23% of errors); Otani ✓; OpenCUA; SWE-agent replaces empty output with "Your command ran successfully and did not produce any output"; an `ok` status with an empty payload gives 45.3% fabricated answers vs 0% for `error` (Sethi) | loops start at an empty `grep`/`find`; 270 of 580 repeated `run_command` calls got nothing |
| Lost context | WebVoyager's clipping; KV compaction (duplicate searches 11.7% → 21.3% ✓); summaries "mask failure signals" (Complexity Trap); summaries "re-dump the same ruled-out-names search log" (Self-Compacting) | 44% of chains re-send a call compaction hid |
| Actions not linked to outcomes | Liu: shuffling past actions costs only 2.55 points | 91% of stale repeats got exactly their earlier copy's result |
| Errors in learning and risk aversion | Pipis: an easy cyclic action keeps its probability when progress is hard to learn; distilled students loop more than their teachers (30% vs 4%) | – |
| Mode and template | NetConfArena; Yuan; Google's "ghost" thought channels with thinking off; HF PR #67; leakage between modes (Demystifying Hybrid Thinking) | 45% vs 7% stale; the model writes the empty thought itself at 82% of calls 1–5 and ~35–45% after call 10 |
| Low temperature | see [Temperature](#temperature) | – |
| Task difficulty (the loop as a symptom) | SWE-smith ✓; search loops track difficulty, at 34% / 50% / 68% of easy / medium / hard tasks (Shu ✓); flawed reasoning "drive[s] agents into persistent loop anomalies" (Liu 2509.13941) | the ceiling: about 2 of 70 tasks |

#### Temperature

- **Google:**
  - The model card gives one sampling configuration "across all use cases": temperature=1.0, top_p=0.95, top_k=64. ✓
  - Its docs say nothing about greedy decoding, low temperature or repetition.
  - Google's own loop advice is about context:
    - larger models "may occasionally generate a thought channel even when thinking mode is explicitly turned off… consider adding an empty thinking token to the prompt";
    - long-running agents may keep their reasoning "to prevent the model from entering cyclical reasoning loops" ✓;
    - the July chat template is headed "Fixed tool-calling loops".
- **Qwen:**
  - "DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions."
  - presence_penalty 0–2 "to reduce endless repetitions";
  - thinking off: T = 0.7, top_p 0.8, with presence 1.5 from Qwen3.5 on.
- **Research:**
  - Low temperature makes text loop: Holtzman; Pipis (76% of AIME runs loop at T = 0 for R1-Distill-Qwen-1.5B, and "temperature is a stopgap rather than a holistic solution" ✓); Word Salad Chopper (62.9% of chunks at T = 0 vs 10.6% at T = 0.6).
  - Wan et al. sample their agents at T = 0.8 because "higher temperature reduces brittle failure modes caused by deterministic repetition loops" ✓.
  - Gumaan's clean-restart result is "specific to deterministic decoding" ✓.
- **Agents:**
  - Most published loop rates are measured at T = 0: SWE-Gym, SWE-smith, MIRAGE-Bench, NetConfArena ✓.
  - We found no study that varies the temperature and counts repeated calls.
- **Practitioners on Gemma 4:**
  - T = 0.7 made the loops worse: 5 of 10 tasks looped, 66 calls on average vs 36 (745774 ✓).
  - The 12B looped at T = 0, 0.7 and 1.0 (HF #41).
  - One 26B call loop broke only at T = 1.0 with three penalties: "bumping only `repetition_penalty` is not enough" (vLLM #40080 ✓).
  - With thinking on, moving T from 0 to 1 changed resolved tasks only within 47–51 of 129 (746250 ✓).
- **Ours:**
  - At 3rd+ copies, p(copy) is 0.73 at T = 1, against 0.94 at T = 0.2. At 2nd copies it is 0.26 at T = 1.
  - `agentic_v2` (V1's prompt at T = 1.0, with its rule against a third identical call) still locks in at 73%.
  - So temperature changes how often loops start, not the lock-in.
- **The answer for the judges:** one closed-loop `off_args` arm at Google's T = 1.0 ([§6](#experiments-a-judge-will-expect)).

**Methods that resolve it**

| Group | Method | Best evidence | On agents? | Our replay |
|---|---|---|---|---|
| Decoding | higher temperature | Pipis: "a stopgap" | no | 0.73 at T = 1 |
| | repetition / presence / frequency penalties | Qwen model cards; three penalties together broke one Gemma 4 loop (vLLM #40080) | anecdotal | – |
| | ban previously failed strings | Gumaan: exact repeats 31% → 8%, success unchanged | yes (≤1.7B) | – |
| | resample until the call differs | SWE-smith: repeats cut sharply, resolved 40.2% → 38.4% ✓ | yes | – |
| Context | collapse the copies | LoopGuard prunes repeated spans in the KV cache (token level) | no | dedup 0.14 |
| | describe the failure instead of repeating the call | Gumaan: removes 76% of the effect; repeats 31% → 16% | yes | – |
| | label outcomes; add an experience note | Liu: p(3rd) 36.9% → 32.2% / 21.4%, but "does not translate into improved task success" ✓ | yes | note 0.49 |
| | clip or mask the history | Wan: Qwen3-8B 54.4 → 68.9 ✓; masking ≈ summaries at half the cost (Complexity Trap); a shorter window helps (Sinha) | yes | – |
| | clean restart | Gumaan: the worst harness, repeats 31% → 80%, under greedy decoding ✓ | yes | not tested |
| Prompt and nudge | "do not repeat" | no effect (Gumaan); ignored once looping (745774 ✓); with V1's rule, `agentic_v2` still locks in at 73% | yes | – |
| | an injected warning | Gemini CLI, OpenHands, Cline; SWE-smith; "LLMs can ignore injected text, but they cannot bypass a genuine execution halt" (OPENDEV) | yes | note 0.49 |
| | a budget display | BATS: SWE-bench Verified 36.8% → 55.4% (Gemini-3-Flash, 50 calls) ✓ | yes | our budget warning ends in an empty patch |
| | "Wait" | removes 89.3% of the self-correction blind spot (Tsui) | no | – |
| Thinking | thinking on | Qwen3-8B repeats 4× less (NetConfArena ✓); Qwen3 thinking models don't self-condition (Sinha) | yes | 45% → 7% stale; 0.80 when it may think |
| | forced or parallel thinking | Second Thought: SWE-Bench Pro −0.7 to +3.3, not significant; "Suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops" (s1) | yes | T-open, running |
| | the empty thought | Google's docs; HF PR #67 (stray thoughts 229/300 → 0/300) | not on loops | block 0.50 |
| | trained effort routers | CogRouter, Ares, TAB, ALAR, RACE: none tested at loops; entropy gates can "select… harmful states" (Li 2605.06908) | yes | – |
| Monitors | detect, then nudge or halt | OpenHands, Gemini CLI, Cline, Roo Code, Qwen Code, OPENDEV; Dubey's breaker ends a loop sooner but doesn't fix it | yes | – |
| | restart on an alarm | Gemma 4 31B: +3.0 with the old diff, +1.4–1.8 from scratch (Fail-Fast ✓) | yes | – |
| Training | unlikelihood training, DITTO | text only | no | – |
| | SFT on agent trajectories | SWE-Gym: loop rate down 4.6–18.6 points | yes | – |
| | recovery and refinement data | Agent-R ("avoiding loops", one figure as evidence); AgentRefine | yes | – |
| | RL length or timeout penalties | Golubev 2508.03501; SWE-Master (~20% of RL trajectories cut in "redundant verification loops") | yes | – |
| | context preference learning; steering | Wan's CPL (72.5% vs 68.9%); TACT (+4.8 on Gemma-4-26B ✓); weight edits for token loops (Lazaridis) | yes | – |

### Q5. Is breaking out of a loop good, bad or uncertain?

**The literature**
- **Suppressing a loop without new information doesn't raise success:**
  - SWE-smith: 38.4% vs 40.2% ✓;
  - Gumaan, where the ban leaves success unchanged: "Removing the loop returns the agent's budget to it; whether it can spend that budget well is a question about the model";
  - Liu ✓;
  - Dubey: loops "escalate rather than recover";
  - Second Thought: fewer turns, no SWE gain.
- **Forced exits mostly end badly:**
  - SWE-agent's cost-limit autosubmits resolve 3.1%, against 14.3% for normal submits. Exits with no edit are "guaranteed to be unresolved".
  - NetConfArena's "premature abandonment": "the agent voluntarily submits after a malformed action or unexpected observation" ✓.
  - Cuadron's "Premature Disengagement".
  - OSWorld agents "easily output FAIL".
- **Escapes that bring information help a little:**
  - a restart with the interrupted diff gives Gemma 4 31B +3.0, against +1.4–1.8 from scratch ✓;
  - a budget display adds 18.6 points ✓;
  - after SWE-Gym's fine-tuning, the drop in loops "coincides with a decrease in the empty patch rate".
- **Counter-case:** in an embodied puzzle, noise that broke loops raised success 2.85× (Zenkri & Brock).

**Ours**
- 7 of 23 runs of 3+ copies end at the budget warning, after which the model submits, mostly an empty patch. That is the autosubmit pattern.
- 8 end at a compaction, which also starts 44% of chains.
- Removing every loop would gain about 2 of 70 tasks.

**Verdict: uncertain, and bad when the escape brings nothing new.** The agent leaves the loop but not the dead end.
- In the paper, report what each escape leads to: an empty patch, a wrong patch, or a resolved task.
- Prefer levers that change what the agent knows.

## 3. Novelty of claims (a)–(f)

| Claim | Verdict | Closest works | What stays ours | Wording to use |
|---|---|---|---|---|
| (a) Confidence gates self-directed thinking or help-seeking | **Partly taken.** The idea and its token-level mechanism are known; the agent-level measurement is open. | Xu (higher-probability sentences self-reinforce more; "the model is confident"); Pipis (confidence grows in a loop); CogRouter (depth trained to "maximize the confidence of the resulting action"); HRBench (entropy gates miss "confident wrong answers"); Ambig-SWE; Caddie (a fixed-step critic beat agent-called critics, 25.94% vs 19.06%) | The model's own thinking switch shuts off inside loops: a real thought at 72% of 2nd copies, 23% of 3rd+, 6% of malformed calls. Where it skipped the thought its call was stale 89% of the time; where it thought, it escaped 62%. The advisor was the next call after a bad call 1.1% of the time. No paper measures this. | "Self-directed thinking switches off where it is needed. A router trained on action confidence (CogRouter, RACE) would learn the same." |
| (b) The empty thought changes repetition and loops | **Partly taken** (the edit); **open** (its effect on loops) | Google's docs ("consider adding an empty thinking token"); HF PR #147/#67, the same template edit, measuring stray thoughts and empty replies; Demystifying Hybrid Thinking; Qwen3's template (the block is always present); Tsui ("Wait", a few-token prefix with a large effect) | One step ahead, the block halves p(copy), yet 80% of the sampled calls stay stale. The model's own block drifts from 82% to 35–45%. The whole-task effect is running now. | "The thinking-off template leaves a gap after tool responses." Cite the PR. |
| (c) Editing the context breaks loops better than thinking | **Partly taken** | Gumaan (describing the failure: −76%; an instruction: no effect; a clean restart: worst); Liu (labels and notes); Wan (clipping beats summaries); Sinha (a window helps, and thinking fixes self-conditioning); Verma (blind resampling beats reflection) | The head-to-head at the same states, at 31B: dedup 0.14, note 0.49, block 0.50, thinking 0.80 | Reconcile with Gumaan: our dedup keeps the latest copy, so it is no clean restart. |
| (d) Probability or KL shifts overstate the change in sampled behaviour | **Partly taken** | Mismatch Matters (median KL 0.0004 nats inside loops; students "exploit repetitive loops" ✓); the KL Agreement Trap; Tool-Call Drift ("Probability mass makes this compression appear safe"); OPSD (style tokens dominate the KL); Gumaan (the probe vs the rollouts); e-CUSUM (log-probability is "nearly silent during confident repetition") | For tool-call loops: p(copy) halves, yet 80% of calls stay stale. The model's own thinking as teacher puts less KL on failure tokens than on routine ones (0.51). On the 292 states where the student went straight to the call, C-none's KL ratio is 0.96 per token but 3.77 per call. | "Score the class of the sampled call, not the probability of the student's call." |
| (e) Loops are a symptom | **Partly taken** | SWE-smith ("repetitive actions are better understood as symptoms of the model's difficulty" ✓); Liu; Gumaan; Shu; Fail-Fast. Against: TACT +4.8 (Gemma-4-26B); Zenkri & Brock 2.85×; 2608.06811 +5.0 | The counterfactual ceiling over 9 runs (6 of 23 loop failures solved by any of them, about 2 of 70 tasks), and failures moving from loops to wrong patches | "Confirms SWE-smith's reading with a counterfactual ceiling." |
| (f) A silently accepted slip is copied by later calls | **Open** as a controlled finding; the copying mechanism is well known | Sinha; Wan; Gumaan (his models copied a "you:" label into their replies); vLLM #51284 (JSON in arrays, then "it retries the identical call" ✓); PRISM (silent correction as a middleware pattern, its downstream cost unmeasured); Sethi (`ok` with an empty payload) | 6% before the first slip, 82% after it, while the harness answers `ok`; rules that prevent the first slip cut malformed calls from 21.1% to 1.4% | "A lenient parser hides the error and turns it into an example." |

## 4. Works

Read levels: **full** means the full text; **secs** means the named sections or tables; **abs** means the abstract only. ✓ marks a number re-checked a second time. Dates give v1, then the latest version.

### 4.1 The works we already cite, checked

All of them exist, and every ID is right.

| Work | Says what we claim? | Correction | Where our text has it |
|---|---|---|---|
| Holtzman et al., ICLR 2020, 1904.09751 (v2 2020-02-14) | yes | Cite Fig. 4: "The probability of a repeated phrase increases with each repetition, creating a positive feedback loop." "Self-reinforcement" is Xu's word. | [../gate1/README.md](../gate1/README.md), "Prior work agrees" |
| Xu et al., NeurIPS 2022, 2206.02369 (v2 2022-10-09) | yes | Add the DOI (§5). The setting is a ~750M Transformer with greedy decoding. | same |
| Sinha et al., ICLR 2026, 2509.09677 (v3 2026-03-13) | **partly** | "Thinking removes that" is shown only for Qwen3 thinking models, on a synthetic execution task, with earlier thoughts stripped. CoT prompting doesn't help. | [../gate1/README.md](../gate1/README.md) line 71 (fixed 2026-10-12) |
| Huang et al., ICLR 2024, 2310.01798 | yes | – | same, line 72 |
| Kamoi et al., TACL 2024, 2406.01297 | yes | Add the DOI 10.1162/tacl_a_00713. | same |
| Cognitive Companion, 2604.13759 (v1 only) | **no** | "52–62%" appears only in the abstract. The body shows −52% mean lexical repetition on one prompt over two sessions, with no tools, at T = 1.0; the paper calls itself "a feasibility study". | [../adaptive_inference/README.md](../adaptive_inference/README.md) line 105 (fixed 2026-10-12) |
| Real-Time Detection and Repair, 2608.02464 | **partly** | Its loops are injected; no organic loops are reported. They "escalate rather than recover", and the 52% → 73% comes from deterministic checks. | same, line 108 (fixed 2026-10-12) |
| Second Thought, 2608.13667 | yes | – | same, line 109 |
| Demystifying Hybrid Thinking, 2510.12680 (v2 2026-10-04) | yes | Single-turn only. | same, line 120 |
| The Danger of Overthinking, 2502.08235 | **partly** | Not "more reasoning can hurt": reasoning models that replace interaction with internal simulation resolve fewer issues. Within o1, more effort helped (29.1% vs 21.0%). Its judge rubric labels "Stuck in a loop is good". The three mode names are right. | [../README.md](../README.md) line 246 (fixed 2026-10-12) |
| NetConfArena, 2608.23179 | **partly** | Read "invalid" as "rejected" actions (0.455 → 0.317 for 8B; 0.090 → 0.139 for 32B). Unparseable actions rose with thinking for every model. | same line (fixed 2026-10-12) |
| Ares 2603.07915; ALAR 2606.02871; HRBench 2605.28398 | yes | ALAR's tool use is single-turn BFCL. | adaptive_inference README |
| CogRouter 2602.12662 | yes | The 62% saving is relative to GRPO. | same, line 102 |
| TAB 2604.05164 (v3 2026-08-08) | **partly** | 35% and 30% are measured on plan-and-execute math. On agent benchmarks: "up to 20%" of tokens and 15% of latency. | same, line 104 (fixed 2026-10-12) |
| SwiftSage 2305.17390, NeurIPS 2023 | yes | Its triggers include "Stuck" (5 zero-reward steps) and invalid actions. | same, line 101 |
| "Know When to Stop, Where to Restart", 2609.14636 | – | It is STRIDE, a speed-up for multi-turn OPD training. Nothing about loops or escapes. | [../opsd_assessment.md](../opsd_assessment.md) §1.5: moved from "not opened" into the table (fixed 2026-10-12) |
| OPSD 2601.18734 (v3 2026-03-20) | yes | Its teacher sees a "reference solution". | opsd_assessment §1.5 |
| PSP 2609.29051 | **quote wrong** | The words are "act with confidence but without the information behind it". The SWE-bench Verified table matches (2.40 vs 3.56; 2.88 vs 4.72). | [../README.md](../README.md) line 289 (fixed 2026-10-12) |
| SeOPD 2609.33181 (v2 2026-10-07); ThinkOPD 2609.37044 (v2 2026-09-30); HERO 2606.11559; SDPO 2601.20802 (v2 2026-02-16); Tool-Call Drift 2607.07050 (v6 2026-09-05) | yes | ThinkOPD's "no-think" means "without the teacher's trace", not the template switch. Tool-Call Drift is multi-teacher OPD. | opsd_assessment §1.5 |

### 4.2 All works, by theme

**A. Repetition mechanisms (token level)**

| Work | ID · dates | What it shows | Setting | Overlap with us | Read |
|---|---|---|---|---|---|
| Xu et al., *Learning to Break the Loop* (NeurIPS 2022) | 2206.02369 · 2022-06-06 → v2 2022-10-09 | "the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence"; it saturates; stronger for high-probability sentences; DITTO training | ~750M Transformer, Wikitext-103, greedy | lock-in, at token level | full |
| Holtzman et al., *The Curious Case of Neural Text Degeneration* (ICLR 2020) | 1904.09751 · 2019-04-22 → v2 2020-02-14 | a repeated phrase's probability grows with each repetition (Fig. 4) | GPT-2 Large | the same | full |
| Pipis et al., *Wait, Wait, Wait… Why Do Reasoning Models Loop?* (ICML 2026 spotlight) | 2512.12895 · 2025-12-15 → v2 2026-08-24 | looping is common at low T (76% at T = 0 for R1-Distill-Qwen-1.5B, 37% for the 32B); students loop more than their teachers; risk aversion and correlated errors; confidence grows while repeating; "temperature is a stopgap" | reasoning models, AIME, T = 0–1 | temperature; (a) at token level | secs ✓ |
| Xie et al., *Word Salad Chopper* (EMNLP 2025) | 2511.00536 · 2025-11-01 | 57–63% of tokens are repetition on GPQA-D; 62.9% of chunks at T = 0 vs 10.6% at T = 0.6; a probe detects the loop | reasoning models | temperature | secs |
| Mahaut & Franzon, *Repetitions are not all alike* | 2504.01100 · 2025-04-01 → v2 2025-11-04 | distinct circuits sustain in-context-induced and natural repetition; entropy falls cycle by cycle | Pythia-1.4B, greedy | (a) at token level | §5 |
| Lazaridis et al., *When Can One Neuron Fix Repetition Loops in LLMs?* | 2606.13705 · 2026-06-09 → v2 2026-08-24 | Gemma 4 token loops; 31B with thinking on loops on 5 of 8 "The Wire" generations, with thinking off none; weight edits: 46/384 → 12/384 | Gemma 4 E2B–31B, T = 0.7 | our model family; a risk for T-open | secs ✓ |
| Hiraoka & Inui, *Repetition Neurons* (NAACL 2025) | 2410.13497 · 2024-10-17 → v2 2025-02-20 | neurons activate "progressively… more strongly as repetition continues" | includes Gemma-2B | mechanism | abs |
| Olsson et al., *In-context Learning and Induction Heads* | 2209.11895 · 2022-09-24 | prefix-match-and-copy heads | – | the copy mechanism | abs |
| Duan et al., *Circular Reasoning* | 2601.05693 · 2026-01-09 | "reasoning impasses trigger the loop onset"; CUSUM detection | reasoning models | the onset | abs |
| Xu et al., *LoopGuard* | 2604.10044 · 2026-04-11 | pruning repeated spans from the KV cache cuts loops by over 90 points | reasoning models | dedup, at token level | abs |
| Dong et al., *Rethinking Repetition Problems of LLMs in Code Generation* (ACL 2025) | 2505.10402 · 2025-05-15 | "structural repetition"; grammar-based decoding (RPG) | code generation, single-turn | – | abs |

**B. Self-conditioning and context**

| Work | ID · dates | What it shows | Setting | Overlap with us | Read |
|---|---|---|---|---|---|
| Gumaan, *Feedback That Backfires: Why Small Language Model Agents Repeat the Call They Just Watched Fail* | 2608.23651 · 2026-08-24 | appending a failed call makes it likelier (p over 4 candidates 0.06 → 0.54); its surface form is 83% of the effect; describing the failure removes 76%; "do not repeat" does nothing; clean restart is the worst harness (repeats 31% → 80%); success doesn't follow the repeats | 6 models, 135M–1.7B; ToolShed and MBPP; greedy rollouts of Qwen2.5-0.5B on 24 tasks | **closest to (c), (e), (f)** | full ✓ |
| Liu et al., *When History Fails to Become Experience: Action Calibration in Language Agents* | 2610.02769 · 2026-10-02 | shuffling past actions costs only 2.55 points; p(3rd identical action) on 1,815 histories is 36.9% pooled (Qwen3-8B 51.4%, Qwen3-32B 18.9%, GPT-5.6-Sol 8.7%); outcome labels → 32.2%, experience notes → 21.4%, with no gain in success | ALFWorld, ScienceWorld, WebShop, AppWorld; 5 models | **our probe's design**; (c), (e) | secs ✓ |
| Wan et al., *Mitigating Conversational Inertia in Multi-Turn Agents* (ICML 2026) | 2602.03664 · 2026-02-03 → v3 2026-05-18 | agents "mimic their own previous responses as few-shot examples"; diagonal attention grows with context; clipping beats summaries (Qwen3-8B 54.4 → 68.9); Qwen3-8B repeats its last action 22–28% of the time | Qwen3-8B with thinking off, Llama-3.1-8B, GPT-4o-mini; T = 0.8 | (c), (f) | secs ✓ |
| Sinha et al., *The Illusion of Diminishing Returns* (ICLR 2026) | 2509.09677 · 2025-09-11 → v3 2026-03-13 | errors in context breed errors; scale doesn't help; Qwen3 thinking models don't self-condition; CoT prompting doesn't help; a shorter window helps; ~20–48% of agent failures in AgentErrorBench look similar | synthetic execution; Qwen3 4–32B, Gemma 3 4–27B | (f) mechanism; (c) | full |
| Verma, *Try Again, Don't Look Back* | 2607.26117 · 2026-07-28 | seeing its own failed program makes 33–68% of retries near-identical, vs 2–14% blind; reflection lowers that by about a third; Q4 behaves as FP16 | Qwen2.5-Coder 1.5–7B, DeepSeek-Coder; MBPP+; T = 0.8 | (c); quantization isn't the cause | full |
| Yang, *Why Retrying Fails: Context Contamination in LLM Agent Pipelines* | 2605.08563 · 2026-05-08 | a failed attempt in context raises the per-step error rate 7.1× (a fit to SWE-bench Verified) | theory | (c) | abs |
| Laban et al., *LLMs Get Lost In Multi-Turn Conversation* | 2505.06120 · 2025-05-09 | −39% in multi-turn conversations; models lean on their earlier attempts; T = 0 doesn't fix it | 15 LLMs | (f) | §6–7 |
| Tsui, *Self-Correction Bench* (COLM 2026) | 2507.02778 · 2025-07-03 → v3 2026-08-02 | a 64.5% blind spot for errors in the model's own turn; appending "Wait" removes 89.3% of it | 14 models | (b): a few tokens, a large effect | §3–4 |
| Lindenbauer et al., *The Complexity Trap* (DL4C @ NeurIPS 2025) | 2508.21433 · 2025-08-29 → v3 2025-10-27 | observation masking matches LLM summaries at half the cost; summaries "mask failure signals" and lengthen runs | SWE-agent; Qwen3-Coder-480B, Gemini 2.5 Flash | compaction | §4.4 |
| Li et al., *Self-Compacting Language Model Agents* | 2606.23525 · 2026-06-22 → v2 2026-07-10 | a summary "re-dump[s] the same ruled-out-names search log", and the agent resumes the same dead end | – | compaction restarts loops | secs |
| Liu et al., *Practical Online KV Cache Compaction for LLM Agents* | 2608.00902 · 2026-08-02 | compaction raises duplicate searches from 11.7% to 21.3%: "the agent uses the environment to recover information weakened by compaction" | BrowseComp-Plus, WideSearch | our 44% | abs, Table 3 ✓ |

**C. Loops in coding agents**

| Work | ID · dates | What it shows | Setting | Overlap with us | Read |
|---|---|---|---|---|---|
| Yang et al., *SWE-agent* (NeurIPS 2024) | 2405.15793 · 2024-05-06 → v3 2024-11-11 | edit loops are 23.4% of unresolved runs; after one failed edit, eventual success falls from 90.5% to 57.2%; the interface caps search hits, replaces empty output with a message, collapses old observations; cost-limit autosubmits resolve 3.1% vs 14.3% | GPT-4 Turbo, SWE-bench | harness guards; Q5 | full |
| Pan et al., *SWE-Gym* | 2412.21139 · 2024-12-30 → v2 2025-06-06 | "Stuck in Loop", the same action 3 times in a row: 29.4–39.6% of zero-shot Verified runs; fine-tuning cuts it, with fewer empty patches | Qwen2.5-Coder 7–32B, OpenHands, T = 0 | a metric to report | §4 ✓ |
| Yang et al., *SWE-smith* | 2504.21798 · 2025-04-30 → v2 2025-05-21 | runs of 10+ repeats in >25% of trajectories (Claude 3.7: <4%), with an 89% failure probability; warnings, resampling and temperature bumps cut repeats while resolved fell to 38.4%: "symptoms of the model's difficulty" | Qwen2.5-Coder-32B fine-tune, T = 0 | **(e)** | §4.2, F.5.4 ✓ |
| Yuan et al., *Agentic AI Workload Characterization* (IISWC 2026, per its artifact) | 2605.26297 · 2026-05-25 → v2 2026-09-21 | Gemma 4 31B "Instant" averages 108.8 turns (up to 786) vs 18.0 with thinking; 2,757 edits with a 95.4% failure rate; "failure-driven retry loops"; no success rates by mode | Gemma 4 31B, Qwen3.6-27B; Claude Code; vLLM 0.20 | **our model and mode** | full ✓ |
| Deng et al., *SWE-Bench Pro* | 2509.16941 · 2025-09-21 → v2 2025-11-14 | "Stuck in Loop": 33.9% of Claude Sonnet 4's no-submit failures, 19.8% of Qwen3-32B's, ≤3.4% for frontier models | GPT-5 judge over the last 20 turns | prevalence | §6.3 |
| Bouzenia & Pradel, *Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories* (ASE 2025) | 2506.18824 · 2025-06-23 → v2 2025-10-08 | repetition "several times higher in failing trajectories"; "some repetitions arise from unsuccessful prior searches" | RepairAgent, OpenHands, AutoCodeRover | onset | full |
| Liu et al., *An Empirical Study on Failures in Automated Issue Solving* | 2509.13941 · 2025-09-17 | "Non-Progressive Iteration"; flawed reasoning drives loops | Claude 3.5 Sonnet agents | (e) | full |
| Wang et al., *Fail-Fast, Restart-Smart* | 2608.03222 · 2026-08-04 | loop counts don't help predict failure; Gemma4-31B 62.2% gains +3.0 from a restart with the diff, +1.4–1.8 from scratch | mini-swe-agent, SWE-bench Verified | (e), Q5 | full ✓ |
| Shu et al., *What Resolve Rate Hides* | 2607.06184 · 2026-07-07 | search loops in 41.1% of resolved and 56.1% of failed runs, and in 34% / 50% / 68% of easy / medium / hard tasks | 2,500 Verified trajectories | (e) | secs ✓ |
| Sui et al., *TACT* | 2605.05980 · 2026-05-07 | steering against "overthinking" and "overacting" gives Gemma-4-26B +4.8 (43.9 → 48.7); a prompt reminder does nothing significant | Qwen3.5-27B, Gemma-4-26B-A4B, thinking on | against (e); the note | secs ✓ |
| Antoniades et al., *SWE-Search* | 2410.20285 · 2024-10-26 → v6 2025-04-02 | base agents "get stuck performing repetitive actions"; its value prompt penalises them | MCTS on SWE-bench Lite | – | most |
| Chen et al., *Beyond Final Code* (ICSE 2026) | 2503.12374 · 2025-03-16 → v4 2026-04-09 | errors recur (a TypeError up to 106 times); resolution falls as errors pile up | – | (f) | ~80% |
| Bui, *Building Effective AI Coding Agents for the Terminal* (OPENDEV) | 2603.05344 · 2026-03-05 → v3 2026-03-13 | a "doom loop" guard: 3 identical (tool, args) within 20 calls → a warning, then a pause | product report | a detector; the term | secs |

**D. Other domains and classic agents**

| Work | ID · dates | What it shows | Read |
|---|---|---|---|
| Yao et al., *ReAct* (ICLR 2023) | 2210.03629 · 2022-10-06 → v3 2023-03-10 | repetitive thoughts and actions; non-informative search, 23% of errors, "derails the model reasoning" (PaLM-540B) | §3.3 |
| Shinn et al., *Reflexion* | 2303.11366 · 2023-03-20 → v4 2023-10-10 | reflect when the agent "executes the same action and receives the same response for more than 3 cycles" | §4.1 |
| Liu et al., *AgentBench* (ICLR 2024) | 2308.03688 · 2023-08-07 → v3 2025-10-04 | Task Limit Exceeded, which it ties to repeated generations, is "the predominant failure cause": 23.9–82.5% of runs by domain (29 LLMs, greedy) | §2, §4.3 |
| Cemri et al., *Why Do Multi-Agent LLM Systems Fail?* (MAST) | 2503.13657 · 2025-03-17 → v3 2025-10-26 | "Step Repetition" (FM-1.3) in 15.7% | §4, App. A |
| Zhou et al., *WebArena* | 2307.13854 · 2023-07-25 → v4 2024-04-16 | halts at "the same action… repeated more than three times on the same observation" (T = 1.0) | §5.1 |
| He et al., *WebVoyager* (ACL 2024) | 2401.13919 · 2024-01-25 → v4 2024-06-06 | "Navigation Stuck" is 44.4% of failures; repeats "due to the input clipping" | §5.4 |
| Zhang et al., *MIRAGE-Bench* | 2507.21017 · 2025-07-28 | WebArena histories forked at the 3rd and 6th repeat; agents repeat again 0.21 (DeepSeek-reasoner) to 0.59 (Llama-3.1-70B) of the time (13 models, T = 0) | Table 6, A.6 ✓ |
| Bai et al., *StressWeb* | 2604.16385 · 2026-03-27 | identical consecutive actions in 58.4% of clean runs, 72.5–96.6% under perturbation | App. I |
| Xie et al., *OSWorld*; Wang et al., *OpenCUA*; Dong et al., *How Benchmarks Mis-Score Computer-Use Agents* | 2404.07972; 2508.09123; 2607.28367 | repeats after actions with no visible effect; "loops indefinitely"; feedback-blind repetition is 29.5% of genuine failures | secs |
| Zenkri & Brock, *Probing an Embodied LLM* (SAB 2026) | 2605.20072 · 2026-05-19 → v2 2026-10-01 | loops in ~80% of trials (r = −0.69 with success); flipping 40% of perceived outcomes cut loops and gave 2.85× success | abs, secs |
| Otani et al., *Do Agents Need to Plan Step-by-Step?* (CAIS 2026) | 2605.08477 · 2026-05-08 | "When a tool call fails or returns an empty result, models such as Gemini-3-Flash often issue the same tool call again in the next step with identical arguments": 30–45%, 66.9% under weak retrieval; plan-first agents 1.9–13.9% | §4.5 ✓ |
| Liu et al., *NetConfArena* | 2608.23179 · 2026-08-24 | thinking off → on, repeated actions: Qwen3-8B 0.496 → 0.124, Qwen3-32B 0.017 → 0.020; unparseable actions rise with thinking in every model; "premature abandonment" (T = 0, 20 turns) | full ✓ |
| Merrill et al., *Terminal-Bench* (ICLR 2026) | 2601.11868 · 2026-01-17 | "Step Repetition" in its taxonomy, with shares in figures only | secs |
| Backlund & Petersson, *Vending-Bench*; Xu et al., *TheAgentCompany* | 2502.15840; 2412.14161 | "meltdown" loops; Gemini-2.0-Flash "got stuck in a loop" | secs |
| Liu et al., *Budget-Aware Tool Use Enables Effective Agent Scaling* (BATS) | 2511.17006 · 2025-11-21 → v2 2026-08-17 | a per-turn budget display: SWE-bench Verified 36.8% → 55.4% (Gemini-3-Flash, 50 calls); ReAct stops "prematurely because it believes it has found a sufficient answer or is stuck" | App. C.3 ✓ |

**E. Hybrid thinking and templates**

| Work | ID · dates | What it shows | Overlap with us | Read |
|---|---|---|---|---|
| Wang et al., *Demystifying Hybrid Thinking* | 2510.12680 · 2025-10-14 → v2 2026-10-04 | no-think mode is the empty `<think></think>`; reasoning leaks into it (92–100% "implicit reasoning" for Qwen3) | (b) | full |
| Cuadron et al., *The Danger of Overthinking* | 2502.08235 · 2025-02-12 | analysis paralysis, rogue actions, premature disengagement; its rubric scores a loop as 0, "Stuck in a loop is good" | Q5; loops aren't overthinking | full ✓ |
| Qwen Team, *Qwen3 Technical Report* | 2505.09388 · 2025-05-14 | non-thinking samples "retain an empty thinking block"; non-thinking evaluation at T 0.7, top-p 0.8, presence penalty 1.5 | (b); temperature | §4.3 |
| Gemma Team, *Gemma 4 Technical Report* | 2607.02770 · 2026-07-02 → v2 2026-07-24 | models of 2.3B–31B; evaluations in thinking mode; the tool-call format; no sampling advice, nothing on loops | the setting | abs ✓, tables |
| Lei et al., *Thinking Inertia: LLMs Keep Thinking When Told Not To* (NeurIPS 2026) | 2610.11765 · 2026-10-08 | measures an "Empty-Thinking Rate" | (b) | abs |
| Usama et al., *Control-Token Injection Suppresses Chain-of-Thought…* | 2609.27542 · 2026-09-23 | a forged, closed empty reasoning channel stops gpt-oss from reasoning; "Two parsers shipped for the Gemma agent give opposite outcomes on identical greedy generations" | (b), (f) | secs |
| Ma et al., *NoThinking*; Zhang et al., *AdaptThink* | 2504.09858; 2505.13417 | prefilled empty or dummy thinking blocks change accuracy and length; repetition not measured | (b) | §3 |
| Muennighoff et al., *s1* | 2501.19393 · 2025-01-31 → v3 2025-03-01 | suppressing the end-of-thinking token too often "can lead the model into repetitive loops" | a risk for T-open | §4.2, §6.2 |

**F. Who decides when to think; confidence; help-seeking**

| Work | ID · dates | What it shows | Overlap with us | Read |
|---|---|---|---|---|
| Yang et al., *CogRouter* | 2602.12662 · 2026-02-13 | the depth that would "maximize the confidence of the resulting action"; the instinctive level rises from 48.5% to 94.8% of steps | (a): rewards a confident copy | main text |
| Ning et al., *HRBench* | 2605.28398 · 2026-05-27 | entropy-gated switching fails on "Confident wrong answers"; agentic switching "remain[s] unexplored" | (a), single-turn | most |
| Cheng et al., *When Should Agents Think? (RACE)* | 2610.12061 · 2026-10-08 | skips reasoning where removing it barely lowers the reference action's likelihood | (a) | §2–3 |
| Zhang et al., *Agentic Uncertainty Quantification* | 2601.15703 · 2026-01-22 | reflect only below 0.85 verbalised confidence; warns of "Delusional Confirmation" | (a) design | secs |
| Li et al., *Same Signal, Opposite Meaning* | 2605.06908 · 2026-05-07 | the entropy–benefit correlation flips sign across environments; "Wrong-direction gates can therefore worsen performance by precisely selecting harmful states" | (a) | secs |
| Kumaran et al., *Causal Evidence that Language Models use Confidence to Drive Behavior* | 2603.22161 · 2026-03-23 → v3 2026-09-18 | an implicit confidence threshold governs abstention (causal steering) | (a), single-turn QA | abs |
| Vijayvargiya et al., *Ambig-SWE* (ICLR 2026) | 2502.13069 · 2025-02-18 → v3 2026-02-21 | "Without explicit prompting, models almost never interact" | (a) help-seeking | §4.2 |
| Polezhaev et al., *Caddie* | 2610.09858 · 2026-10-07 | a fixed-step critic beat agent-initiated `call_critic` (25.94% vs 19.06%) | our advisor | secs |
| Kaddour et al., *Agentic Uncertainty Reveals Agentic Overconfidence* | 2602.06948 · 2026-02-06 | agents that succeed 22% of the time predict 77% | (a) | abs |
| Lin et al., *SwiftSage* (NeurIPS 2023) | 2305.17390 · 2023-05-27 → v2 2023-12-06 | hands control to the slow model when "Stuck" (5 zero-reward steps) or on an invalid action | a mechanical trigger | secs |
| Yang et al., *Ares*; Jali et al., *TAB*; Jung et al., *ALAR* | 2603.07915; 2604.05164 (v3 2026-08-08); 2606.02871 | trained per-step effort routers, with no confidence signal and no loops | background | full |

**G. Monitors and repair**

| Work | ID · dates | What it shows | Read |
|---|---|---|---|
| Khan & Khan, *The Cognitive Companion* | 2604.13759 · 2026-04-15 | a parallel monitor; −52% lexical repetition on one loop-prone prompt over two sessions (Gemma 4 E4B, T = 1.0, no tools); "a feasibility study" | full |
| Dubey, *Real-Time Detection and Repair of LLM Agent Failures* | 2608.02464 · 2026-08-03 | success 52% → 73% through deterministic checks; injected loops "escalate rather than recover"; a circuit breaker only shortens them | full |
| Sun et al., *Second Thought* | 2608.13667 · 2026-08-13 | parallel reasoning while acting: SWE-Bench Pro −0.7 to +3.3 (not significant), 6–11% fewer turns | full |
| Pham et al., *AgentStop* (CAIS 2026); Barazandeh et al., *OnTrack*; Nanda et al., *Wink* | 2605.15206; 2610.12375; 2602.17037 | early stopping and monitoring; misbehaviours, including "stuck in repetitive loops", in ~30% of trajectories (Wink) | secs / abs |

**H. Probability vs behaviour, and on-policy distillation (OPD)**

| Work | ID · dates | What it shows | Read |
|---|---|---|---|
| Yu et al., *Mismatch Matters: On-Policy Distillation Beyond Token Agreement* | 2608.09836 · 2026-08-10 | students "exploit repetitive loops": repetitive rollouts rise from 16.8% to 48.4%; inside loops, top-1 agreement is 100% and the median KL is 0.0004 nats; 16 repeats cut the KL 63×; the teacher continues the student's loops 93% of the time (math) | §2.2, App. C.1 ✓ |
| Xin et al., *Escaping the KL Agreement Trap in On-Policy Distillation* | 2606.09471 · 2026-06-08 | the teacher "may locally agree with the degraded state"; over 60% of rollouts hold a long low-KL segment | §1, §3–4 |
| Shen et al., *Tool-Call Drift* | 2607.07050 · 2026-07-08 → v6 2026-09-05 | top-32 logits hold 0.9999 of the mass yet contain `<tool_call>` in 0.4% of response prompts; Loop@3 15.1% → 5.1% with mixed SFT | main text |
| Zhao et al., *OPSD* | 2601.18734 · 2026-01-26 → v3 2026-03-20 | per-token KL is skewed toward style tokens (0.85 vs 0.14 for math tokens); its default pairing differs by the empty block | full |
| Su et al., *PSP* | 2609.29051 · 2026-09-24 | OPSD "teaches the student to act with confidence but without the information behind it"; below the base on SWE-bench Verified with thinking off; Gemma4-E4B collapses on AppWorld (14.6 → 0.7) | full |
| Gui et al., *Know When to Stop, Where to Restart* (STRIDE) | 2609.14636 · 2026-09-13 | faster multi-turn OPD training; nothing about agent loops | full |
| SeOPD; ThinkOPD; HERO; SDPO | 2609.33181; 2609.37044; 2606.11559; 2601.20802 | self-teachers from thinking or feedback. SeOPD: without its CoT the model writes "repetitive text… to mask its uncertainty", the reverse of (a) | full |
| Gonuguntla, *The Replay Gap* (COLM 2026) | 2608.08239 · 2026-08-08 | 74–77% of model swaps diverge at the first action after a fork; FP8 vs AWQ serving matters | abs |
| Ettifouri et al., *Calibrated e-CUSUM Decoding for Quantized Reasoning Models* | 2607.11317 · 2026-07-13 | log-probability is "nearly silent during confident repetition" (INT4) | abs |

**I. Lenient harnesses and silent failures**

| Work | ID · dates | What it shows | Read |
|---|---|---|---|
| Sethi et al., *Fabrication After Tool Failure* | 2609.14758 · 2026-09-13 | an `ok` status with an empty or stale payload gives 45.3% dishonest answers, vs 0% with `error` | abs |
| Zhao et al., *PRISM* (EMNLP 2026) | 2609.05736 · 2026-09-04 → v2 2026-09-09 | treats "silent correction, rewriting malformed arguments before execution" as a legitimate pattern; what later calls copy is unmeasured | §4.3 |
| An agent-runtime bug study | 2607.15684 · 2026-07-17 | "many AR bugs manifest as silent errors" | abs |

**J. Documentation, code, issues and forums (non-archival)**

| Source | What it says | Read |
|---|---|---|
| Google, Gemma 4 model card | "temperature=1.0, top_p=0.95, top_k=64" "across all use cases"; 12B–31B with thinking off "will still generate the tags but with an empty thought block" | ✓ |
| Google, Gemma 4 prompt formatting (updated 2026-06-03) | larger models "may occasionally generate a thought channel even when thinking mode is explicitly turned off… consider adding an empty thinking token"; long-running agents may keep their reasoning "to prevent the model from entering cyclical reasoning loops" | ✓ |
| Google, Gemma 4 thinking guide (2026-06-04) | the empty thinking token added to the 12B, 26B and 31B templates "stabilizes model output by suppressing 'ghost' thought channels" | ✓ |
| The Gemma 4 chat templates on Hugging Face | the competition model ships the June-5 template; Google's July template (merged 2026-07-15) is headed "Fixed tool-calling loops", yet with thinking off it still adds nothing after a tool response (per the template and HF PR #147) | secs |
| HF PR "Emit an empty thought after a tool response when enable_thinking=false" (gemma-4-31B-it #147, gemma-4-26B-A4B-it #67; opened ~2026-09-30, open) | our edit. On 26B FP8 only: self-opened thoughts 80/80 → 0/80 and 229/300 → 0/300; empty replies 25/735 → 0/1,035; tool calls unchanged. No Google reply | ✓ |
| Qwen3-8B chat template | tool results go in a user turn; with `enable_thinking=false`, every generation prompt gets the empty think block, after tool results too | ✓ |
| Qwen3 model cards (3, 3.5, 3.6, 3.8, Coder) | the greedy-decoding warning; presence_penalty "to reduce endless repetitions"; non-thinking settings | secs |
| OpenHands StuckDetector (V0 at commit 6e4ac8e, 2026-01-15; V1 SDK at c8a65f0, 2026-08-04) | 4 identical action–observation pairs; 3 identical errors; a 3-message monologue; an alternating pattern over 6 steps; V1 adds one nudge after 3 identical errors (PR #4332, from issue #4331's 7/107 GLM-5.2 runs) | ✓ (V0) |
| Gemini CLI `loopDetectionService.ts` (acae712, 2026-07-17) | 5 identical (name, args) calls, or a cycle of up to 5 repeated 5 times; a content chunk repeated 10 times; an LLM judge from turn 30; the first detection injects "Potential loop detected… Please take a step back", the second ends the turn | ✓ |
| Cline, Roo Code, Qwen Code, OpenRouter's doom-loop docs, SWE-agent, Aider, LangGraph, Claude Code | warn at 3, stop at 5 (Cline); block the 4th (Roo); 5 identical calls, plus DashScope's server-side "Repetitive tool calls detected" (Qwen Code); observe at 2, block at 3, stop at 6 (OpenRouter); autosubmit at the cost limit (SWE-agent); only step caps (LangGraph; Claude Code's `--max-turns`) | secs |
| vLLM #40080 (2026-04-17), #51284 (2026-08-06), #39468 (2026-04-10) | Gemma 4 loops and the fix with three penalties; JSON strings in arrays → the identical retry; the string delimiter leaking into arguments | ✓ (first two) |
| HF gemma-4-12B-it #41 (2026-06-18); llama.cpp #20164 (2026-03-06); NVIDIA forum, Qwen3.6-35B-A3B thread | a looping thought channel at T = 0, 0.7, 1.0 (Google: "we have a fix coming"); Qwen3.5, 3.6 and Coder-Next repeat the same call "dozens of times" | secs |
| Kaggle main track 745774 (Hitarth Jain, 2026-10-04) | "repeats the same command up to 69 times"; T = 0.7 worse; "looping isn't a sampling-temperature artifact"; a lower call budget beat prompt wording | ✓ |
| Kaggle 744844 (Amanie El-Hajj, 2026-10-01) | the competition serves the June-5 template; "Loops and malformed tool calls are a large share of failures for every agent here"; staff: "I will investigate!" (2026-10-12) | ✓ |
| Kaggle 746250 (Ayush Thakur, 2026-10-06) | thinking on, T = 0.0 / 0.2 / 0.4 / 1.0 → 47 / 48 / 51 / 51 of 129; thinking disabled: 23 | ✓ |
| Kaggle 748287 (staff, 2026-10-11) | allowed: 31B, 12B, E4B and E2B `-it-qat-w4a16-ct` | secs |

## 5. The must-cite list

**20 core references, in the order the paper would use them.** Each line says what to cite it for.

1. **Xu et al. 2022:** lock-in is self-reinforcement at the action level.
2. **Pipis et al. 2026:** loops at low temperature, and "temperature is a stopgap".
3. **Sinha et al. 2026:** self-conditioning, and Qwen3's thinking removing it. Our contrast: thinking doesn't break lock-in.
4. **Wan et al. 2026:** conversational inertia, and clipping vs summaries.
5. **Gumaan 2026:** a failed call's surface form, the levers on it, and clean restart as the worst harness. The closest work for (c), (e) and (f).
6. **Liu et al. 2026 (2610.02769):** p(3rd identical action), outcome labels, and no gain in success.
7. **Yang et al. 2024, SWE-agent:** interface guards, and what autosubmit leads to.
8. **Pan et al. 2025, SWE-Gym:** the "Stuck in Loop" metric, to report on our runs.
9. **Yang et al. 2025, SWE-smith:** prevalence, and repeats as "symptoms". Prior work for (e).
10. **Yuan et al. 2026:** Gemma 4 31B non-thinking loops in another harness.
11. **Liu et al. 2026, NetConfArena:** thinking off vs on, and repeated actions in Qwen3.
12. **Zhang et al. 2025, MIRAGE-Bench:** a replay test at the 3rd and 6th repeat.
13. **Cuadron et al. 2025:** premature disengagement; its rubric treats loops as separate from overthinking.
14. **Wang et al. 2025, Demystifying Hybrid Thinking:** the no-think mode as an empty block, and leakage.
15. **Google, Gemma 4 documentation:** the model card's sampling, ghost thought channels, and the empty thinking token.
16. **Qwen Team 2025, Qwen3 Technical Report:** the empty think block in non-thinking mode.
17. **HF PR google/gemma-4-31B-it #147:** the same template edit as our always-block run.
18. **Yu et al. 2026, Mismatch Matters:** KL is blind inside loops. Prior work for (d).
19. **Yang et al. 2026, CogRouter:** a router built to maximise action confidence. Contrast for (a).
20. **The production loop detectors:** OpenHands' StuckDetector and Gemini CLI's loop detection service.

**Add these if the word count allows (5):**

21. Vijayvargiya et al. 2026, Ambig-SWE (help-seeking, for (a)).
22. Kaggle forum thread 745774 (the same loops in this competition).
23. Gemma Team 2026, Gemma 4 Technical Report.
24. Wang et al. 2026, Fail-Fast, Restart-Smart (restarts on Gemma 4 31B, for Q5).
25. Holtzman et al. 2020 (the classic reference for repetition).

**Next in line:** Huang et al. 2024, Tsui (Self-Correction Bench), Tool-Call Drift, Otani et al., the KV-compaction study, The Replay Gap.

**The references:**
- **arXiv-only preprints and conference papers without a DOI:**
  - Pipis, C. et al. *Wait, Wait, Wait… Why Do Reasoning Models Loop?* ICML 2026. arXiv:2512.12895, https://arxiv.org/abs/2512.12895
  - Sinha, A. et al. *The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs.* ICLR 2026. arXiv:2509.09677, https://arxiv.org/abs/2509.09677
  - Wan, Y. et al. *Mitigating Conversational Inertia in Multi-Turn Agents.* ICML 2026. arXiv:2602.03664, https://arxiv.org/abs/2602.03664
  - Gumaan, E. *Feedback That Backfires: Why Small Language Model Agents Repeat the Call They Just Watched Fail.* arXiv:2608.23651, https://arxiv.org/abs/2608.23651
  - Liu, J. et al. *When History Fails to Become Experience: Action Calibration in Language Agents.* arXiv:2610.02769, https://arxiv.org/abs/2610.02769
  - Yang et al. *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS 2024. arXiv:2405.15793, https://arxiv.org/abs/2405.15793
  - Pan, J. et al. *Training Software Engineering Agents and Verifiers with SWE-Gym.* arXiv:2412.21139, https://arxiv.org/abs/2412.21139 (the arXiv page shows no venue)
  - Yang et al. *SWE-smith: Scaling Data for Software Engineering Agents.* arXiv:2504.21798, https://arxiv.org/abs/2504.21798
  - Yuan, Y. et al. *Agentic AI Workload Characterization.* arXiv:2605.26297, https://arxiv.org/abs/2605.26297 (IISWC 2026 per its artifact links)
  - Liu, C. et al. *NetConfArena: An Executable Benchmark for LLM Agents in Closed-Loop Network Configuration.* arXiv:2608.23179, https://arxiv.org/abs/2608.23179
  - Zhang, W. et al. *MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them.* arXiv:2507.21017, https://arxiv.org/abs/2507.21017
  - Cuadron, A. et al. *The Danger of Overthinking: Examining the Reasoning-Action Dilemma in Agentic Tasks.* arXiv:2502.08235, https://arxiv.org/abs/2502.08235
  - Wang, S. et al. *Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?* arXiv:2510.12680, https://arxiv.org/abs/2510.12680
  - Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, https://arxiv.org/abs/2505.09388
  - Gemma Team. *Gemma 4 Technical Report.* arXiv:2607.02770, https://arxiv.org/abs/2607.02770
  - Yu, Z. et al. *Mismatch Matters: On-Policy Distillation Beyond Token Agreement.* arXiv:2608.09836, https://arxiv.org/abs/2608.09836
  - Yang, R. et al. *Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents* (CogRouter). arXiv:2602.12662, https://arxiv.org/abs/2602.12662
  - Vijayvargiya et al. *Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering.* ICLR 2026. arXiv:2502.13069, https://arxiv.org/abs/2502.13069
  - Wang, C. et al. *Fail-Fast, Restart-Smart: Early Failure Prediction and Restart for SWE Agentic Tasks.* arXiv:2608.03222, https://arxiv.org/abs/2608.03222
  - Holtzman, A. et al. *The Curious Case of Neural Text Degeneration.* ICLR 2020. arXiv:1904.09751, https://arxiv.org/abs/1904.09751
- **With a DOI (resolved through LibKey):**
  - Xu, J. et al. *Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation.* NeurIPS 2022, pp. 3082–3095. DOI 10.52202/068431-0223. Full text not available through your institution. Open copy: https://arxiv.org/abs/2206.02369
  - Kamoi, R., Zhang, Y., Zhang, N., Han, J., Zhang, R. *When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs.* TACL 12 (2024), 1417–1440. DOI 10.1162/tacl_a_00713 (next in line, not core).
    🔗 [**Access Options**](https://onesearch.library.northeastern.edu/openurl/01NEU_INST/01NEU_INST:NUL?genre=article&aulast=Kamoi&issn=2307-387X&title=Transactions%20of%20the%20Association%20for%20Computational%20Linguistics&atitle=When%20Can%20LLMs%20Actually%20Correct%20Their%20Own%20Mistakes%3F%20A%20Critical%20Survey%20of%20Self-Correction%20of%20LLMs&volume=12&issue=&spage=1417&epage=1440&date=2024-11-04&rft_id=info:doi/10.1162%2Ftacl_a_00713&rft_id=info:pmid/&sid=LibKey)
- **Non-archival:**
  - Google AI for Developers, *Gemma 4 model card*, *Prompt formatting for Gemma 4* and *Thinking*: https://ai.google.dev/gemma/docs/core/model_card_4, https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4, https://ai.google.dev/gemma/docs/capabilities/thinking
  - The Hugging Face PR: https://huggingface.co/google/gemma-4-31B-it/discussions/147 (and https://huggingface.co/google/gemma-4-26B-A4B-it/discussions/67)
  - OpenHands StuckDetector: https://github.com/OpenHands/OpenHands/blob/6e4ac8e2cec9c8c0a0f640f511916dc3fc294e89/openhands/controller/stuck.py
  - Gemini CLI loop detection: https://github.com/google-gemini/gemini-cli/blob/acae7124bdd8/packages/core/src/services/loopDetectionService.ts
  - Kaggle thread 745774: https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745774

## 6. Implications

### For the two running experiments

**T-open (the forced thought)**
- **What the literature predicts is mixed:**
  - Thinking removes self-conditioning in Qwen3 (Sinha), and it cuts Qwen3-8B's repeated actions 4× (NetConfArena).
  - Reflection only partly lowers anchoring (Verma).
  - Forcing thinking to continue can itself loop (s1), and Gemma 4 31B loops more at the token level with thinking on (Lazaridis).
- **Add to its report.** All of these are CPU-only, from `gen_T-open.jsonl`:
  - degenerate thoughts: repeated n-grams and the "tight loop", since `min_tokens` holds `<channel|>` back for 32 tokens;
  - MIRAGE-Bench's awareness score: does the thought name the repetition (0 / 1 / 2)?
  - the class of the sampled call (stale, fresh, malformed), next to p(call). Claim (d) says the class is the honest readout.
- **Reading it:**
  - If T-open escapes far more often than T-think on the states where T-think skipped the thought, then (a) holds ("it can, but doesn't choose to"). That would extend Sinha's result from wrong answers to tool-call loops.
  - If T-open doesn't escape, (c) holds, as Gumaan's and Verma's results would predict.

**The always-block run**
- **It is the HF PR #147 edit**, and Google's docs advise it too. Cite both. Our contribution is the effect on loops, over whole tasks, at 31B.
- **The prediction from the literature:**
  - Probe effects shrink in closed loops: Gumaan's log-prob effect nearly vanishes, yet repeats fall only from 31% to 16%; Liu cuts repeats with no gain in success.
  - So expect fewer copy starts, partial lock-in, and a resolve rate within noise.
- **Add three metrics to `loop_report.py`:**
  - SWE-Gym's "3 identical in a row" share and SWE-smith's share of runs of 10+, so judges can compare with published rates;
  - the PR's two counts: completions that open a thought after a tool response, and replies with empty content;
  - what each escape led to: an empty patch, a wrong patch, or a resolved task.
- **Check the July template, on the CPU:** render the recorded `off_args` conversations with Google's July template and diff them against the June one.
  - If the thinking-off prompts are identical, as PR #147 implies, say so in one line.
  - If they differ, a July-template arm is missing.

### For the paper's framing

- **The contribution:** why a thinking-off agent locks into repeated tool calls, and which lever breaks the lock.
- **The order of the claims:**
  1. the lock-in hazard;
  2. the four levers at the same states;
  3. self-selection ("a stuck agent is a confident agent");
  4. the ceiling.
- **Positioning:**
  - Name the five closest works and say what each lacks:
    - Gumaan: ≤1.7B models, explicit failures, greedy decoding;
    - Liu: no thinking switch, and no closed loop for the levers;
    - Wan: no tool-call slips;
    - Sinha: a synthetic task;
    - SWE-smith: no counterfactual ceiling.
  - Yuan shows the same model loops in another harness. Use it for generality.
- **A practical result for harness builders:**
  - Lock-in starts at the 2nd copy. Gemini CLI waits for the 5th and OpenHands for the 4th, so the loop runs on for 2–3 more copies before they act.
  - The replay also says what to inject: dedup, then a note or the block, then thinking.
- **The OPSD section:** our 0.51 ratio is a tool-call instance of Mismatch Matters' blind KL, and of Tool-Call Drift's warning about probability mass.

### Risks

1. **(c) and (e) are partly taken.** Cite Gumaan, Liu, Wan and SWE-smith in the introduction, not only in related work. The new parts are the head-to-head at the same states and the counterfactual ceiling.
2. **Gumaan finds clean restart the worst harness,** while our dedup gives 0.14. Our dedup keeps the latest copy and its result ([../gate1/gate1.py](../gate1/gate1.py), `dedup`), so the context still differs from the pre-loop one. We also sample at T = 0.2, while he decodes greedily. State both.
3. **Liu's p(3rd identical action) is 37% pooled; our lock-in is 83–94%.** Liu's histories were screened for feedback that argues against repeating, the models and environments differ, and their decoding isn't stated. Present ours as this model in this mode, and report our number under Liu's definition: p(3rd | 2 identical) is our k = 2 lock-in.
4. **Temperature.** Our runs sample at T = 0.2, against Google's T = 1.0. Run one closed-loop arm at T = 1.0.
5. **One model** costs on the Quality score. Add a second model (Q2).
6. **The template may change under us:**
   - Our numbers rest on the June-5 template that the competition serves.
   - The host is investigating a template change (744844), and PR #147 may be merged before 2026-11-12.
   - State the template's hash in the paper. If Google adopts the PR, our always-block run measures the new default.
7. **Replay validity:**
   - One-step replays can mislead: model swaps diverge at the first forked action 74–77% of the time, and the serving format matters (The Replay Gap).
   - Keep the closed-loop runs as the test.
   - Note int4. Verma found Q4 retries behave as FP16; the OpenHands LM blog notes sensitivity to quantization.
8. **Others may write the same paper.** The authors of 745774 and 744844 hold loop data, and the PR's author holds template data. Submit a complete version early, and cite the threads.
9. **"Confident" is measured indirectly.** We measure it as the thinking rate, not as confidence. Add the confidence analysis below, or soften the word.

### Experiments a judge will expect

Ranked by value for cost:

1. **CPU only, on existing data (hours):**
   - SWE-Gym's and SWE-smith's loop metrics, and Liu's p(3rd | 2), on our runs;
   - what each escape led to;
   - confidence vs the choice to think: S's p(copy), and the next-token entropy at the call's first token, against T-think's thought or no thought (AUROC). That turns (a) into a direct measurement;
   - the July-template render diff.
2. **A closed-loop `off_args` arm at T = 1.0, top_p 0.95, top_k 64** (Google's configuration). One run. It answers the temperature question with closed-loop numbers.
3. **A second model:** Gemma 4 12B (the template prefills the block, as for the 31B) and E4B (no block), through swegemma v30. 1–2 runs each.
4. **A closed-loop version of the replay's strongest lever:** dedup in the token proxy. Optionally add a production-style nudge, Gemini CLI's two-strike message, fired at the 2nd identical call. This needs proxy work and 1–2 runs.
5. **A second seed of `block` and `ctrl`** (`PASS=2`).
6. **Optional: Qwen3-8B with thinking off,** which is always-block by construction. The harness port is the cost.

## 7. Sources

**How the review was done (2026-10-12)**
- **The search.** Six research agents ran in this session, each on one area:
  - the works we cite, in two groups;
  - agent failure studies across domains;
  - detectors, model docs, issue trackers and forums;
  - mechanisms and fixes;
  - a sweep of 2025–Oct 2026 arXiv.
- **What they read:** arXiv abstract pages for metadata and versions, arXiv HTML for full text, GitHub source at the commits named, Hugging Face pages, Google's docs, and Kaggle's forum through its JSON API.
- **Read levels** are in §4.
- **What was re-checked a second time (✓):**
  - NetConfArena's Table II; Cuadron's rubric; Gumaan's §1–2 and §8; Liu's Table 3; Wan's §3 and App. Q;
  - SWE-smith's §4.2 and F.5.4; SWE-Gym's loop definition and rates; MIRAGE-Bench's Table 6 and A.6;
  - Mismatch Matters' §2.2; Fail-Fast's Gemma4-31B rows; Shu's search-loop rates; TACT's Gemma numbers; Lazaridis' thinking-mode result; Otani's §4.5;
  - Pipis' abstract and §5; Yuan's turn counts; the KV-compaction study's Table 3; BATS' App. C.3; the Gemma 4 report's metadata;
  - the Gemma 4 docs' quotes; the Qwen3-8B template; the HF PR; vLLM #40080 and #51284;
  - the thresholds of OpenHands V0 and Gemini CLI;
  - Kaggle 745774, 744844 and 746250, read from the saved JSON.
- **DOIs:**
  - Every DOI listed was checked against the doi.org handle registry.
  - LibKey resolved Kamoi et al.
  - LibKey has no record of the others: Xu (10.52202/068431-0223), SwiftSage (10.52202/075280-1034), Dong et al. (10.18653/v1/2025.acl-long.48), Word Salad Chopper (10.18653/v1/2025.emnlp-main.1705), Repetition Neurons (10.18653/v1/2025.naacl-short.41), Otani et al. (10.1145/3786335.3813129) and Zenkri & Brock (10.1007/978-3-032-39953-3_25). Full text is not available through your institution for these. Their open copies are on arXiv or the ACL Anthology.
- **arXiv:** every ID in §4 resolves at `https://arxiv.org/abs/<ID>`.

**Unverified, or seen but not opened**
- **Unverified claims:**
  - SWE-Gym's ICML 2025 venue: the arXiv page doesn't show it.
  - BATS' COLM 2026 venue.
  - The ICSE 2026 venue of Majgaonkar et al. (2511.00197).
  - The repetition shares of Terminal-Bench, TRAIL and AgentBench's App. J.2.4, which are in figures only.
  - That Codex CLI has no loop detector.
  - A blog's "48× more loops at T = 0" (no source found).
- **Do not cite:** 2505.13514, withdrawn in v2.
- **Abstract only; read before citing:** 2607.01641 (infinite agentic loops, by static analysis), 2603.04783, 2610.08722, 2606.29490 (confidence vs commitment, including Gemma 4), 2609.28682, 2608.06811, 2608.06701, 2610.04429 (asking in BFCL, including gemma-4-31B-it), 2610.10833, 2606.00198, 2607.25152, 2610.13238, 2609.03966, 2610.00917, 2610.05533, 2610.05348.
- **Seen in search results only:** 2602.01725, 2607.00692, 2609.34649, 2608.10676; Liquid AI's "Antidoom" post; the doom-loop guards of Kilocode, Hermes and OpenCode.
