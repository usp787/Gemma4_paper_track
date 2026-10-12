# Start prompt: the literature review of the loop study

<!-- drafted 2026-10-12 for a new Claude Code session in this repo. Paste everything below the line, or start the session with "Read loops/lit_review_prompt.md and follow it." -->

---

You are starting the literature review for our Kaggle paper. You have none of the earlier conversations, so this prompt and the files it names are your context. Today is 2026-10-12.

## The task

Our paper is being re-scoped from on-policy self-distillation (OPSD) to **the loop behaviour of a Gemma 4 coding agent's failed rollouts**: with thinking off, it re-sends the same tool call again and again. Before we commit to that scope, find out what is already known.
- Write `loops/literature.md`.
- Then answer the five questions at the end of this prompt in the chat, briefly.
- Then stop. The user discusses every step and will ask follow-up questions.

Don't run experiments, queue jobs, commit or push. The user commits by hand.

## The paper's constraints

- **Venue:** Google's "Gemma 4 Developer Agent" paper track on Kaggle. The writeup is at most **3,000 words**. Until the host says otherwise, count the title, abstract, captions and references. The deadline is **2026-11-12**.
- **Judging:** five criteria, scored equally:
  - Novelty: "new insights, deepen understanding, or highlight important properties of existing methods";
  - Quality, which the rubric defines as generality beyond the competition;
  - Relevance to software engineering and agentic learning;
  - Verifiability;
  - Clarity.
- **So a must-cite list has to be short:** about 15–25 references.
- More on the track: [README.md](../README.md).

## What we found (our own data; you may quote these numbers)

**Setting:**
- Gemma 4 31B, int4 QAT (`gemma-4-31b-it-qat-w4a16-ct`), served by vLLM 0.19 in the host's harness (wheelhouse v30).
- 70 dev tasks: fastapi 60, requests 9, httpx 1.
- A 4-minute clock and 60 tool calls per task; thinking off; T = 0.2.

**The arms:**

| | `off` (the host's prompt) | `off_args` (+ rules for tool-call arguments) | `on` (thinking on) |
|---|---|---|---|
| Resolved of 70 | 22 | 24, then 22 | 28 |
| Stale repeats | 45.0% | 28.2% | 7.3% |
| Malformed calls | 21.1% | 1.4% | 5.8% |
| P(bad call \| the previous one bad) | 86.5% | 80.4% | 53.8% |

**Findings:**
- **Anatomy (`off_args`):**
  - 88% of the stale repeats left are identical `run_command` calls: a `grep` or `find` that returns nothing or the same lines, re-sent up to 53 times.
  - A loop starts when a search comes back empty. Variants of it follow, then the agent returns to an earlier call; 44% of chains re-send a call that compaction had removed from view.
  - Lock-in: after two identical calls in a row, every further copy follows 83–94% of the time.
  - Escapes come from outside. Of 23 runs of 3+ identical calls, 8 end at a context compaction, 7 at the harness's budget warning (after which the model submits, mostly an empty patch), 5 on their own, and 3 with the task.
- **Copying a format slip:**
  - Most of `off`'s malformed calls are one slip: JSON written inside Gemma 4's tool-call syntax, such as a key `start_line"` with a stray quote.
  - The harness drops the bad key silently and returns `ok`.
  - The first slip is rare (6% of reads with a line range), but 82% of the ranged reads after it repeat it.
  - Prompt rules that prevent the first slip cut malformed calls from 21.1% to 1.4%.
- **The replay ("Gate 1"):** 581 recorded states, each re-scored one call ahead in an edited context. At the 3rd or later identical copy, the probability that the model copies again is:
  - 0.94 for the student itself;
  - 0.14 with the earlier copies removed from the context;
  - 0.49 with a controller's note;
  - 0.50 with the 4-token empty thought `<|channel>thought\n<channel|>` forced before the call;
  - 0.80 when the same weights may think first.
- **A stuck agent is a confident agent (our working claim):**
  - Allowed to think, the model writes a real thought at 72% of 2nd copies, 23% of 3rd+ copies and 6% of malformed calls.
  - On the looping states where it skipped the thought (114), its own call was stale 89% of the time. Where it thought (99), it escaped 62% of the time.
  - An advisor sub-agent that the model can call itself was called before the first edit in 52 of 67 cases. Right after a bad call, the next call was the advisor 1.1% of the time.
- **Probability moves more than behaviour:**
  - Forcing the empty thought halves p(copy), yet 80% of the calls sampled at 3rd+ copies stay stale.
  - On 116 looping states, the sampled call escapes 28% of the time with the forced empty thought (thinking off), 32% with it after the thinking-on prompt, and 13% when the model chooses itself.
- **Drift out of the empty thought:** with thinking off, the model writes the empty thought itself at 82% of calls 1–5 and at ~35–45% after call 10, and once it skips it, it keeps skipping (78%). The template prefills it only after the task prompt.
- **Loops are a symptom:**
  - Of 23 failures with at least 20% stale calls, only 6 were solved by any of 9 other runs, so removing every loop would gain about 2 of 70 tasks. One run varies by ±2.2 tasks.
  - When the argument rules removed most malformed calls, failures moved from loops (21) to wrong patches (`off_args`: 7 loops among 46 failures).
- **For OPSD:** the model's own thinking as a teacher put less KL on the failure tokens than on routine ones (ratio 0.51). The only teacher that passed got its signal from the 4-token empty thought.
- **Running now** (results will come later):
  - a forced real thought at every state (`T-open`);
  - a closed-loop run that prefills the empty thought at every call, with a paired control, on the 70 dev tasks plus 39 rich tasks. See [loops/README.md](README.md).

**Where the detail is:** [gate1/README.md](../gate1/README.md), [baseline/README.md](../baseline/README.md), [adaptive_inference/README.md](../adaptive_inference/README.md) and [opsd_assessment.md](../opsd_assessment.md). Sections 1.5 of the last two hold the earlier literature tables, read 2026-10-08, mostly from abstracts.

## The five questions (the user's own)

1. Has the loop behaviour we met already been discovered? What is the progress on it, and what challenges remain?
2. Can such behaviour be reproduced stably in Gemma 4, and in other open models such as Qwen?
   - Answer from the literature: which models, sizes, modes and settings show it.
   - Add a short feasibility note on reproducing it ourselves, without running anything. Our harness is the host's `swegemma` (v30 added "support quantized e2b/e4b/12b models"). Qwen3's no-think mode also prefills an empty `<think>\n\n</think>`.
3. Does it occur outside coding: web and embodied agents, multi-turn tool use, terminal tasks, long-horizon execution?
4. What are the reasons behind it, and the methods for resolving it when it isn't what we want? For example, letting rollouts try different tool calls: action-level repetition penalties, resampling with the repeated call excluded, exploration in agent RL. Also loop detectors and nudges, context editing, forced reflection or thinking, and training-time fixes.
5. Is breaking out of a loop good, bad or uncertain? What does an escape lead to? Our own escapes at the budget warning mostly submit an empty patch.

## Key points for the review

**The claims to check for novelty.** For each, give a verdict (taken / partly taken / open) and the closest works:
- (a) confidence gates self-directed thinking or help-seeking in agents ("a stuck agent is a confident agent");
- (b) the empty-thought block of hybrid-thinking templates (Gemma 4, Qwen3) changes repetition and loop behaviour;
- (c) editing the context (removing copies, a note) breaks loops better than letting the model think;
- (d) shifts in token probability or teacher–student KL overstate the change in sampled behaviour. This matters for OPD/OPSD diagnostics;
- (e) loops are a symptom: removing them buys few solved tasks;
- (f) a format slip, silently accepted by a lenient harness, is copied by later calls (self-conditioning).

**Separate the phenomena.** The literature names them differently:
- token-level degeneration (text repetition);
- action-level repetition (identical tool calls);
- semantic loops (variants of the same failed search);
- copying of one's own errors (self-conditioning).

Map our findings onto these terms. Suggest what to call ours, so that it matches the literature and searches find it.

**Temperature.** Our runs sample at T = 0.2, while Gemma's own generation config says T = 1.0, top_p 0.95, top_k 64. Low temperature is a known cause of repetition, so judges may ask whether this is just low temperature. Our replay says no: at 3rd+ copies p(copy) is 0.73 at T = 1, though only 0.26 at 2nd copies. Find what the literature and Google's Gemma 4 documentation say about this.

**Start from what we already cite.** Check that each entry exists and says what we claim, and correct any that doesn't:
- Holtzman et al. (ICLR 2020), text degeneration;
- Xu et al. (NeurIPS 2022), "Learning to Break the Loop";
- Sinha et al. (ICLR 2026), self-conditioning in long-horizon execution;
- Huang et al. (ICLR 2024) and Kamoi et al. (TACL 2024), self-correction without outside feedback;
- loop monitors and repair:
  - Cognitive Companion, arXiv 2604.13759, a loop monitor on Gemma 4 E4B;
  - Real-Time Detection and Repair, 2608.02464;
  - Second Thought, 2608.13667;
- hybrid thinking and overthinking:
  - "Demystifying Hybrid Thinking", 2510.12680;
  - "The Danger of Overthinking", 2502.08235, whose failure modes include premature disengagement;
  - NetConfArena, 2608.23179;
- per-step effort routers:
  - Ares, 2603.07915;
  - CogRouter, 2602.12662;
  - TAB, 2604.05164;
  - ALAR, 2606.02871;
  - SwiftSage, 2305.17390;
  - HRBench, 2605.28398;
- "Know When to Stop, Where to Restart", 2609.14636 (seen, never opened);
- the OPSD family: OPSD 2601.18734, PSP 2609.29051, SeOPD 2609.33181, ThinkOPD 2609.37044, HERO 2606.11559, SDPO 2601.20802, and Tool-Call Drift 2607.07050.

Look beyond these too:
- SWE-agent and OpenHands trajectory and failure studies;
- the loop and stuck detection of OpenHands and Gemini CLI;
- multi-agent failure taxonomies, which list step repetition;
- the Gemma 4 and Qwen3 technical reports and model cards;
- the Kaggle main-track forum.

**Evidence rules:**
- Read at least the abstract of everything you cite, and the full text or the key tables of the 5–8 closest works.
- Record each work's ID, date and version, and whether you read it in full or the abstract only.
- Never cite from memory. Mark anything you couldn't verify.
- arXiv IDs in this field run to 2609.xxxxx (September 2026), so search recent work.

**References (an organization rule):**
- Resolve every article that has a DOI through the LibKey connector (Scholarly Full Text Article Access with LibKey) before you list it.
- For arXiv-only preprints, give the arXiv ID and URL.
- Present only the final, verified metadata and access links. Don't narrate lookups or searches.

**Tools:** WebSearch, WebFetch, the Scholar Gateway semantic search and LibKey. Read-only access to Explorer is possible ([explorer-access] in memory) but shouldn't be needed.

**When you're done:** add one or two lines to the memory note `paper-pivot-loops` with the verdict.

## What `loops/literature.md` should hold

1. **The bottom line:** at most 10 bullets, including whether the loop study is novel enough to carry the paper.
2. **Answers to Q1–Q5,** each with the literature's evidence next to ours.
3. **A novelty verdict for claims (a)–(f),** with the closest works.
4. **A table of works:** ID, date, what it shows, its setting (model, domain), its overlap with us, and how fully you read it.
5. **The must-cite list** for a 3,000-word paper (15–25 entries).
6. **Implications:** for the two running experiments and the paper's framing, the risks you see, and the experiments the literature says a judge will expect.
7. **Sources.**

Keep it skimmable: short bullets, tables where they help, and one idea per bullet.
