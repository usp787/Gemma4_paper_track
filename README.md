# Gemma4_paper_track

Our entry for **[Google – The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)** (Kaggle hackathon, host: Google DeepMind). It is the research-paper companion to the [main leaderboard competition](https://www.kaggle.com/competitions/gemma-4-developer-agent). Our agent for the main competition lives in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo).

## Goal

Submit a **Kaggle Writeup of at most 3,000 words** that reports original, unpublished research advancing agentic software engineering.

**Our direction (decided 2026-10-07):** on-policy self-distillation (OPSD) of Gemma 4 31B into a coding agent that runs with thinking off. See [Primary direction](#primary-direction-opsd-for-a-thinking-off-agent).

- The main leaderboard counts the issues an agent fixes. The paper track asks for the "how and why" ([host welcome, 743012](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743012)).
- Both independent research and a writeup of a main-competition approach qualify.
- Entering the main competition isn't required. The paper track is a separate Kaggle competition, so you have to join it and accept its rules.
- Field on 2026-10-07: Kaggle's API counted 3,863 joined users, 158 teams and 162 submissions.

## Key dates (UTC)

| Date | Event |
|---|---|
| 2026-09-22 | Start date |
| **2026-11-12 23:59** | **Paper deadline.** Also the last day to join the track or merge teams. Drafts and un-submitted writeups are not judged. |
| 2026-11-25 23:59 | Main track: entry and team-merger deadline |
| 2026-12-02 23:59 | Main track: final submission deadline |
| Dec 2026 | Winning papers are highlighted at a Google-hosted expo during NeurIPS 2026 |

## Prizes ($35,000)

| Award | Prize | What qualifies |
|---|---|---|
| Overall Best Paper | $15,000 | Highest average over the five judging criteria |
| Best New Resource | $10,000 | A new dataset, tool or software, as opposed to an empirical paper that produces new findings |
| Best New Application | $10,000 | A new use case for the code graphs and/or the provided embeddings |

- **The host's resource examples:**
  - a dataset built from the provided embeddings or other sources, for evaluating or fine-tuning Gemma 4
  - a library that uses Gemma plus another algorithm to cluster code by its behaviour
  - a dashboard for exploring GitHub repos that are similar in structure to a target repo
- **The host's application examples:**
  - code graphs used for writing TPU kernels
  - new evaluation datasets for new tasks
  - the graphs or embeddings applied to other existing benchmarks where they haven't been used
- **Awards per writeup:** a writeup can be considered for several awards but wins at most one. A writeup that wins Best Paper drops out of the other two ([743255](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743255)).
- **NeurIPS expo:** only the prize winners are promised a slot ([743310](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743310)).
- **Main track prizes:** separate from these. $65k ($37k / $18k / $10k), so $100k in total.

## How papers are judged

Each criterion is scored 0–5. The final score is the mean of the five, equally weighted.

| Criterion | What the judges ask |
|---|---|
| **Novelty** | Does the work give new insights, deepen understanding, or highlight important properties of existing methods? |
| **Quality** | How general is the approach beyond the competition? Does it transfer to similar problems? |
| **Relevance** | How significant is the potential impact on software engineering and agentic learning? |
| **Verifiability** | Is there enough detail to see how the innovation works, and how the data was obtained, analyzed and interpreted? |
| **Clarity** | Is the paper clearly presented and written? |

- **Winners and scores:** the three highest-scoring writeups win. Rubric scores are not shared with entrants.
- **Ties:** a tie goes to the paper "entered first to the Competition". If a winner is disqualified, the next-highest paper takes its place.
- **Resource and Application awards:** judged on the same five criteria, read "in the context of that prize". So novelty, quality and relevance are scored as a resource or an application ([743255](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743255)).
- **AI assistance:** coding assistants are fine for code. For the text, the host warns: "Purely LLM generated 'slop' will not score well", while LLM editing used "tastefully" is fine ([745712](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745712)).

## What we submit

To submit, click **New Writeup** on the track's page, then **Save**. The **Submit** button appears at the top right.

**Required content:**
- Title and subtitle
- Abstract
- Introduction
- Description of the research, including methods and experiments
- Related work and citations

**Optional:**
- **Public notebook** in the Project Links field. If it's a private Kaggle notebook, Kaggle makes it public after the deadline.
- **Public Project Link** to an arXiv-ready PDF (LaTeX or Word), instead of writing the paper in the Writeup body.
  - It must be readable with no login or paywall.
  - An arXiv link is acceptable as the submission ([745872](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745872)).
- **Attachments:** any private Kaggle resource attached to the Writeup becomes public after the deadline.

**Practical notes:**
- Other teams' writeups stay hidden until the track closes, so we can't read them before submitting.
- A save bug in the Writeup editor was reported on 2026-09-24, and no fix was confirmed ([743094](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743094)). Create and save a writeup early to make sure the editor works for us.

## Rules that matter

- **Team and writeup limits:**
  - Teams have at most 5 people. Each team may submit at most **two writeups**.
  - There is a single submission track, and the judges decide which awards each writeup is eligible for ([743255](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743255)).
  - The rules' generic limits are looser (5 submissions a day, 2 final selections). The two-writeup cap is the one that applies.
- **No required model:** the main track's single-model rule doesn't carry over, and the paper track's rules name no model.
- **"Unpublished":**
  - An arXiv preprint is fine ([745872](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745872)).
  - An extension of a pilot study that already appeared in a blog post or Kaggle benchmark is a valid submission ([745150](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745150)).
- **Non-archival:** the same work can go to other venues in parallel ([743310](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743310)).
- **Identity:**
  - Kaggle flags the track as requiring identity verification.
  - Winners can opt out of photo publicity ([745326](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745326)).
  - An individual without a company can receive a prize ([745712](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745712)).
- **Winner obligations (rules 2.5, 2.8):**
  - License the winning submission and the code that produced it under an OSI-approved license that doesn't limit commercial use. The rules list Apache 2.0.
  - Deliver code and documentation that follow Kaggle's [winning model documentation guidelines](https://www.kaggle.com/WinningModelDocumentationGuidelines). That includes the training code, the inference code and a description of the computing environment.
  - Sign the prize paperwork, including tax forms.
- **Competition data (rule 2.4):**
  - We may use it for any purpose, including research.
  - We must not redistribute it to anyone who hasn't joined the competition.
  - **Releasing graphs regenerated from the public upstream repos:** the host said this "should be acceptable with citation and with linking to the Kaggle dataset", but would double-check. It was still unconfirmed on 2026-10-07 ([743255](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743255)).
- **External data and models (rule 2.6):** allowed if they are public and equally accessible to everyone, at no cost or minimal cost.
- **Code sharing:**
  - While the competition runs, code may be shared only publicly, on the competition's Kaggle forum or notebooks. Sharing it there licenses it under an OSI license.
  - Whether a notebook or dataset linked from a writeup satisfies this is unanswered ([744941](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744941)).
  - Precedent: the author of codegraph-loc posted its GitHub link on the paper-track forum ([746035](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/746035)).

## Topics the host suggests

"Including, but not limited to":

- **Tuning & Optimization:** parameter-efficient fine-tuning and RL that improve SWE agents.
- **Code Comprehension:** new techniques for generating, parsing and embedding code graphs, so a model understands complex repositories more deeply.
- **Tasks & Benchmarks:** new tasks, evaluation datasets or resources that target code structure, graphs and structured code generation.
- **Graph Reasoning:** methods and architectures that improve LLM reasoning over large-scale graphs.
- **Novel applications:** creative uses of Gemma 4, DiffusionGemma or code graphs beyond standard benchmarks ([743012](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743012)). DiffusionGemma is Google's experimental open diffusion LM, released June 2026.

The main competition's Prizes page adds that using the released graph and embedding dataset is "highly encouraged, but not a requirement".

## The released code graphs and embeddings

These come from the [main competition's data](https://www.kaggle.com/competitions/gemma-4-developer-agent/data), which we keep on HPC (see the sibling repo).

- **`graphs/`:** NetworkX node-link JSON, one graph per repo snapshot.
  - Each node has a fully qualified symbol `id` (e.g. `fastapi.routing._prepare_response_content`), a `name`, and the definition's full source in `text`.
  - Each edge has `source`, `target`, `type` and `key`.
  - There are 256 files: 129 task-named files hard-linked to 127 commit-named files.
- **`embeddings/`:** `.npz` archives holding one 256-dim `float32` vector per graph node id.
- **Tools that use them:** `get_code_neighbors`, `get_code_subgraph` and `search_similar_code`.
- **Public tasks:** 129 tasks in `tasks.jsonl`: fastapi 67, rich 48, requests 13, httpx 1. Each comes with its gold `patch` and `test_patch`.
- **Hidden test set:**
  - About 120 tasks from private repositories, with the graphs built by the same method.
  - Each task was checked fail-to-pass and pass-to-pass.
  - Each also passed a "near completion" check: a larger frontier model passes the task or comes within one test of passing.

Known problems, reported on the main forum. These are both pitfalls and topics other entrants have already claimed (next section):

- **Coverage gaps ([742911](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911)):** the graphs hold no `async def` nodes (0 of 305 in the files gold patches touch), no module-level nodes, and a single edge type, `calls`. Embedding keys match the graph node ids exactly, so `search_similar_code` can't surface async code either.
  - The host says no private-repo task touches an async function. At most 1.8% of the private repos' functions are async, so the graphs won't be regenerated ([745998](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745998)).
- **Collapsed embeddings ([744040](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744040)):**
  - Random node pairs have a mean cosine of 0.78 (fastapi) to 0.90 (rich), and 8–19% of pairs are at ≥ 0.99.
  - The gold file lands in the top 10 for 35 of 129 tasks, against 67 for a plain grep.
  - The embedding model is undisclosed.
- **Graph-tool lookups ([745731](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745731)):**
  - `search_similar_code` can't embed free text. It only looks up a stored vector by node id or node source. In our three v28 runs, every call (over 400) returned nothing.
  - `get_code_neighbors` matches exact ids only. It doesn't show edge direction, and it sorts neighbours alphabetically and cuts them at 50.
- **Graph quality ([745731](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745731)):**
  - The released graphs hold 14–21% of 6,432 call pairs recorded at runtime in fastapi, rich and requests, and 2% in httpx. A rebuilt `ast` graph holds 62–69%.
  - The embeddings track graph distance (Spearman −0.28 to −0.34) more than code text (0.06–0.18 against TF-IDF).
  - TF-IDF puts a changed definition in the top 10 for 52% of tasks. The embeddings manage 9%.
- **Unbounded output ([744577](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744577)):** `search_similar_code` returned whole node sources with no cap. `fastapi.applications.FastAPI` is about 130k characters, and one call could overflow the 32k context. The host said it would cap the output like the other tools.

## Public work already out (novelty check)

Any paper in these areas has to cite this work and go beyond it.

- **[codegraph-loc](https://github.com/danielmantiglia/codegraph-loc)** (Apache-2.0; [746035](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/746035)), the code behind a paper-track writeup:
  - an `ast` code-graph generator with async functions, module-level code and typed edges (`calls`, `contains`, `imports`, `inherits`), in the released schema
  - function-level gold labels and a localization benchmark built from public git history, with a held-out set from SWE-bench Lite and pymatgen
- **Code-graph audit:** a [notebook](https://www.kaggle.com/code/akhilchinta1505/code-graph-audit) and a [call-trace dataset](https://www.kaggle.com/datasets/akhilchinta1505/code-graph-call-traces), both Apache 2.0. They rebuild all 127 graphs and measure them against runtime calls ([745731](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745731)).
- **Earlier audits:** the async and empty-file audit ([742911](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911)), and the embedding-collapse and exact-name findings ([744040](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744040)).
- **"Making a Legacy PHP Monolith Verifiable for a Gemma 4 Bug-Fixing Agent"** ([745027](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745027)):
  - It uses Playwright end-to-end tests over Dockerized MariaDB snapshots as hidden test oracles.
  - It reports a +6.8 pt lift at N=33 (p ≈ 0.11), and 30%+ verifier gaming when agents were bootstrapped with model-written tests.
- **[Passing tests vs. solving](https://www.kaggle.com/benchmarks/shiroganemaji/passing-tests-vs-solving)** ([745150](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/745150)): a Kaggle benchmark that uses "poisoned" examples to test whether a model solves the spec or special-cases the tests. An extended version is planned for this track.
- **[gemma4-swe-kit](https://github.com/damsolanke/gemma4-swe-kit)** (Apache-2.0): it converts SWE-rebench OpenHands trajectories into scorer-rendered training windows.

## Open questions (no host answer as of 2026-10-07)

<!-- last checked: 2026-10-07 -->
- **Word count:** do the title, abstract, figure and table captions, and references count toward the 3,000 words? Is there an official counter? ([743255](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743255))
- **PDF vs. Writeup body:** if the paper is a linked PDF, is the PDF or the Writeup body what gets judged? ([743255](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743255))
- **Tie-breaks** ([746168](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/746168)):
  - Does "entered first" mean when the team joined or when the writeup was submitted?
  - Does editing or re-submitting a writeup reset that time?
  - Which version of a linked PDF do the judges read?
- **Data releases** ([743255](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/discussion/743255)): may we release regenerated graphs, or agent traces that contain task-repo code?
- **Linked code:** does a notebook or dataset linked from a writeup count as public code sharing? ([744941](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744941))

## What this means for us

- **"Quality" means generality.** Results on only the 129 public tasks will score low on it, since fastapi and rich make up 115 of them. Plan at least one held-out evaluation; codegraph-loc used SWE-bench Lite and pymatgen.
- **Verifiability means releasing code and numbers, within the data rules.** Release scripts that rebuild everything from the Kaggle dataset rather than the data itself.
- **The graph-audit ground is taken.** The async gap, the `calls`-only edges, the collapsed embeddings and exact-name lookup are already public. A graph paper needs a new result built on top of them.
- **Count every word.** Until the host says otherwise, count the title, abstract, captions and references, and stay under 3,000.
- **Submit a complete version early, then improve it.** This avoids a deadline crunch and may help in a tie. It is still unknown whether edits reset the "entered first" time.
- **Watch the timing against the main track.**
  - Attached notebooks and resources become public after 2026-11-12, and other teams' writeups once the track closes.
  - The main track runs until 2026-12-02, so anything we publish reaches the other main-track teams during its last 20 days.
- **Two writeups at most.** We could submit an empirical paper and a resource paper, but each can win only one award.

## Primary direction: OPSD for a thinking-off agent

<!-- decided 2026-10-07; the design gets worked out in a dedicated OPSD session -->

**Goal:** train Gemma 4 31B, by self-distillation, into an agent that runs with thinking off but acts like the thinking-on model.
- **Teacher:** the same model, which also sees the task's reference patch.
- **Training signal:** the teacher scores the student's own rollouts token by token.
- **Method:** On-Policy Self-Distillation (OPSD), [arXiv 2601.18734](https://arxiv.org/abs/2601.18734). Our copy is [opsd.pdf](opsd.pdf).
- **Fallback:** if this fails the go/no-go check in the [plan](#plan), the paper becomes the thinking-budget study (option A below). Its runs double as this method's baselines.

### Why OPSD

- **It fixes RFT round 0's data problem.**
  - Round 0 kept 19 successful trajectories from 11 tasks: 221 samples with just 52k loss tokens. Every failed rollout was thrown away.
  - OPSD gets a dense, per-token signal from every rollout, failures included.
  - It needs only a reference patch, not a passing test. So the 20 public tasks our scorer can't grade become training data too.
- **It attacks speed in the one way the main-track harness allows.** We can't change decoding, but we can change how many tokens the agent generates.
  - Thinking takes 65–71% of agent time.
  - With thinking off, the agent loops and mangles calls. 50% of tool calls repeat an identical earlier call, and 24% are malformed (with thinking on: 9% and 2.2%). After one malformed call, the next is malformed 92% of the time.
  - OPSD's best setting in its paper was exactly this pairing: a thinking-off student and a thinking-on teacher.
- **It keeps the LoRA slowdown small.** Our notes blame much of round 0's loss on speed: the rank-64 adapter decoded at 22.5 tok/s against the base's 28.4. At rank 16 the slowdown is 8%, so the adapter breaks even at about 2 extra tasks per 70.
- **It extends OPSD past its stated limits.** The paper only tested Qwen3 up to 8B on single-turn math, and it names larger scales as an open question. We'd test a 31B int4 multi-turn coding agent under a time limit.
- **It matches the host's first listed topic:** "Tuning & Optimization", meaning parameter-efficient fine-tuning and RL for SWE agents.
- **Its effects are measurable on 70 tasks.** The repeat and malformed shares are far less noisy than the resolve rate.
- **It can be submitted to the main track.** A rank-16 adapter plus `thinking_budget: 0` is a valid submission. Kaggle's hidden set, from private repos, then doubles as an out-of-distribution test.

### Alternatives we weighed

| | A. Thinking-budget study | **B. OPSD (chosen)** | C. DSpark drafter |
|---|---|---|---|
| Core claim | How much thinking a local agent needs under a time limit | A thinking-off student that acts like the thinking-on model | Faster decoding for the same agent |
| Engineering | 1–3 days (config sweeps) | About 1 week (a new loss in `train_lora.py`) | 2–3 weeks (drafter training plus serving integration) |
| GPU time (rough) | 50–100 GPU-h | 100–150 GPU-h | 150–300+ GPU-h |
| Usable in main track | Yes, as config | Yes: LoRA plus `thinking_budget: 0` | No |
| Novelty | Clear gap, but measurement only | Crowded area, but this angle is untaken | Small-scale replication of DeepSeek's work |
| Risk | Low | Medium to high | High |

- **Writing up our main-track agent:**
  - Our rank isn't the problem. The rubric doesn't score rank, and 0.13 vs 0.15 is 8 vs 9 tasks out of 58, inside our measured noise.
  - The problem is that most of our findings are quirks of this harness (LoRA zeroing, the broken search tool, Kaggle's pacing), which score low on Quality.
  - Most of them are also single runs, and v28 invalidated the v25 results.
- **A topic from scratch:**
  - 36 days is too short. Our 70-task runs take 2.6–5 h and vary by ±2.2 tasks, so every claim needs repeated runs.
  - Code graphs and benchmarks are crowded areas.
  - We'd give up tooling that already works.
- **A. Thinking-budget study:**
  - We found no study that varies the thinking budget of open models on SWE tasks while counting malformed and repeated calls.
  - The nearest evidence is mixed. In [NetConfArena](https://arxiv.org/pdf/2608.23179), thinking raised invalid actions for Qwen3-32B but cut them for Qwen3-8B. ["The Danger of Overthinking"](https://arxiv.org/pdf/2502.08235) found that more reasoning can hurt agent tasks.
  - It becomes OPSD's motivation and baselines, and the fallback paper.
- **C. DSpark** ([arXiv 2607.05147](https://arxiv.org/abs/2607.05147); our copy is [dspark.pdf](dspark.pdf)):
  - **Main track:** it can't be used there. The harness launches vLLM with its own flags, and a drafter is a second model.
  - **Training cost:** its recipe regenerates 1.3M target responses and trains for 10 epochs on the target's hidden states. At 31B that is far beyond 5 weeks.
  - **Checkpoints:** the released drafters cover Qwen3 4B/8B/14B, Gemma 4 12B and DeepSeek-V4. None is for the 31B.
  - **Wrong setting:** its scheduler is built for high-concurrency serving. A local agent runs about one request at a time, where verifying extra draft tokens is almost free.
  - **At most:** a training-free n-gram speculation run in vLLM could fill one paragraph of the paper, since `edit_file` arguments copy file text verbatim.
- **Code graphs:** other entrants have already published the audit findings, and we have no positive graph result.
- **Silent serving bugs** (the LoRA-zeroing probe, unannounced harness changes): too specific to this harness to stand alone. They could fill a short section if space allows.

### Initial design (to refine in the OPSD session)

- **Student:** the competition checkpoint `gemma-4-31b-it-qat-w4a16-ct`, with thinking off (`thinking_budget: 0`) and a rank-16 LoRA.
- **Teacher:**
  - The same weights with the adapter disabled. That is the initial policy, which is what OPSD prescribes.
  - It sees the student's context plus a privileged block: the gold patch, or a location-only hint.
  - Its thinking can be on or off.
- **Loss:**
  - Per-token forward KL on the assistant tokens of the student's own rollouts, with OPSD's pointwise clipping.
  - Start with the sampled-token variant, which is cheap; OPSD reports 82.1 vs 84.1 for full-vocabulary KL.
  - Move to chunked full-vocabulary KL if memory allows.
- **Split:**
  - Train on the 59 non-dev public tasks: rich 48, fastapi 7, requests 4. These include the 20 tasks our scorer can't grade, so first check that their sandboxes still run.
  - Evaluate on the 70 dev tasks: fastapi 60, requests 9, httpx 1. Since training is mostly rich and evaluation mostly fastapi, this is a cross-repo test.
- **Metrics:**
  - resolved tasks, with 2 seeds and paired tests
  - repeat share and malformed share
  - the chance that a malformed call follows a malformed call
  - time and tokens per task
- **Baselines already measured on v28:**
  - thinking-off base: 19/70
  - thinking-on base: 25/70 (20/70 in a second run)
- **Ablations and controls:**
  - full gold patch vs. a location-only hint
  - a wrong-reference control (another task's patch)
  - teacher thinking on vs. off

### Risks from the 2026 literature

This area is crowded with preprints from June–September 2026, and plain OPSD has known failure modes in multi-turn agents:

- **[PSP (2609.29051)](https://arxiv.org/abs/2609.29051v1):** OPSD students "act confidently without the knowledge behind that confidence". PSP reports gains on SWE-bench Verified by moving the hint into the sampler.
- **[HERO (2606.11559)](https://arxiv.org/pdf/2606.11559):** naive multi-turn OPSD made performance worse.
- **[Rethinking OPSD for Thinking Models (2607.05184)](https://www.alphaxiv.org/abs/2607.05184):** full reference solutions in the teacher's context hurt more than short hints. Hence the location-only ablation.
- **[Rethinking Privileged Information in OPSD (2608.18271)](https://arxiv.org/abs/2608.18271):** references from other problems produced much of the gain. Hence the wrong-reference control.

The paper must also set itself apart from:
- **[Hindsight Hint Distillation (2605.11556)](https://arxiv.org/pdf/2605.11556):** SWE-specific, +8 points on SWE-bench Verified. It trains by SFT on successful runs guided by hints.
- **[PivotOPD (2609.40285)](https://arxiv.org/abs/2609.40285)**
- **[Patches-to-Trajectories (2605.21996)](https://arxiv.org/abs/2605.21996):** uses the gold patch to curate SFT data.
- **[SDPO (2601.20802)](https://arxiv.org/pdf/2601.20802):** uses environment feedback as the privileged context.

As of 2026-10-07, we found nothing that combines a gold-patch teacher, on-policy self-distillation, a thinking-off student and a time limit.

### Fit with our stack

Paths are in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo).

- **Trainer:** `baseline/train_lora.py` already trains a LoRA on the decompressed bf16 checkpoint, with HF and PEFT in a custom loop that chunks logits and applies Gemma's soft-cap.
  - The teacher pass is the same model under PEFT's `disable_adapter()`, so the H200 needs no second copy.
  - Apply the soft-cap (`head_logits`) to both teacher and student.
- **Token alignment:** the teacher needs the student's exact token ids, which `baseline/token_proxy.py` already logs. Insert the privileged block after the system prompt.
- **Throughput:** OPSD needs two forward passes per token. So build first the prefix sharing proposed in `post_train/RFT/README.md`, which cuts the tokens processed by 4.7×.
- **Compaction:** ADK compacts the context at about 14k tokens, so per-call prompts aren't always prefixes of each other. Decide between exact per-call training and packed trajectories.
- **Cluster limits:**
  - Training runs on an H200 (141 GB). The `gpu` partition caps jobs at 8 h and 1 GPU each, with 4 running per user, so training must resume from checkpoints.
  - Rollouts and evals run on L40S.
- **Serving:** vLLM 0.19.1 with wheelhouse v28 serves each adapter at its own rank. No adapter bundle has a confirmed Kaggle score yet, so our Kaggle submission also tests that path.

### Plan

| Dates | Work |
|---|---|
| Oct 8–14 | Split the tasks, run thinking-off rollouts on the training set, and implement the loss. First run a cheap diagnostic: where does the teacher that sees the patch disagree with the student? We expect malformed-call tokens and repeated calls, and the result doubles as a paper figure. |
| Oct 15–21 | Round 1: rank 16, about 100 steps. Evaluate on dev with 2 seeds against both baselines. |
| **~Oct 22** | **Go/no-go:** if the repeat and malformed shares don't fall on the dev tasks, switch to the fallback paper (A). |
| Oct 22–Nov 2 | Round 2 on fresh rollouts, the ablations, and one Kaggle submission of the best adapter. |
| Nov 3–10 | Write at most 3,000 words and submit early. The deadline is Nov 12. |

### Evidence from our main-track runs

Paths are in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo). Numbers are as of 2026-10-07. Most come from a single run on 70 local tasks.

- **Thinking on vs. off** (`agentic_workflow/README.md`):
  - Resolved: 25/70 vs 19/70 locally, and 7/58 vs 3/58 on Kaggle.
  - Repeated calls: 9% vs 50%. Malformed calls: 2.2% vs 24%.
  - A malformed call follows a malformed call 47% vs 92% of the time.
  - With thinking on, thinking takes 65–71% of agent time.
- **LoRA slowdown on v28** (`post_train/RFT/v28.md`, `post_train/README.md`): 8–9% at rank 8–16, 16% at rank 32 and 21% at rank 64.
- **RFT round 0** (`post_train/RFT/round0.md`, v25):
  - Trained at rank 64 on 11 tasks, 19 trajectories and 221 samples.
  - Resolved 8/70 against the base's 23/70, and 2/59 vs 12/59 on unseen tasks.
  - Decoded at 22.5 tok/s against the base's 28.4.
- **Noise** (`post_train/v28/README.md`):
  - Over three runs, 16 tasks always resolve, 40 never do and 14 flip. That is about ±2.2 tasks per run.
  - The same zip scored 25/70 locally and 7/58 on Kaggle, a 3× gap.
- **Context overflow:** it throws away the working patch on 3–4 of 70 tasks per run.
- **Grader check** (`post_train/RFT/README.md`): in our replica of the scorer, 111 of 129 gold patches pass and 109 tasks are sound.

## Repo workflow (proposed)

- **This repo** holds the paper (Writeup markdown and/or LaTeX), the figures, and the scripts that turn our logs into the numbers we cite.
- **Experiments, data and models** stay in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo) and run on NEU Explorer, per that repo's CLAUDE.md.
- **Don't commit competition data here** (tasks, snapshots, graphs, embeddings). The rules forbid redistributing it, and this repo may go public along with the paper.

## Sources

Read on 2026-10-07:
- the paper track's Overview pages (Description, Evaluation, Submission Requirements, Timeline), Rules, prize tracks and all 15 forum threads
- the main competition's Data, Evaluation, Prizes, Timeline and Rules pages, and its forum threads on graphs, embeddings and the paper track
- the OPSD and DSpark papers ([opsd.pdf](opsd.pdf), [dspark.pdf](dspark.pdf)), and the 2026 literature linked under [Primary direction](#primary-direction-opsd-for-a-thinking-off-agent)

Kaggle's pages override this file. Re-check them before submitting.
