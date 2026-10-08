# Gemma4_paper_track

Our entry for **[Google – The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)** (Kaggle hackathon, host: Google DeepMind). It is the research-paper companion to the [main leaderboard competition](https://www.kaggle.com/competitions/gemma-4-developer-agent). Our agent for the main competition lives in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo).

## Goal

Submit a **Kaggle Writeup of at most 3,000 words** that reports original, unpublished research advancing agentic software engineering.

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

## Candidate directions (from our main-track work; to decide)

The main-track notes already hold measured results that fit the host's invitation to "document your approach".
- File paths below are in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo).
- Numbers are as of 2026-10-07. Most come from a single run on 70 local tasks.

### 1. Under a wall-clock budget, serving cost decides what helps

Topic: Tuning & Optimization. Aims at Best Paper.

- **Thinking off scored lower:** 19/70 vs 25/70 locally, and 3/58 vs 7/58 on Kaggle (`agentic_workflow/README.md`).
  - Without thinking, 50% of tool calls repeat an identical earlier call and 24% are malformed. With thinking on, those shares are 9% and 2.2%.
  - With thinking on, thinking takes 65–71% of agent time, and resolved tasks finish close to the per-task clock.
- **LoRA has a serving cost** (`post_train/README.md`, `post_train/RFT/v28.md`).
  - On wheelhouse v28, decoding slows 8–9% at rank 8–16, 16% at rank 32 and 21% at rank 64.
  - To break even, a rank 8–16 adapter must gain about 2 tasks per 70, and a rank 64 adapter about 10.
- **Our first RFT adapter lost to the base** (`post_train/RFT/round0.md`, v25 harness).
  - It was rank 64, trained on 221 samples from 11 tasks, and resolved 8/70 against the base's 23/70.
  - It decoded at 22.5 tok/s against the base's 28.4, and every trained task it lost hit the 4-min timeout.
- **Overflow loses work:** a context overflow throws away the working patch, on 3–4 of 70 tasks per run.
- **Gaps:**
  - one run per variant
  - v25 results no longer match the scorer
  - no adapter has been trained or scored on v28
  - thinking on/off was compared on only one agent config, the host's sample

### 2. How far a small SWE leaderboard can be trusted

Topic: Tasks & Benchmarks / evaluation method. Could be its own writeup, or the evaluation section of 1.

- **Local and Kaggle scores disagree.** The same zip resolved 25/70 (35.7%) locally and 7/58 (12.1%) on Kaggle's hidden set (`post_train/v28/README.md`). That is a 3× gap, far beyond the noise of either.
- **One run is noisy.** Over three runs of one agent, 16 tasks always resolve, 40 never do and 14 flip.
  - That is about ±2.2 tasks per run.
  - So one run per variant only detects a change of about 6 tasks.
- **The public board is coarse.** It shows floor(100·k/58)/100, and byte-identical zips from other teams scored 0.12 and 0.15.
- **Our grader check** (`post_train/RFT/README.md`): in our replica of the scorer, 111 of 129 gold patches pass and 109 tasks are sound. A public notebook already reports 119 of 129 sound.
- **Gaps:** repeated runs, and an explanation for the local-vs-Kaggle gap.

### 3. Catching silent serving bugs

A smaller resource, or a section of 1.

- **Silently zeroed LoRAs** (`docs/issue_743508_lora_wipe.md`):
  - Up to wheelhouse v22, the host's vLLM silently zeroed every LoRA.
  - A poisoned-adapter probe exposed it, because the logprobs stayed identical to the base.
  - The v25 fix matches the one in our write-up.
- **Unannounced harness changes:** wheelhouse v28 changed adapter serving, thinking and tool output without notice. Our base agent went from 23/70 to 20/70; a sign test gives p = 0.55, so that is noise.

### 4. Code graphs: a poor fit for us

We have no positive graph result: every `search_similar_code` call in our v28 runs returned nothing. Other entrants have also published the audit findings already.

**Suggested lead: 1, with 2 as its evaluation section.**
- It reuses what we have already measured and answers "document your approach" directly.
- Its biggest gap, repeated runs on v28, is the same one 2 needs filled anyway.

## Repo workflow (proposed)

- **This repo** holds the paper (Writeup markdown and/or LaTeX), the figures, and the scripts that turn our logs into the numbers we cite.
- **Experiments, data and models** stay in [Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo) and run on NEU Explorer, per that repo's CLAUDE.md.
- **Don't commit competition data here** (tasks, snapshots, graphs, embeddings). The rules forbid redistributing it, and this repo may go public along with the paper.

## Sources

Read on 2026-10-07:
- the paper track's Overview pages (Description, Evaluation, Submission Requirements, Timeline), Rules, prize tracks and all 15 forum threads
- the main competition's Data, Evaluation, Prizes, Timeline and Rules pages, and its forum threads on graphs, embeddings and the paper track

Kaggle's pages override this file. Re-check them before submitting.
