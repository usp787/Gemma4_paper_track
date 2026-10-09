# baseline/: the OPSD paper's baseline runs

<!-- status read from Explorer on 2026-10-09 (results/, submissions/, squeue); update the Status table after each run -->

This folder holds the baseline arms of the [OPSD study](../opsd_assessment.md): their status, the scripts that queue their runs, and the report that turns the runs into the paper's numbers.
- **The runs themselves use the main-track repo's eval tooling** ([Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo)): `post_train/v28/submit_baseline.sh`, `baseline/submit_eval.sh` and `baseline/diagnose_run.py`. They are scored exactly as the main track scores its variants.
- **Who does what:** Claude drafts and tests these scripts locally. You commit, push, pull on Explorer, and queue every job.

## Status (2026-10-09)

**Kaggle moved to wheelhouse v30 on 2026-10-09 (14:43 UTC), a day after v29, so every arm waits for the v30 re-alignment.** v30 bumps swegemma to 0.2.11 ("support quantized e2b/e4b/12b models and consolidate discovery/registry"); by file name no other wheel changed. The main track's `post_train/RFT/v29.md` has v29's changes, its re-alignment, its results and the v30 commands.
- **v29's re-alignment finished on 2026-10-09, and GPU jobs run again** (R and G below). Its first two builds failed because the monthly /scratch purge had deleted the conda envs' CA bundle. Until that is fixed, env builds need `PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt`.
- **On v29, `on`'s `base_v29_l40s` resolved 26/70,** no measurable change from its two v28 runs (20 and 25). The grader check kept the same 109 sound tasks, so the dev set, the split and the train check stand.
- **What v29 changed:** the code-graph tools, their descriptions and the handling of tool errors. A submitted patch also loses its hunks in test and config files before it is graded. Thinking works as on v28 for all our agents, which set `include_thoughts: true`.
- **Runs, smokes and gates are named after the wheelhouse version.** `arms.tsv`'s `{v}` stands for it, and `baseline_report.py` counts one version's runs. The v28 and v29 runs in the table below are their versions' record; `bash baseline/opsd_baselines.sh report 28` (or `29`) rebuilds their tables.
- **The Kaggle anchor for v29** is `agentic_v1_think`'s unchanged zip (sha256 6418b8752004, 8/58 on v28), sent 2026-10-09 00:22 UTC. It was still pending at 16:21 UTC, after v30 went live, so it may be scored on v30.

**The model is complete again.** `/scratch/$USER/gemma4/models/gemma-4-31b-it-qat-w4a16-ct` had held 0 bytes since 2026-10-07 13:19 EDT, when the monthly /scratch purge deleted it. `fetch_model` (job 10927935) re-downloaded it on 2026-10-08, 17:13–17:19 EDT: `model.safetensors` is 23,265,352,448 bytes, and `chat_template.jinja` is the June one (16,934 bytes). The weights live on scratch, not in either checkout, so both tracks read the same copy. If `opsd_baselines.sh status` ever reports the model incomplete again, rerun `cd ~/Gemma4_kaggle_repo && sbatch slurm/fetch_model.slurm` (CPU, up to 2 h). The purge goes by file age, and the re-fetched files keep their archive's date (2026-06-05). So before the next one, on the first Tuesday of November (2026-11-03), run `touch /scratch/$USER/gemma4/models/gemma-4-31b-it-qat-w4a16-ct/*`.

**The split and the train check are done (2026-10-08).**
- `split` (job 10928660) wrote the five split files to `/scratch/$USER/gemma4/paper/split/`. Its first try (10928412) died of a broken pipe in `srun` before Python started, which is why `split` is now a batch job.
- The train check (job 10928765) took 6 min. Every training task's sandbox started in both modes: gold resolved 41/59 and empty 2/59, so 39 are sound, and no task's result changed since 2026-09-29.
  - Results: `/scratch/$USER/gemma4/results/opsd_train_check_10928765/`.
  - Its log went to `/scratch/$USER/gemma4/paper/logs/`, from before logs moved to this repo's `logs/`.

| Arm | Agent (zip sha256) | What it is | v29 runs (the record) | v28 runs (the record): resolved of 70 | Kaggle | Next, on v30 |
|---|---|---|---|---|---|---|
| `off` | `baseline_nothink` (3909932c84e0) | thinking off; the host's sample prompt, temperature 0.2, 60 calls | none | `baseline_nothink_v28_zip` **19** (10-05) | v28: 3/58 | step 1 |
| `on` | `baseline_noadapter` (d9db81dd3d76) | thinking on (budget 4096); otherwise the same files as `off` | `base_v29_l40s` **26** (10-09, from `agents/`) | `base_v28_zip` **25** (10-04), `base_v28_l40s` **20** (10-01, from `agents/`) | v28: 7/58 | step 4 |
| `v1` | `agentic_v1` (6cd3c3ec1a74) | thinking off; V1's rewritten prompt, temperature 1.0, 100 calls, 300-s commands, no analyzer | none | none | – | steps 2–3 |
| `v1_think` | `agentic_v1_think` (6418b8752004) | thinking on (4096) with V1's prompt, but temperature 0.2, 60 calls and one more prompt line | none | none | v28: 8/58; v29: the anchor, pending (may be scored on v30) | decision 2 |
| `v2` | `agentic_v2` (8dcadfe313d3) | V1 plus the advisor (thinking on, called by the model) and two skills: **the model decides** | none | `agentic_v2_v28_zip` **25** (10-06) | – | step 5 |

- **Not arms (v28):**
  - `base_v28_zip_paced` (18/70) is paced, so the report leaves it out.
  - `agentic_v1.1_think_v28_zip` (24 of 55 finished) ran on an 8-minute clock with 28 calls.
- **`v2`'s run isn't in the main-track README yet.** It ran 2026-10-06 21:17–23:56, and its `diagnosis.md` is in its run dir:
  - 25/70 resolved. Against `off`: 17 tasks in both, 8 only here, 2 only there; p = 0.11.
  - The advisor ran 68 times in 59 tasks, a median 38 s each. Thinking was 22% of output tokens.
  - 171 `run_skill_script` calls lacked `skill_name`, so the skills add malformed calls of their own.
  - 28 tasks hit the 4-minute clock.
  - Where its model calls the advisor, against bad calls, is in [../adaptive_inference/README.md](../adaptive_inference/README.md#what-the-traces-say).

**What the v28 runs say** (`baseline_report.py` on the pulled runs, 2026-10-08). This is the v28 record: v29's tool changes may move every column.

| Arm | Tool calls / task (median) | Agent s / task | First source edit (median s; never) | Stale repeats | Malformed | Next call bad, after a bad / a clean call |
|---|---|---|---|---|---|---|
| `off` | 44 | 122 | 47; 31 | 47.2% | 29.8% | 88.2% / 14.8% |
| `on` (`base_v28_zip`) | 22 | 214 | 135; 26 | 4.1% | 2.7% | 34.1% / 4.6% |
| `v2` | 26 | 182 | 106; 22 | 23.5% | 18.4% | 78.4% / 8.9% |

- `base_v28_l40s` has no pulled traces, so `on`'s row is run A's alone.
- `v2`'s malformed share counts the skill tools too, as its diagnosis does.
- **`on` on v29 (`base_v29_l40s`):** 19 tool calls per task (median), 212 s, first source edit at 121 s (never in 19 tasks), stale repeats 9.8%, malformed 8.9%, next call bad 66.5% / 5.2%.
  - One task's loop drives the bad-call shares: 65 `edit_file` calls in a row without `old_string`.
  - So before arms are compared, the report needs a per-task view of bad calls (the main track's v29.md).

- **`on` against `off`, over both `on` runs:** better on 9 tasks, worse on 5. That is +3.5 tasks (95% CI −2.0 to +9.5), sign test p = 0.42. Run A alone: +6, 8 better and 2 worse, p = 0.11.
- **`v2` matches `on`'s best run, but differs from `off` in seven things at once:** prompt, temperature, three budgets, no analyzer, the advisor, and two skills. `v1` splits config from structure, which is why it runs first.

## Order of work

Each line is one command on the Explorer login node, from `~/Gemma4_paper_track`. Without `GO=1` a command only prints what it would do and what blocks it.

| # | Command | What it does | Cost |
|---|---|---|---|
| 0 | `cd ~/Gemma4_kaggle_repo && sbatch slurm/fetch_model.slurm` | the model: done 2026-10-08 (job 10927935); rerun only if `status` reports it incomplete | CPU, ≤ 2 h |
| 0a | `GO=1 bash baseline/opsd_baselines.sh split` | the split files, on scratch: done 2026-10-08 (job 10928660), from the v28 grader's sound set | CPU, ~1 min |
| 0b | `GO=1 bash baseline/opsd_baselines.sh check-train` | gold and empty patches on the 59 training tasks: done on v28 (job 10928765): all start, 39 sound. Rerun it only if G changes the sound set | CPU, 6 min |
| R | `cd ~/Gemma4_kaggle_repo && PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt bash baseline/submit_realign.sh`, the main track's re-alignment (its `post_train/RFT/v29.md`, "Then: v30") | the envs, a smoke and its gate, the LoRA check, then `base_<v>_l40s`, `on`'s first run on that version. Done for v29 on 2026-10-09: `base_v29_l40s` 26/70. Next for v30 | builds ~45 min, then ~4 h |
| G | `cd ~/Gemma4_kaggle_repo && PINS=1 sbatch baseline/grader_check.slurm` | gold and empty patches on all 129 tasks: do the 109 sound tasks, and so the 70 dev tasks, still hold? After R's harness build. Done for v29 (job 10933579): the same 109 | CPU, 10 min |
| 1 | `GO=1 bash baseline/opsd_baselines.sh run off` | `off`'s first v30 run, through the main track's chain (check, smoke, gate, run, report against `base_v30_l40s`) | ~3.5 h, ~6 L40S-h |
| 2 | `GO=1 bash baseline/opsd_baselines.sh run v1` | `v1`'s first v30 run, the same way, against `off` | ~3.5 h, ~6 L40S-h |
| 3 | `GO=1 bash baseline/opsd_baselines.sh run v1` | `v1`'s second run, then its report | ~2.5 h, ~5 L40S-h |
| 4 | `GO=1 bash baseline/opsd_baselines.sh run on` | `on`'s zip run, `base_v30_zip`: its second v30 run | ~3.5 h, ~6 L40S-h |
| 5 | `GO=1 bash baseline/opsd_baselines.sh run v2`, twice | `v2`'s two v30 runs | ~6 h, ~11 L40S-h |
| 6 | `GO=1 bash baseline/opsd_baselines.sh run <thinking-on arm>`, twice | the thinking-on arm for the student's config (decision 2) | ~6 h |
| 7 | `GO=1 bash baseline/opsd_baselines.sh run off` | `off`'s second run, only if `off` becomes the student | ~2.5 h |
| – | `GO=1 bash baseline/opsd_baselines.sh report` | the tables above, for every arm, on our wheelhouse (`report 28`: the v28 record) | CPU, minutes |

- **R, then G, before any v30 run.** Before R, `opsd_baselines.sh` refuses, since our envs are v29 and Kaggle's v30. A run before G could count tasks the v30 grader no longer scores.
- **Runs go one after another.** Only 2 L40S jobs of ours run at a time, and the main track shares them.
  - A first run (a chain) needs an empty eval queue: `submit_baseline.sh` refuses to start otherwise.
  - A later run can wait for the one before: `AFTER=<driver job id>`.
  - `opsd_baselines.sh` refuses to queue while Kaggle's wheelhouse differs from ours, and a first run passes `PACED=0`, since only unpaced runs count here.
- **Freeze the student config after step 3:**
  - `v1`, if both its v29 runs finish all 70 tasks with no harness errors and resolve at least what `off`'s v29 run does;
  - otherwise `off`.
- **Then start the round-0 rollouts** for OPSD, on v29 (A100, `gpu` partition, in parallel with steps 4–7). For example, with `v1` as the student:
  ```bash
  cd ~/Gemma4_kaggle_repo && PINS=1 TIME=02:00:00 bash baseline/submit_rollouts.sh agents/agentic_v1 /scratch/$USER/gemma4/paper/split/train59.txt opsd_r0_v1 4 1
  ```
  That is 4 passes over the 59 training tasks, one 2-h A100 job each, about 8 A100-hours.
- `bash baseline/opsd_baselines.sh status` shows the prerequisites, every arm's runs with their scores, and each arm's next step.

## One-time setup on Explorer

- **Clone this repo next to the main one,** over SSH as `~/Gemma4_kaggle_repo` was: `git clone git@github.com:usp787/Gemma4_paper_track.git ~/Gemma4_paper_track`. The HTTPS URL also works from Explorer (both checked with `git ls-remote`, 2026-10-08).
- **Before each command, `git pull` in both checkouts.** A job sees only pushed code, as in the main repo.
- **Logs go to this repo's `logs/`,** as `<job name>_<id>.out` (and `.err`). Git ignores them, since they name task ids.
  - That covers every job `opsd_baselines.sh` queues itself: `split`, `check-train`, `report`, and a later run's report (`diagnose`). Each command prints its log's path.
  - The main track's chain writes its own logs (check, smoke, gate, driver, shards, a first run's report) to `~/Gemma4_kaggle_repo/logs/`. `opsd_baselines.sh` links that folder in as `logs/main_track/`, so every log can be reached from here.
  - `logs/main_track/` is a link: deleting files inside it deletes the main track's logs.
- **Outputs go to scratch:**
  - the split and the reports in `/scratch/$USER/gemma4/paper/`;
  - the train check's results and the runs in `/scratch/$USER/gemma4/results/<run>/`, as for the main track.
- **No task ids go into this repo.** They come from the competition's `tasks.jsonl`, and the README's data rule applies.

## What every run shares

- **Tasks:** the 70 dev-sound tasks of `/scratch/$USER/gemma4/train/rft0/dev_sound_ids.txt` (fastapi 60, requests 9, httpx 1).
  - These are the dev repos' tasks whose gold patch passes and empty patch fails (grader check 10686757 on v28; 10933579 found the same on v29).
  - The OPSD split trains on the other 59: rich 48, and the 11 fastapi and requests tasks the grader can't score.
- **The environment:**
  - the training-reward env (`PINS=1`), unpaced, one task at a time on one L40S;
  - one wheelhouse version, v30 since 2026-10-09 (v29 for the day before), and the agent extracted from its zip; `run_meta.json` records both the version and the zip's sha256;
  - the report counts one wheelhouse version at a time, and leaves out any run that is paced or not `PINS=1`.
- **Runs:** an arm's first run is `<agent>_<v>_zip`, then `<first run>_r<k>` (`arms.tsv`, where `{v}` is the version: `_v30_zip` now; `_v29_zip` and `_v28_zip` in the record).
- **Noise:** one run varies by about ±2.2 tasks, and the difference between two runs by about ±3.1 (`post_train/v28/README.md`). So every arm gets 2 runs, and the report pairs tasks across arms.
- **Metrics:**
  - resolved per run;
  - against the reference arm, task by task: better, worse, the difference with a bootstrap CI, and a sign test;
  - from the traces: tool calls and agent time per task, the first source edit, stale repeats, malformed calls, and how bad calls chain;
  - from `calls.jsonl`: output tokens and the thinking share.
  - The trace rules match [../adaptive_inference/trigger_replay.py](../adaptive_inference/trigger_replay.py).

## Files

| File | What it does |
|---|---|
| [arms.tsv](arms.tsv) | one row per arm: agent, zip, run names with `{v}` for the wheelhouse version, reference arm, Kaggle scores as `v<N>:k/n`. Add a row for a new arm. |
| [opsd_baselines.sh](opsd_baselines.sh) | `status`, `run <arm>`, `split`, `check-train` and `report [N]`, all on our wheelhouse version. Nothing is queued without `GO=1`. |
| [make_split.py](make_split.py) | writes `dev70.txt`, `train59.txt`, `train59_sound.txt`, `train59_unsound.txt` and `split.json` to scratch, after checking the counts and that dev is all sound |
| [train_check.slurm](train_check.slurm) | gold and empty patches on the training tasks, in the `PINS=1` env; lists what changed since the 2026-09-29 grader check |
| [baseline_report.py](baseline_report.py) | the tables, as Markdown, plus JSON for figures. stdlib only; runs on scratch or on pulled run dirs |

## Decisions pending

1. **The student config:** `v1` or `off`, by the rule under Order of work.
2. **`v1`'s thinking-on arm.** `v1_think` differs from `v1` in more than thinking: temperature 0.2 against 1.0, 60 calls against 100, and one prompt line.
   - (a) Use it as it is. It has a Kaggle score (8/58 on v28, the v29 anchor pending), but every comparison carries that confound.
   - (b) A new agent dir in the main-track repo, `agents/agentic_v1_on`: `agentic_v1` with `thinking_budget: 4096`, one line.
     - Then package it: `sbatch slurm/package_submission.slurm agents/agentic_v1_on`.
     - Then add its row to `arms.tsv`, with the zip name the log prints.
   - Recommended: (b) for the paper, with `v1_think` kept as the main-track link.
3. **The model-decides arm.** `v2` also adds two skills, and its advisor sees only the request it is sent, not the conversation.
   - A cleaner arm is V1 plus the advisor alone: `agentic_v2` without `skills:` in `agent.yaml` and without the skill lines in `system.md`.
   - Decide this in the "who decides" discussion.
4. **The rule-decides arm,** a thinking controller in the token proxy: its design is in [../adaptive_inference/README.md](../adaptive_inference/README.md). It needs a change to the main-track repo's `baseline/token_proxy.py`, and has no arm here yet.

## Not scripted yet

- **The SFT self-distillation baseline:**
  - thinking-on rollouts of the student's config on the 59 training tasks;
  - strip the thoughts, re-render with thinking off, and train;
  - then 2 runs here, as a new arm.
- **The Gate 1 diagnostic** (teacher scoring on `off`'s 70 trajectories), from [../opsd_assessment.md](../opsd_assessment.md).

## Tested (locally, 2026-10-08)

- **`baseline_report.py`, on the pulled v28 runs:**
  - it reproduces 19/70 for `off`, and 25 and 20 for `on` with 19 tasks in both;
  - run A against `off` gives 8 better and 2 worse, p = 0.109;
  - A takes 214 s per task, and `off`'s first source edit comes at 47 s.
  - A synthetic run checks the thinking count, the output tokens and the time span.
- **`opsd_baselines.sh`, against stub `sbatch` and `squeue` and stub main-track scripts** (40 checks, rerun after the logs moved to `logs/`):
  - status;
  - a first run: dry, `GO=1`, and blocked by a busy queue;
  - a later run, with `AFTER` and its report job, and a later `off` run with its Kaggle score;
  - a run without a summary, then `FORCE=1`;
  - an unknown arm and an unknown command;
  - `split`, `check-train` and `report`: dry, `GO=1`, a failing job, and `check-train` before the split;
  - the logs: each job's log lands in `logs/`, which is made if missing, and `logs/main_track` reaches the main track's logs without nesting on a second call;
  - `srun` is never called.
- **The wheelhouse versions** (42 checks, 2026-10-08, after v29), against the same stubs:
  - `opsd_baselines.sh`: on v29 the v28 runs and gates don't count; `off`'s first run compares with `base_v29_l40s` and `v1`'s with `off`'s v29 run; `on`'s zip run is still a first run beside `base_v29_l40s`; Kaggle scores come per version; a later run needs a v29 gate; it refuses when Kaggle is ahead or no version is known; `report` and `report 28`.
  - `baseline_report.py` on synthetic runs: one version's runs per table, a run named v29 that ran on v28 left out, Kaggle scores per version, and no table without a version.
  - The main track's `submit_baseline.sh`, its logic against stubs: runs and smokes named `_v29_`, a v28 gate that doesn't skip the v29 smoke, the default zip's Kaggle score on v28 only, the refusal when Kaggle is ahead, and a started run not queued again.
- **`make_split.py`,** on synthetic ids (70 / 59 / 39 / 20), and with a dev task outside the sound set, which it refuses.
- `bash -n` and `py_compile` on every script.
