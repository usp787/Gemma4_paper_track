# baseline/: the OPSD paper's baseline runs

<!-- status read from Explorer on 2026-10-10 (results/, submissions/, squeue, sacct); update the Status table after each run -->

This folder holds the baseline arms of the [OPSD study](../opsd_assessment.md): their status, the scripts that queue their runs, and the report that turns the runs into the paper's numbers.
- **The runs themselves use the main-track repo's eval tooling** ([Gemma4_kaggle_repo](https://github.com/usp787/Gemma4_kaggle_repo)): `post_train/v28/submit_baseline.sh`, `baseline/submit_eval.sh` and `baseline/diagnose_run.py`. They are scored exactly as the main track scores its variants.
- **Who does what:** Claude drafts and tests these scripts locally. You commit, push, pull on Explorer, and queue every job.

## Status (2026-10-10)

**We work on wheelhouse v30 now.** Kaggle moved to it on 2026-10-09 (14:43 UTC), and our re-alignment to it finished on 2026-10-10 (R and G below). v30 bumps swegemma to 0.2.11 ("support quantized e2b/e4b/12b models and consolidate discovery/registry"). By name and size no other wheel changed, the vllm wheel included. The main track's `post_train/RFT/v30.md` has v30's changes, its re-alignment and its results; its `v29.md` has v29's.
- **The v30 re-alignment** ran 2026-10-09 17:58 → 2026-10-10 00:32 EDT (jobs 10950493–10950499, then 10 shards), and every job completed.
  - Both envs were rebuilt with the v29 builds' PyPI versions pinned, so they differ from v29's only in swegemma. The smoke gate passed, and the LoRA cost is as on v29.
  - **`on`'s `base_v30_l40s` resolved 28/70,** no measurable change from v29's 26 (4 better, 2 worse, p = 0.69).
  - The grader check (10950499) kept the same 109 sound tasks, so the dev set, the split and the train check stand.
- **Step 1 is done:** `off`'s first v30 run, `baseline_nothink_v30_zip`, resolved **22/70** (fastapi 16/60, requests 6/9, httpx 0/1).
  - Its chain ran 2026-10-10 12:43–15:12 EDT, and every job completed: zip check 10969649, smoke 10969650, gate 10969651 (PASS: no reply thought, no budget sent), driver 10969652 (10 shards), report 10969653.
  - All 70 tasks finished, with no harness errors and no context overflow. Its report is `results/baseline_nothink_v30_zip/diagnosis.md`.
  - What the run shows is under [What the v30 runs say](#what-the-v30-runs-say).
- **Step 2 is `off_args`, written 2026-10-10.** It is `off`'s files plus rules for tool-call arguments in both prompts (`agents/baseline_nothink_args` in the main-track repo), and it replaces `v1` as step 2.
  - **Why:** half of `off`'s ranged `read_file` and `edit_file` calls were malformed, and later calls copied the slips ([below](#what-the-v30-runs-say)).
  - **The rules avoid what broke:** no line ranges or other optional arguments (other lines come through `sed -n` in `run_command`), `edit_file` always with all three arguments, and code with real line breaks.
  - **Why not built on `v1`:** its prompt asks for narrow line ranges, the pattern that broke. Its other changes (temperature 1.0, 100 calls, anti-repeat rules) act on the repeats, which come later. A single change keeps the effect on malformed calls measurable against `off`.
  - **Commands:** package it with `cd ~/Gemma4_kaggle_repo && sbatch slurm/package_submission.slurm agents/baseline_nothink_args` (~3 min). Once that job has ended, run `GO=1 bash baseline/opsd_baselines.sh run off_args` from this repo.
  - **Packaged** 2026-10-10 18:52 (job 10975791, git ebc522f) as `baseline_nothink_args_20261010-1852_ebc522f.zip` (sha256 532374430fe2), now pinned in `arms.tsv`. A first try (10975770) stopped at its check, since the checkout hadn't been pulled.
  - **Its first run is done: `baseline_nothink_args_v30_zip` resolved 24/70** (fastapi 16/60, requests 7/9, httpx 1/1).
    - Its chain ran 18:55–21:07 EDT, and every job completed: zip check 10975830, smoke 10975831, gate 10975832 (PASS: no reply thought, no budget sent), driver 10975833 (10 shards), report 10975834.
    - All 70 tasks finished, with no harness errors.
  - **What it shows:** malformed calls fell from 21.1% to 1.4%, while the score moved within noise. Details are under [What step 2 shows](#what-step-2-shows).
- **Env builds still need `PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt`.** The monthly /scratch purge deleted the conda envs' CA bundle on 2026-10-07, and that isn't fixed.
- **What v29 changed, and v30 keeps:** the code-graph tools, their descriptions and the handling of tool errors. A submitted patch also loses its hunks in test and config files before it is graded. Thinking works as on v28 for all our agents, which set `include_thoughts: true`.
- **Runs, smokes and gates are named after the wheelhouse version.** `arms.tsv`'s `{v}` stands for it, and `baseline_report.py` counts one version's runs. The v28 and v29 runs in the table below are their versions' record; `bash baseline/opsd_baselines.sh report 28` (or `29`) rebuilds their tables.
- **The Kaggle anchor** is `agentic_v1_think`'s unchanged zip (sha256 6418b8752004, 8/58 on v28), sent 2026-10-09 00:22 UTC, before v30 went live. It was still pending on 2026-10-10, so it may be scored on v30.
- **The main track's working baseline on v30 is `budgetfit`,** the public Budget-Fit bundle: 30/70 locally (`budgetfit_v30_zip`, 2026-10-10). It is no arm here: it thinks, and its budgets, prompt and sub-agents differ from `off`'s. Its runs share our 2 L40S slots.

**The model is complete again.** `/scratch/$USER/gemma4/models/gemma-4-31b-it-qat-w4a16-ct` had held 0 bytes since 2026-10-07 13:19 EDT, when the monthly /scratch purge deleted it. `fetch_model` (job 10927935) re-downloaded it on 2026-10-08, 17:13–17:19 EDT: `model.safetensors` is 23,265,352,448 bytes, and `chat_template.jinja` is the June one (16,934 bytes). The weights live on scratch, not in either checkout, so both tracks read the same copy. If `opsd_baselines.sh status` ever reports the model incomplete again, rerun `cd ~/Gemma4_kaggle_repo && sbatch slurm/fetch_model.slurm` (CPU, up to 2 h). The purge goes by file age, and the re-fetched files keep their archive's date (2026-06-05). So before the next one, on the first Tuesday of November (2026-11-03), run `touch /scratch/$USER/gemma4/models/gemma-4-31b-it-qat-w4a16-ct/*`.

**The split and the train check are done (2026-10-08).**
- `split` (job 10928660) wrote the five split files to `/scratch/$USER/gemma4/paper/split/`. Its first try (10928412) died of a broken pipe in `srun` before Python started, which is why `split` is now a batch job.
- The train check (job 10928765) took 6 min. Every training task's sandbox started in both modes: gold resolved 41/59 and empty 2/59, so 39 are sound, and no task's result changed since 2026-09-29.
  - Results: `/scratch/$USER/gemma4/results/opsd_train_check_10928765/`.
  - Its log went to `/scratch/$USER/gemma4/paper/logs/`, from before logs moved to this repo's `logs/`.

| Arm | Agent (zip sha256) | What it is | v30 runs (now): resolved of 70 | v29 runs (the record) | v28 runs (the record) | Kaggle | Next, on v30 |
|---|---|---|---|---|---|---|---|
| `off` | `baseline_nothink` (3909932c84e0) | thinking off; the host's sample prompt, temperature 0.2, 60 calls | `baseline_nothink_v30_zip` **22** (10-10) | none | `baseline_nothink_v28_zip` **19** (10-05) | v28: 3/58 | step 7, if it becomes the student |
| `on` | `baseline_noadapter` (d9db81dd3d76) | thinking on (budget 4096); otherwise the same files as `off` | `base_v30_l40s` **28** (10-10, from `agents/`) | `base_v29_l40s` **26** (10-09, from `agents/`) | `base_v28_zip` **25** (10-04), `base_v28_l40s` **20** (10-01, from `agents/`) | v28: 7/58 | step 4 |
| `v1` | `agentic_v1` (6cd3c3ec1a74) | thinking off; V1's rewritten prompt, temperature 1.0, 100 calls, 300-s commands, no analyzer | none | none | none | – | not scheduled: `off_args` took step 2 |
| `v1_think` | `agentic_v1_think` (6418b8752004) | thinking on (4096) with V1's prompt, but temperature 0.2, 60 calls and one more prompt line | none | none | none | v28: 8/58; the anchor, pending since 10-09 (may be scored on v30) | decision 2 |
| `v2` | `agentic_v2` (8dcadfe313d3) | V1 plus the advisor (thinking on, called by the model) and two skills: **the model decides** | none | none | `agentic_v2_v28_zip` **25** (10-06) | – | step 5 |
| `off_args` | `baseline_nothink_args` (532374430fe2) | thinking off; `off`'s files plus rules for tool-call arguments in both prompts | `baseline_nothink_args_v30_zip` **24** (10-10) | none | none | – | step 3 |

- **Not arms (v28):**
  - `base_v28_zip_paced` (18/70) is paced, so the report leaves it out.
  - `agentic_v1.1_think_v28_zip` (24 of 55 finished) ran on an 8-minute clock with 28 calls.
- **`v2`'s run isn't in the main-track README yet.** It ran 2026-10-06 21:17–23:56, and its `diagnosis.md` is in its run dir:
  - 25/70 resolved. Against `off`: 17 tasks in both, 8 only here, 2 only there; p = 0.11.
  - The advisor ran 68 times in 59 tasks, a median 38 s each. Thinking was 22% of output tokens.
  - 171 `run_skill_script` calls lacked `skill_name`, so the skills add malformed calls of their own.
  - 28 tasks hit the 4-minute clock.
  - Where its model calls the advisor, against bad calls, is in [../adaptive_inference/README.md](../adaptive_inference/README.md#what-the-traces-say).

### What the v30 runs say

From `baseline_report.py`, [trigger_replay.py](../adaptive_inference/trigger_replay.py) and a per-task split, on the traces pulled 2026-10-10. Each arm has one v30 run so far.

| Arm | Tool calls / task (median) | Agent s / task | First source edit (median s; never) | Stale repeats | Malformed | Next call bad, after a bad / a clean call |
|---|---|---|---|---|---|---|
| `off` (`baseline_nothink_v30_zip`) | 38 | 129 | 40; 26 | 45.0% | 21.1% | 86.5% / 14.8% |
| `on` (`base_v30_l40s`) | 19 | 213 | 121; 18 | 7.3% | 5.8% | 53.8% / 5.6% |

- **`on` against `off`:** 10 tasks better and 4 worse, so +6.0 (95% CI −1.0 to +13.0), sign test p = 0.18. On v28 the gap was the same 6 tasks.
- **`off` from v28 to v30, the same zip:** 19 → 22, with 5 tasks gained and 2 lost (p = 0.45).
  - Malformed calls fell from 29.8% to 21.1%, mostly through fewer `edit_file` calls whose code was parsed as argument names.
  - Stale repeats (47.2% → 45.0%) and the chain (88.2% → 86.5%) didn't move.
- **The two arms fail in opposite ways.** Here a loop is a task with at least half its calls bad, and a clean task has under 20% bad.
  - **`off`, 48 failures:**
    - 21 are loops, 13 are mixed and 14 are clean.
    - Only 9 reach the 4-min clock, and its mean agent time is 129 s.
    - 23 submit a failing patch.
    - 16 end with no patch, 14 of them after using 52–60 of their 60 tool calls.
  - **`on`, 42 failures:** 37 are clean, and 38 end on the clock. Its resolved tasks take a median 265 s from start to grading, against `off`'s 97 s.
- **Most of `off`'s malformed calls are one slip** (445 of 551): `read_file` with the keys `start_line"` and `end_line"`, each with a stray quote.
  - ADK drops unknown keys silently. So all 445 returned `ok` with the file's opening lines, not the range asked for.
  - The next call repeated it exactly 62% of the time, and was another malformed `read_file` 15% of the time. Only 4% recovered with a well-formed one.
  - Of `off`'s stale repeats, 42% re-run a command, 30% repeat a malformed `read_file`, 19% re-read with a well-formed one, and 8% repeat a malformed `edit_file`.
  - **A line range is what breaks:** 445 of 880 `read_file` calls with one were malformed, and none of 296 without. A task's first slip is rare (6% of ranged reads), but 82% of the ranged reads after it repeat it.
  - **`edit_file` breaks the same way:** 106 of its 210 calls were malformed. 96 of them carry JSON-escaped code (backslash-n and no real line breaks), and `old_string` is lost. 20% of a task's edits are escaped before its first escaped one, and 88% after it.
  - **The cause:** the model writes JSON inside Gemma 4's tool-call syntax, which has bare argument names and its own string delimiter. vLLM's parser then reads `start_line":151` as an argument named `start_line"`, and code outside the delimiters as more argument names.
- **`off`'s bad calls are spread over many tasks, while `on`'s sit in a few.**
  - `off`: its 10 worst tasks hold 44% of its bad calls, and its chain stays at 83% without the 5 worst.
  - `on`: its 3 worst hold 42%, and its chain falls from 53.8% to 19.4% without the 5 worst.
  - So pooled chain rates can't compare these arms; the report needs a per-task view.
- **On the 10 tasks only `on` resolved,** `off` looped on 5 (65–90% bad calls), was clean but wrong or slow on 4 (0–19%), and mixed on 1 (37%). On the 4 only `off` resolved, `on` ran out of the clock every time.
- **The code-graph tools answer again:** `off`'s `search_similar_code` came back empty 14 times in 86. On v28, every call did.

### What step 2 shows

`off_args` against `off`, from the same tools on the traces pulled 2026-10-10. Each has one run.

| Arm | Tool calls / task (median) | Agent s / task | First source edit (median s; never) | Stale repeats | Malformed | Next call bad, after a bad / a clean call |
|---|---|---|---|---|---|---|
| `off` | 38 | 129 | 40; 26 | 45.0% | 21.1% | 86.5% / 14.8% |
| `off_args` | 28 | 116 | 39; 16 | 28.2% | 1.4% | 80.4% / 8.6% |

- **The rules removed the malformed calls.** They fell from 551 to 32 (21.1% → 1.4%).
  - The model still asked for a line range 112 times, against 880, and read other lines through `sed -n` 279 times. Only 3 ranged reads broke: with no earlier slip in the context, there was nothing to copy.
  - `edit_file` calls without `old_string` fell from 82 to none. The 29 malformed edits left all carry escaped code, 14 of them in one task's loop.
  - Escaped code is still in half the edits (83 of 161). The harness unescapes well-formed ones, so most of them apply.
  - Side effect: 22 of the 279 `sed` reads hit `run_command`'s 5,000-character cap and came back cut.
- **The repeats halved as a side effect,** since many had been copies of malformed calls.
  - Stale repeats fell from 1,173 to 659, and tasks repeating in at least half their calls from 23 to 10.
  - Runs of 5 or more identical calls fell from 56 in 33 tasks to 17 in 16 tasks.
- **The score moved within noise:** 24 against 22, with 4 tasks gained and 2 lost (+2.0, 95% CI −3.0 to +7.0, p = 0.69).
  - Against `on`: 9 tasks worse and 5 better, so the gap fell from 6 tasks to 4.
- **What is left is a stuck loop, not a format problem.**
  - 88% of the remaining stale repeats are identical `run_command` calls, typically a `grep` or `find` that returns nothing or the same lines, re-run up to 53 times.
  - Once a call is bad, the next one still is 80% of the time.
- **The failures moved from loops to wrong patches.**
  - Of 46 failures, 7 are loops (`off`: 21), 16 are mixed and 23 are clean.
  - 20 of the clean ones submitted a wrong patch.
  - 11 tasks reached the clock (`off`: 9).
- **On the 9 tasks only `on` resolved,** `off_args` looped on 3 (50–83% bad calls), was clean but wrong on 4, and ran out of the clock on 2.

### What the v28 runs say

From `baseline_report.py` on the pulled runs, 2026-10-08. This is the v28 record: v29's tool changes, which v30 keeps, may move every column.

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
| R | `cd ~/Gemma4_kaggle_repo && PIP_CERT=/etc/pki/tls/certs/ca-bundle.crt bash baseline/submit_realign.sh`, the main track's re-alignment (for v30, the full commands are in its `post_train/RFT/v30.md`) | the envs, a smoke and its gate, the LoRA check, then `base_<v>_l40s`, `on`'s first run on that version. Done for v30 on 2026-10-10 (jobs 10950493–10950499): `base_v30_l40s` 28/70. On v29 (2026-10-09): 26/70 | builds ~45 min, then ~4 h |
| G | `cd ~/Gemma4_kaggle_repo && PINS=1 sbatch baseline/grader_check.slurm` | gold and empty patches on all 129 tasks: do the 109 sound tasks, and so the 70 dev tasks, still hold? After R's harness build. Done for v30 (job 10950499) and v29 (10933579): the same 109 both times | CPU, 10 min |
| 1 | `GO=1 bash baseline/opsd_baselines.sh run off` | `off`'s first v30 run, through the main track's chain (check, smoke, gate, run, report against `base_v30_l40s`). Done 2026-10-10 (jobs 10969649–10969653): 22/70 | ~3.5 h, ~6 L40S-h |
| 2 | `cd ~/Gemma4_kaggle_repo && sbatch slurm/package_submission.slurm agents/baseline_nothink_args`; once it has ended, `GO=1 bash baseline/opsd_baselines.sh run off_args` | package `off_args` (~3 min), then its first v30 run through the main track's chain, against `off`. Done 2026-10-10 (jobs 10975830–10975834): 24/70 | ~3.5 h, ~6 L40S-h |
| 3 | `GO=1 bash baseline/opsd_baselines.sh run off_args` | `off_args`'s second run, then its report | ~2.5 h, ~5 L40S-h |
| 4 | `GO=1 bash baseline/opsd_baselines.sh run on` | `on`'s zip run, `base_v30_zip`: its second v30 run | ~3.5 h, ~6 L40S-h |
| 5 | `GO=1 bash baseline/opsd_baselines.sh run v2`, twice | `v2`'s two v30 runs | ~6 h, ~11 L40S-h |
| 6 | `GO=1 bash baseline/opsd_baselines.sh run <thinking-on arm>`, twice | the thinking-on arm for the student's config (decision 2) | ~6 h |
| 7 | `GO=1 bash baseline/opsd_baselines.sh run off` | `off`'s second run, only if `off` becomes the student | ~2.5 h |
| – | `GO=1 bash baseline/opsd_baselines.sh report` | the tables above, for every arm, on our wheelhouse (`report 28`: the v28 record) | CPU, minutes |

- **R, then G, after every wheelhouse update, before any run on it.** Both are done for v30. Until R has run, `opsd_baselines.sh` refuses, since our envs are behind Kaggle's. A run before G could count tasks the new grader no longer scores.
- **Runs go one after another.** Only 2 L40S jobs of ours run at a time, and the main track shares them.
  - A first run (a chain) needs an empty eval queue: `submit_baseline.sh` refuses to start otherwise.
  - A later run can wait for the one before: `AFTER=<driver job id>`.
  - `opsd_baselines.sh` refuses to queue while Kaggle's wheelhouse differs from ours, and a first run passes `PACED=0`, since only unpaced runs count here.
- **Steps 4–7 and decisions 2–3 were planned around `v1`.** They are re-planned once `off_args` has run.
- **Freeze the student config after step 3:**
  - `off_args`, if both its v30 runs finish all 70 tasks with no harness errors and resolve at least what `off`'s v30 run does (22);
  - otherwise `off`.
- **Then start the round-0 rollouts** for OPSD, on v30 (A100, `gpu` partition, in parallel with steps 4–7). For example, with `off_args` as the student:
  ```bash
  cd ~/Gemma4_kaggle_repo && PINS=1 TIME=02:00:00 bash baseline/submit_rollouts.sh agents/baseline_nothink_args /scratch/$USER/gemma4/paper/split/train59.txt opsd_r0_off_args 4 1
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
  - These are the dev repos' tasks whose gold patch passes and empty patch fails (grader check 10686757 on v28; 10933579 on v29 and 10950499 on v30 found the same).
  - The OPSD split trains on the other 59: rich 48, and the 11 fastapi and requests tasks the grader can't score.
- **The environment:**
  - the training-reward env (`PINS=1`), unpaced, one task at a time on one L40S;
  - one wheelhouse version, v30 now (Kaggle's since 2026-10-09, our envs' since 2026-10-10), and the agent extracted from its zip; `run_meta.json` records both the version and the zip's sha256;
  - the report counts one wheelhouse version at a time, and leaves out any run that is paced or not `PINS=1`.
- **Runs:** an arm's first run is `<agent>_<v>_zip`, then `<first run>_r<k>` (`arms.tsv`, where `{v}` is the version: `_v30_zip` now; `_v29_zip` and `_v28_zip` in the record).
- **Noise:** one run varies by about ±2.2 tasks, and the difference between two runs by about ±3.1 (`post_train/v28/README.md`). So every arm gets 2 runs, and the report pairs tasks across arms.
  - `on`'s four runs from v28 to v30 (20, 25, 26, 28) agree: 16 tasks resolve in all four, 36 in none, and 18 flip, so 24.8 ± 3.4 per run (`post_train/RFT/v30.md`).
- **Metrics:**
  - resolved per run;
  - against the reference arm, task by task: better, worse, the difference with a bootstrap CI, and a sign test;
  - from the traces: tool calls and agent time per task, the first source edit, stale repeats, malformed calls, and how bad calls chain;
  - from `calls.jsonl`: output tokens and the thinking share.
  - The trace rules match [../adaptive_inference/trigger_replay.py](../adaptive_inference/trigger_replay.py).

## Files

| File | What it does |
|---|---|
| [arms.tsv](arms.tsv) | one row per arm: agent, zip (`-` until pinned), run names with `{v}` for the wheelhouse version, reference arm, Kaggle scores as `v<N>:k/n`. Add a row for a new arm. |
| [opsd_baselines.sh](opsd_baselines.sh) | `status`, `run <arm>`, `split`, `check-train` and `report [N]`, all on our wheelhouse version. Nothing is queued without `GO=1`. An unpinned zip is the agent's only packaged zip. |
| [make_split.py](make_split.py) | writes `dev70.txt`, `train59.txt`, `train59_sound.txt`, `train59_unsound.txt` and `split.json` to scratch, after checking the counts and that dev is all sound |
| [train_check.slurm](train_check.slurm) | gold and empty patches on the training tasks, in the `PINS=1` env; lists what changed since the 2026-09-29 grader check |
| [baseline_report.py](baseline_report.py) | the tables, as Markdown, plus JSON for figures. stdlib only; runs on scratch or on pulled run dirs |

## Decisions pending

1. **The student config:** `off_args` or `off`, by the rule under Order of work. Decisions 2 and 3 were written around `v1`, and are revisited once `off_args` has run.
2. **`v1`'s thinking-on arm.** `v1_think` differs from `v1` in more than thinking: temperature 0.2 against 1.0, 60 calls against 100, and one prompt line.
   - (a) Use it as it is. It has a Kaggle score (8/58 on v28; as the anchor, pending since 2026-10-09), but every comparison carries that confound.
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

## Tested (locally, 2026-10-10)

- **`opsd_baselines.sh` with an unpinned zip,** against a stub scratch root, stub `squeue` and `sbatch`, and a stub `submit_baseline.sh`:
  - no zip: `run` and `status` print the packaging command, and `GO=1` refuses;
  - one zip: the script uses it and says so, skipping `.zip.invalid` files, and `GO=1` hands the chain that zip and `REF=baseline_nothink_v30_zip`;
  - two zips: it refuses and lists them;
  - pinned arms are unchanged.
- **`agents/baseline_nothink_args`:**
  - its YAML parses;
  - it differs from `baseline_nothink` only in the two prompts and `agent.yaml`'s header;
  - the prompts hold no braces, since ADK reads `{name}` in an instruction as a session-state variable;
  - packaging on Explorer validates it with the host's compiler.
- `bash -n` on `opsd_baselines.sh`.

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
