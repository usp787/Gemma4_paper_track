# When Coding Agents Repeat

Code and records for the paper **When Coding Agents Repeat: Persistence, Selection, and the Limits of Loop Breaking** (*A study of repeated tool calls in Gemma 4's coding agent*). It is our entry to Kaggle's [Google – The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper).

- **Current manuscript:** [PDF](paper/submission_v3/submission_v3_refine.pdf) and [Word source](paper/submission_v3/submission_v3_refine.docx), with figures and the word-count record in [paper/submission_v3/](paper/submission_v3/).
- **Earlier Kaggle submission:** [submission_v1.pdf](paper/submission_v1/submission_v1.pdf), with its source and Writeup files in [paper/submission_v1/](paper/submission_v1/).
- **Author:** Jiarui Zha (zha.j@northeastern.edu)

## The study in brief

Gemma 4 31B (int4), running as a coding agent with thinking off, often sends the same tool call again and again until its budget runs out. We study why, on 109 tasks in the harness of Kaggle's Gemma 4 Developer Agent competition. We use two methods:
- one-step replays at recorded histories that we edit;
- whole-task runs of three levers against a control.

We find:

1. **Lock-in is mostly selection.** A proxy holds the model's request fixed while a run lasts. With it, the chance of one more identical call still rises from 78% after two calls to 99% after ten. Identical copies in the context add to it, on their exact tokens.
2. **Long runs end from outside.** At T = 0.2, the agent never ended a run of five or more identical calls itself, in 1,342 chances. The harness's budget warning or a context compaction ended them.
3. **The pattern of the context cues the copy, not what the results say.** Rewriting the results leaves the copy where it was. Where the model chose not to think, a forced thought lets the next call escape 58% of the time, against 11%.
4. **No lever that breaks loops measurably changes resolution.** Over two passes, T = 1 resolves 71 of 218 task runs and the proxy 73, against the control's 72. An empty thought before every call resolves 38 of 109 in its one pass.

Gemma 4 12B, run as a generality check, starts runs more readily but locks in at the same rates.

## Where each claim comes from

Every number in the paper is quoted from a record in [ablation/](ablation/), [loops/](loops/) or [gate1/](gate1/). [findings.md](findings.md) gathers them claim by claim, with caveats, the claims withdrawn on the way and the limits.

| Paper | Records | Code |
|---|---|---|
| §2 Setting and measures | [findings.md](findings.md) (setting); [ablation_G2.md](ablation/ablation_G2.md) §1 (replay check, 0.639 against 0.634) | [agent/](agent/), [loops/loop_report.py](loops/loop_report.py) |
| §3 Loops are common and mark lost runs | [ablation_G3.md](ablation/ablation_G3.md) §3, §5.3–5.7; [ablation_control.md](ablation/ablation_control.md) §3 | [loops/loop_report.py](loops/loop_report.py), [baseline/baseline_report.py](baseline/baseline_report.py) |
| §4 Selected, then sealed; Fig. 1 | [ablation_G4.md](ablation/ablation_G4.md) §5.1 (proxy); [ablation_G3.md](ablation/ablation_G3.md) §5.1 (control); [ablation_G1.md](ablation/ablation_G1.md) (copies at fixed histories); [ablation_control.md](ablation/ablation_control.md) §5.1 (12B) | [agent/token_proxy.py](agent/token_proxy.py), [loops/dose.py](loops/dose.py), [paper/figures/](paper/figures/) |
| §5 Long runs end from outside | [ablation_G3.md](ablation/ablation_G3.md) §5.1–5.2, §5.6; [ablation_G4.md](ablation/ablation_G4.md) §5.2; [ablation_control.md](ablation/ablation_control.md) §5.1–5.2 | [loops/loop_report.py](loops/loop_report.py) |
| §6 What cues the copy | [ablation_G2.md](ablation/ablation_G2.md) (rewritten results, empty thought); [gate1/README.md](gate1/README.md) (controller's note); [loops/loops_reason_10_13.md](loops/loops_reason_10_13.md) §5 (forced thought) | [loops/feedback.py](loops/feedback.py), [gate1/gate1.py](gate1/gate1.py) |
| §7 Breaking loops over whole tasks; Table 1 | [ablation_G3.md](ablation/ablation_G3.md), [ablation_G4.md](ablation/ablation_G4.md), [ablation_control.md](ablation/ablation_control.md) | [loops/submit_loops.sh](loops/submit_loops.sh), [loops/arms.tsv](loops/arms.tsv), [agent/](agent/) |
| §8 Related work | [loops/literature.md](loops/literature.md) | – |

### Names in the records

The records use working labels that the paper doesn't:

| Paper | Records |
|---|---|
| the control | `ctrl`; the agent `off_args` (`baseline_nothink_args`) |
| T = 1 | `ctrl_t1`, in G3 (temperature and replication) |
| an empty thought before every call | `block`, the always-block template |
| the proxy that holds the request fixed | `dedup`, in G4 |
| Gemma 4 12B | `ctrl_12b`, the 12B control |
| identical copies at fixed histories | G1, the dose test |
| rewritten results | G2, the feedback factorial |
| the forced thought, against the model free to think | T-open against T-think |
| the earlier serving stack of the replays | wheelhouse v30. The whole-task runs used v31. |

## Layout

- **[agent/](agent/):** the agent and the request proxy behind every whole-task run, copied from our main-track repo. It also explains how each lever changes the control.
- **[loops/](loops/):** the loop study.
  - The whole-task runs: `submit_loops.sh`, `loops.slurm` and `arms.tsv`. Each arm is built by `make_template.py`, `make_t1_zip.py` or `make_model_zip.py`.
  - The one-step replays: `dose.py` (copies at fixed histories) and `feedback.py` (rewritten results).
  - The loop metrics: `loop_report.py`.
  - The task lists: `tasks/dev_70.txt` (fastapi 60, requests 9, httpx 1) and `tasks/rich_39.txt`.
  - The design and its predictions (`README.md`), the mechanisms (`loops_reason_10_13.md`) and the literature review (`literature.md`).
- **[ablation/](ablation/):** one record per experiment, each with its design, runs, results with intervals, what it shows and what it doesn't.
- **[gate1/](gate1/):** `gate1.py`, the replay engine. It scores recorded histories under edited contexts, including the forced thought and the controller's note. The other replays import its call labels.
- **[baseline/](baseline/):** `baseline_report.py`, resolution per arm, paired over tasks. Also the launchers of the earlier OPSD baselines (see Background).
- **[adaptive_inference/](adaptive_inference/):** `trigger_replay.py`, which holds the shared rules for stale and malformed calls.
- **[paper/](paper/):** the paper, its figures with their data (`figures/fig_data.py` names the record of every number; `figures/make_figures.py` draws them), earlier drafts and the outline.

## Reproducing

**What you need:**
- **From Kaggle:** the competition's data: the harness (`adk_submission`), wheelhouse v31 with its vLLM, the checkpoints `gemma-4-31b-it-qat-w4a16-ct` and `gemma-4-12b-it-qat-w4a16-ct`, and the tasks. Join the [main competition](https://www.kaggle.com/competitions/gemma-4-developer-agent/data) to get them. Its rules don't allow redistributing them, so none of them are here.
- **A GPU:** one A100 80 GB runs 4 tasks at once. One pass of one arm over the 109 tasks took about 1.5 hours. The replays need a GPU too.
- **Python:** the report scripts and the zip builders use the standard library only. The replays run in the vLLM environment, and the figures need matplotlib.

**The steps:**
1. **Whole-task runs.** For each arm, build its zip or chat template, then serve vLLM as in [agent/README.md](agent/README.md), with one `token_proxy.py` per task lane (`--dedup 2` for the proxy arm). Run the harness on the 109 tasks in [loops/tasks/](loops/tasks/), two passes per arm. Our launcher is [loops/submit_loops.sh](loops/submit_loops.sh), which starts without arguments as a dry run. It is written for our SLURM cluster (Northeastern's Explorer) and calls our main-track launcher, which isn't released. [agent/README.md](agent/README.md) lists everything that launcher changes about a run.
2. **Metrics.** `python loops/loop_report.py --arm ctrl=<run>,<run_r2> --arm ctrl_t1=<run>,<run_r2> --pair ctrl_t1:ctrl --group dev=loops/tasks/dev_70.txt --group rich=loops/tasks/rich_39.txt --json loops.json`. Resolution comes from `baseline/baseline_report.py --arms loops/arms.tsv`.
3. **One-step replays.** [loops/dose.py](loops/dose.py), [loops/feedback.py](loops/feedback.py) and [gate1/gate1.py](gate1/gate1.py) start from a run's recorded `calls.jsonl` and traces. Each file's header gives its commands.
4. **Figures.** `python paper/figures/make_figures.py`.

**What this repo holds, and what it doesn't:**
- No competition data: no task statements, gold or test patches, code graphs or embeddings.
- No agent traces, which quote task text and repository code.
- Only task IDs and aggregate results. The raw runs stay on our cluster.

## Background: how the study started

The project began on 2026-10-07 as on-policy self-distillation (OPSD): training Gemma 4 31B into a thinking-off agent that acts like the thinking-on model.

Before any training, Gate 1 ([gate1/](gate1/)) scored recorded thinking-off runs under candidate teachers, to see where a teacher would correct the student. The thinking teacher shared the student's loop: it put less KL divergence on the failure tokens than on routine ones, a ratio of 0.51.

On 2026-10-12 the work turned into this loop study, and no training was run. The OPSD plan and its baselines stay as background: [opsd_assessment.md](opsd_assessment.md), [baseline/README.md](baseline/README.md), [gate1/README.md](gate1/README.md) and [adaptive_inference/README.md](adaptive_inference/README.md).

## Use of AI tools

Claude (Anthropic) helped write the experiment and analysis code, these documents, and drafts of the paper's text from the author's records. The paper's own statement gives the details.

## License

[Apache License 2.0](LICENSE). Copyright 2026 Jiarui Zha. The competition's data, the harness and the Gemma 4 weights are not included and keep their own terms.
