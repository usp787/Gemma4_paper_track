#!/bin/bash -l
#SBATCH --job-name=loops
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# The loop study's CPU steps (loops/README.md), one per job; loops/submit_loops.sh queues them:
#   STEP=template  make_template.py: the always-block template from the model's own, checked on probe conversations
#                  (OUT, the template file). A failed check fails the job, which cancels the runs after it.
#   STEP=t1zip     make_t1_zip.py: G3's T = 1 zip from the pinned one (SRC, the pinned zip; OUT, the T = 1 zip;
#                  REPORT, its check.md). Built if missing, else only checked. A failed check fails the job.
#   STEP=report    baseline/baseline_report.py with loops/arms.tsv, for the dev and the rich tasks, then
#                  loop_report.py on every run (RUNS, space-separated run dirs, ctrl's first), into DIR. With ARMSPEC
#                  (space-separated NAME=RUN[,RUN...]) and PAIRS (space-separated A:B), loop_report.py also pools each
#                  arm's passes and compares the pairs task by task.
# Env: IDS_DEV, IDS_RICH (the task id files). Submit from this repo's root, where logs/ is.

ROOT=/scratch/$USER/gemma4
MODEL=$ROOT/models/gemma-4-31b-it-qat-w4a16-ct
PY=$ROOT/venvs/vllm_host/bin/python            # transformers 5, the version vLLM renders the template with
PRELOAD=$ROOT/envs/py312/lib/libstdc++.so.6    # conda sqlite3 vs system libstdc++, as in the main track's jobs
echo "job=$SLURM_JOB_ID node=$(hostname) git=$(git rev-parse --short HEAD 2>/dev/null) step=${STEP:-} start=$(date -Is)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case ${STEP:-} in
  template)
    LD_PRELOAD=$PRELOAD "$PY" loops/make_template.py --model "$MODEL" --out "${OUT:?set OUT to the template file}" \
      || { echo "FAIL: template"; exit 1; } ;;
  t1zip)
    LD_PRELOAD=$PRELOAD "$PY" loops/make_t1_zip.py --src "${SRC:?set SRC to the pinned zip}" \
      --out "${OUT:?set OUT to the T=1 zip}" --temperature "${TEMP:-1.0}" ${REPORT:+--report "$REPORT"} \
      || { echo "FAIL: t1zip"; exit 1; } ;;
  report)
    DIR=${DIR:?set DIR to the report dir}
    mkdir -p "$DIR"
    V=$(cat "$ROOT/wheelhouse/.version")
    for g in dev rich; do
      if [ "$g" = dev ]; then ids=${IDS_DEV:?set IDS_DEV}; else ids=${IDS_RICH:?set IDS_RICH}; fi
      LD_PRELOAD=$PRELOAD "$PY" baseline/baseline_report.py --arms loops/arms.tsv --results "$ROOT/results" \
        --ids "$ids" --wheelhouse "$V" --json "$DIR/baseline_$g.json" > "$DIR/baseline_$g.md" \
        || { echo "FAIL: baseline_report.py ($g)"; exit 1; }
    done
    ARGS=()
    for s in ${ARMSPEC:-}; do ARGS+=(--arm "$s"); done
    for p in ${PAIRS:-}; do ARGS+=(--pair "$p"); done
    # shellcheck disable=SC2086  # RUNS is a list of run dirs
    LD_PRELOAD=$PRELOAD "$PY" loops/loop_report.py ${RUNS:+--runs $RUNS} "${ARGS[@]}" --group "dev=$IDS_DEV" \
      --group "rich=$IDS_RICH" --json "$DIR/loops.json" > "$DIR/loops.md" || { echo "FAIL: loop_report.py"; exit 1; }
    cat "$DIR/baseline_dev.md" "$DIR/baseline_rich.md" "$DIR/loops.md"
    echo "report: $DIR/{baseline_dev,baseline_rich,loops}.md" ;;
  *) echo "FAIL: STEP=${STEP:-} (template, t1zip or report)"; exit 1 ;;
esac
echo "end=$(date -Is)"
