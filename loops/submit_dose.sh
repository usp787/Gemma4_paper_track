#!/bin/bash
# Queue the dose test (loops/README.md section 3, G1) on Explorer. Nothing is queued without GO=1: by default it
# prints the jobs it would queue and what blocks them. Run it on the login node, from this repo's checkout; it only
# uses ls, squeue and sbatch.
#   bash loops/submit_dose.sh [run ...]      the runs to pick histories from (dirs in /scratch/$USER/gemma4/results);
#                                            default: baseline_nothink_args_v30_zip (Gate 1's) and loops_ctrl_v30_a100
# The chain, each job waiting for the one before:
#   1. prep   (CPU, short partition, minutes): picks the histories, builds the added blocks and checks them against
#             the model's chat template. A failed check stops the chain. Its prep.md is worth a look first.
#   2. gpu    (one A100 80 GB each, JOBS jobs, 2-h limit): score, then sample. Both resume, so a later job finishes
#             what an earlier one left, or exits before starting vLLM. Expect ~1 h for 60 histories.
#   3. report (CPU, minutes): dose.md and dose.json, once the jobs before it have ended either way.
# The work dir is /scratch/$USER/gemma4/paper/loops/$NAME. A rerun with the same NAME resumes, e.g. after a job hit its
# time limit: STEPS="gpu report" GO=1 bash loops/submit_dose.sh.
# Env:
#   NAME     work dir name (default dose_<first run>)
#   STEPS    any of "prep gpu report" (default: all three)
#   JOBS (2), TIME (02:00:00), GPU (a100), PARTITION (gpu), CONSTRAINT (a100@80g; empty for none)
#   ARMS     the arms to score and sample, comma-separated (default: all of dose.py's)
#   PREP_ARGS, SCORE_ARGS, SAMPLE_ARGS, REPORT_ARGS: more dose.py arguments, e.g. PREP_ARGS="--target 20" for a try
#   AFTER=<job id>: start the chain once that job has ended
#   FORCE=1: queue even though dose jobs of ours are queued or running (they would write the same work dir)
#   For tests: GEMMA4_ROOT=<dir> replaces /scratch/$USER/gemma4.
set -euo pipefail
cd "$(dirname "$0")/.."
GO=${GO:-0}
ME=${USER:-$(id -un)}
R=${GEMMA4_ROOT:-/scratch/$ME/gemma4}
MAIN=${MAIN:-$HOME/Gemma4_kaggle_repo}
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || RUNS=(baseline_nothink_args_v30_zip loops_ctrl_v30_a100)
NAME=${NAME:-dose_${RUNS[0]}}
WORK=$R/paper/loops/$NAME
STEPS=${STEPS:-prep gpu report}
JOBS=${JOBS:-2}; TIME=${TIME:-02:00:00}
GPU=${GPU:-a100}; PARTITION=${PARTITION:-gpu}; CONSTRAINT=${CONSTRAINT-a100@80g}
CON=(); [ -n "$CONSTRAINT" ] && CON=(--constraint="$CONSTRAINT")
has() { [[ " $STEPS " == *" $1 "* ]]; }

blocked=()
for s in $STEPS; do
  [[ " prep gpu report " == *" $s "* ]] || blocked+=("unknown step $s (prep, gpu, report)")
done
[[ $JOBS =~ ^[1-9]$ ]] || blocked+=("JOBS=$JOBS is not 1-9")
dirs=()
for r in "${RUNS[@]}"; do
  d=$R/results/$r
  dirs+=("$d")
  if [ ! -d "$d" ]; then blocked+=("no run dir $d"); continue; fi
  ls "$d"/shard_*/calls.jsonl >/dev/null 2>&1 || blocked+=("$r has no shard_*/calls.jsonl")
  ls "$d"/shard_*/traces/trace_*.json >/dev/null 2>&1 || blocked+=("$r has no shard_*/traces")
done
[ -s "$R/models/gemma-4-31b-it-qat-w4a16-ct/model.safetensors" ] || blocked+=("the model is missing: (cd $MAIN && sbatch slurm/fetch_model.slurm)")
[ -x "$R/venvs/vllm_host/bin/python" ] || blocked+=("no venv $R/venvs/vllm_host")
[ -f "$MAIN/baseline/gpu_guard.sh" ] || blocked+=("no main-track checkout at $MAIN (set MAIN=...)")
if ! has prep && { has gpu || has report; } && [ ! -s "$WORK/states.jsonl" ]; then
  blocked+=("no $WORK/states.jsonl yet: add prep to STEPS")
fi
running=$(squeue -h -u "$ME" -o "%i %j %T" 2>/dev/null | awk '$2 ~ /^dose_/' || true)
if [ -n "$running" ] && [ "${FORCE:-0}" != 1 ]; then
  blocked+=("dose jobs of ours are queued or running ($(echo "$running" | awk '{print $1"("$2")"}' | paste -sd' ')); FORCE=1 queues anyway")
fi

echo "dose test: runs ${RUNS[*]}; work dir $WORK; steps: $STEPS; arms: ${ARMS:-all}"
if [ ${#blocked[@]} -gt 0 ] && [ "$GO" = 1 ]; then
  printf 'blocked: %s\n' "${blocked[@]}"
  echo "FAIL: nothing queued"
  exit 1
fi
prev=${AFTER:-}
mode=afterany   # the first job waits for AFTER either way; then prep must succeed, and the rest run either way
queue() {   # queue <label> <dependency mode> <env assignments...> -- <sbatch args...>: prints and (GO=1) queues a job
  local label=$1 dmode=$2 envs=() args=() j
  shift 2
  while [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  args=(--parsable -J "dose_$label" "$@")
  [ -n "$prev" ] && args+=(--dependency="$dmode:$prev" --kill-on-invalid-dep=yes)
  echo "  $label: env ${envs[*]} sbatch ${args[*]} loops/dose.slurm"
  if [ "$GO" = 1 ]; then
    j=$(env "${envs[@]}" sbatch "${args[@]}" loops/dose.slurm)
    prev=${j%%;*}
    echo "    -> job $prev; log logs/dose_${label}_$prev.out"
  else
    prev="<$label>"
  fi
}
COMMON=(WORK="$WORK" MAIN="$MAIN")
if has prep; then
  queue prep "$mode" "${COMMON[@]}" STEPS=prep RUNS="${dirs[*]}" PREP_ARGS="${PREP_ARGS:-}" -- \
    -p short -c 4 --mem=32G -t 01:00:00
  mode=afterok
fi
if has gpu; then
  for k in $(seq 1 "$JOBS"); do
    queue "gpu$k" "$mode" "${COMMON[@]}" STEPS="score sample" ARMS="${ARMS:-}" SCORE_ARGS="${SCORE_ARGS:-}" \
      SAMPLE_ARGS="${SAMPLE_ARGS:-}" -- -p "$PARTITION" --gres=gpu:"$GPU":1 "${CON[@]}" -c 8 --mem=128G -t "$TIME"
    mode=afterany   # a job cut by its time limit leaves resumable files; the next one and the report go on
  done
fi
if has report; then
  queue report afterany "${COMMON[@]}" STEPS=report REPORT_ARGS="${REPORT_ARGS:-}" -- -p short -c 2 --mem=16G -t 00:30:00
fi
if [ "$GO" = 1 ]; then
  echo "results: $WORK/dose.md once the report job ends (prep's checks: $WORK/prep.md)"
else
  [ ${#blocked[@]} -eq 0 ] || printf 'blocked: %s\n' "${blocked[@]}"
  echo "dry run: nothing queued. GO=1 queues these jobs."
fi
exit 0
