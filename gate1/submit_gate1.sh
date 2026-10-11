#!/bin/bash
# Queue Gate 1 (gate1/README.md) on Explorer. Nothing is queued without GO=1: by default it prints the jobs it would
# queue and what blocks them. Run it on the login node, from this repo's checkout; it only uses ls, squeue and sbatch.
#   bash gate1/submit_gate1.sh [run ...]      the runs to score (dirs in /scratch/$USER/gemma4/results);
#                                             default: baseline_nothink_args_v30_zip, off_args's first v30 run
# The chain, each job waiting for the one before:
#   1. prep   (CPU, short partition, ~15 min): states, prompts, and the checks of the prompt edits against the model's
#             chat template. A failed check stops the chain. Its prep.md is worth a look before the GPU hours.
#   2. core   (one A100 80 GB, ~3-4 h): generate (T-think's thoughts and its own calls), then score the contexts
#             S C-none T-think T-hint note dedup.
#   3. patch  (one A100, ~2.5 h, only with PATCH=1): score the patched teachers T-patch T-patch-on T-loc C-wrong.
#   4. report (CPU, minutes): gate1.md and gate1.json, once the jobs before it have ended either way.
# The work dir is /scratch/$USER/gemma4/paper/gate1/$NAME. A rerun with the same NAME resumes: generate and score skip
# what is already written, e.g. after a job hit its time limit: STEPS="core report" GO=1 bash gate1/submit_gate1.sh.
# Env:
#   NAME     work dir name (default gate1_<first run>)
#   STEPS    any of "prep core patch report" (default: prep core report, and patch with PATCH=1)
#   PATCH=1  add the patched teachers (they read the gold patches from /scratch/$USER/gemma4/data/tasks.jsonl)
#   CORE_TIME (06:00:00), PATCH_TIME (05:00:00), GPU (a100), PARTITION (gpu), CONSTRAINT (a100@80g; empty for none)
#   PREP_ARGS, GEN_ARGS, SCORE_ARGS, REPORT_ARGS: more gate1.py arguments, e.g. PREP_ARGS="--scale 0.1" for a try
#   AFTER=<job id>: start the chain once that job has ended, e.g. step 3's driver, to score its run as well
#   FORCE=1: queue even though gate1 jobs of ours are queued or running (they would write the same work dir)
set -euo pipefail
cd "$(dirname "$0")/.."
GO=${GO:-0}
ME=${USER:-$(id -un)}
R=${GEMMA4_ROOT:-/scratch/$ME/gemma4}
MAIN=${MAIN:-$HOME/Gemma4_kaggle_repo}
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || RUNS=(baseline_nothink_args_v30_zip)
NAME=${NAME:-gate1_${RUNS[0]}}
WORK=$R/paper/gate1/$NAME
PATCH=${PATCH:-0}
STEPS=${STEPS:-prep core$([ "$PATCH" = 1 ] && echo " patch" || true) report}
GPU=${GPU:-a100}; PARTITION=${PARTITION:-gpu}; CONSTRAINT=${CONSTRAINT-a100@80g}
CON=(); [ -n "$CONSTRAINT" ] && CON=(--constraint="$CONSTRAINT")
GPU_RES=(-p "$PARTITION" --gres=gpu:"$GPU":1 "${CON[@]}" -c 8 --mem=128G)
has() { [[ " $STEPS " == *" $1 "* ]]; }

blocked=()
dirs=()
for r in "${RUNS[@]}"; do
  d=$R/results/$r
  dirs+=("$d")
  if [ ! -d "$d" ]; then blocked+=("no run dir $d"); continue; fi
  ls "$d"/shard_*/calls.jsonl >/dev/null 2>&1 || blocked+=("$r has no shard_*/calls.jsonl")
  [ -f "$d/summary_all.json" ] || [ -n "${AFTER:-}" ] || blocked+=("$r has no summary_all.json: is it still running? (AFTER=<its driver> waits for it)")
done
[ -s "$R/models/gemma-4-31b-it-qat-w4a16-ct/model.safetensors" ] || blocked+=("the model is missing: (cd $MAIN && sbatch slurm/fetch_model.slurm)")
[ -x "$R/venvs/vllm_host/bin/python" ] || blocked+=("no venv $R/venvs/vllm_host")
[ -f "$MAIN/baseline/gpu_guard.sh" ] || blocked+=("no main-track checkout at $MAIN (set MAIN=...)")
if ! has prep && { has core || has patch; } && [ ! -s "$WORK/states.jsonl" ]; then
  blocked+=("no $WORK/states.jsonl yet: add prep to STEPS")
fi
if has patch && ! has prep && [ ! -s "$WORK/priv.jsonl" ]; then
  blocked+=("no gold patches in $WORK/priv.jsonl: prep reads them from $R/data/tasks.jsonl")
fi
running=$(squeue -h -u "$ME" -o "%i %j %T" 2>/dev/null | awk '$2 ~ /^gate1_/' || true)
if [ -n "$running" ] && [ "${FORCE:-0}" != 1 ]; then
  blocked+=("gate1 jobs of ours are queued or running ($(echo "$running" | awk '{print $1"("$2")"}' | paste -sd' ')); FORCE=1 queues anyway")
fi

echo "Gate 1: runs ${RUNS[*]}; work dir $WORK; steps: $STEPS"
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
  args=(--parsable -J "gate1_$label" "$@")
  [ -n "$prev" ] && args+=(--dependency="$dmode:$prev" --kill-on-invalid-dep=yes)
  echo "  $label: env ${envs[*]} sbatch ${args[*]} gate1/gate1.slurm"
  if [ "$GO" = 1 ]; then
    j=$(env "${envs[@]}" sbatch "${args[@]}" gate1/gate1.slurm)
    prev=${j%%;*}
    echo "    -> job $prev; log logs/gate1_${label}_$prev.out"
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
if has core; then
  queue core "$mode" "${COMMON[@]}" STEPS="generate score" CONTEXTS=S,C-none,T-think,T-hint,note,dedup \
    GEN_ARGS="${GEN_ARGS:-}" SCORE_ARGS="${SCORE_ARGS:-}" -- "${GPU_RES[@]}" -t "${CORE_TIME:-06:00:00}"
  mode=afterany   # a core job cut by its time limit leaves resumable files; patch and report go on
fi
if has patch; then
  queue patch "$mode" "${COMMON[@]}" STEPS=score CONTEXTS=S,T-patch,T-patch-on,T-loc,C-wrong \
    SCORE_ARGS="${SCORE_ARGS:-}" -- "${GPU_RES[@]}" -t "${PATCH_TIME:-05:00:00}"
  mode=afterany
fi
if has report; then
  queue report afterany "${COMMON[@]}" STEPS=report REPORT_ARGS="${REPORT_ARGS:-}" -- -p short -c 2 --mem=16G -t 00:30:00
fi
if [ "$GO" = 1 ]; then
  echo "results: $WORK/gate1.md once the report job ends (prep's checks: $WORK/prep.md)"
else
  [ ${#blocked[@]} -eq 0 ] || printf 'blocked: %s\n' "${blocked[@]}"
  echo "dry run: nothing queued. GO=1 queues these jobs."
fi
exit 0
