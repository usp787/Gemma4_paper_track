#!/bin/bash -l
#SBATCH --job-name=fb
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append

# The feedback factorial (loops/README.md section 3, G2): the steps of loops/feedback.py, in one job.
# loops/submit_feedback.sh queues it with each step's resources, which this file leaves to the command line:
# - prep and report: CPU, e.g. -p short -c 4 --mem=32G -t 01:00:00;
# - score and sample: one A100 80 GB, e.g. -p gpu --gres=gpu:a100:1 --constraint=a100@80g -c 8 --mem=128G
#   -t 02:00:00. Not an H200 or H100: vLLM can't serve the int4 model there (gate1/gate1.slurm).
# Env: WORK (the work dir, required), STEPS (default "score sample"), RUNS (prep's run dirs, space-separated), ARMS
# (score's and sample's arms, comma-separated; default all), MAIN (the main-track checkout, for its gpu_guard.sh), and
# more feedback.py arguments per step: PREP_ARGS, SCORE_ARGS, SAMPLE_ARGS, REPORT_ARGS (e.g. PREP_ARGS="--target 40").
# Submit from this repo's root, where logs/ is.

ROOT=/scratch/$USER/gemma4
MAIN=${MAIN:-$HOME/Gemma4_kaggle_repo}
WORK=${WORK:?set WORK to the work dir}
STEPS=${STEPS:-score sample}
MODEL=$ROOT/models/gemma-4-31b-it-qat-w4a16-ct
PY=$ROOT/venvs/vllm_host/bin/python            # the serving env: its vLLM and transformers render and score
PRELOAD=$ROOT/envs/py312/lib/libstdc++.so.6    # conda sqlite3 vs system libstdc++, as in the main track's jobs
has() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p "$WORK"

module load explorer
export HF_HOME=$ROOT/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=$ROOT/cache/vllm TRITON_CACHE_DIR=$ROOT/cache/triton
export XDG_CACHE_HOME=$ROOT/cache/xdg FLASHINFER_WORKSPACE_BASE=$ROOT/cache/flashinfer
export TMPDIR=$ROOT/cache/tmp/$SLURM_JOB_ID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK   # else vLLM startup can stall for an hour
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) git=$(git rev-parse --short HEAD 2>/dev/null) steps=$STEPS work=$WORK start=$(date -Is)"
[ -s "$MODEL/model.safetensors" ] || { echo "FAIL: the model is missing: $MODEL (the main track's slurm/fetch_model.slurm)"; exit 1; }
echo "wheelhouse=$(cat "$ROOT/wheelhouse/.version" 2>/dev/null) vllm=$(LD_PRELOAD=$PRELOAD "$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

if has score || has sample; then
  GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
  echo "gpu: $GPU"
  if [[ $GPU == *H200* || $GPU == *H100* ]]; then
    echo "FAIL: vLLM can't serve the int4 model on Explorer's Hopper GPUs; submit on an A100 or an L40S"
    exit 1
  fi
  source "$MAIN/baseline/gpu_guard.sh"   # requeues the job once if CUDA can't use the allocated GPU
  gpu_guard "$PY"
fi

run() {   # run <step> <args...>
  echo "== $1 ($(date +%T))"
  LD_PRELOAD=$PRELOAD "$PY" loops/feedback.py "$@" || { echo "FAIL: $1"; exit 1; }
}
# shellcheck disable=SC2086  # RUNS and the *_ARGS are lists
{
  if has prep; then run prep --work "$WORK" --model "$MODEL" --runs ${RUNS:?set RUNS to the run dirs} $PREP_ARGS; fi
  if has score; then run score --work "$WORK" --model "$MODEL" ${ARMS:+--arms "$ARMS"} $SCORE_ARGS; fi
  if has sample; then run sample --work "$WORK" --model "$MODEL" ${ARMS:+--arms "$ARMS"} $SAMPLE_ARGS; fi
  if has report; then run report --work "$WORK" $REPORT_ARGS; fi
}
echo "end=$(date -Is)"
