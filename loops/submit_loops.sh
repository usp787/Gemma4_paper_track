#!/bin/bash
# Queue the loop study's closed-loop runs (loops/README.md, experiment 2): off_args with the always-block chat template
# (block) and with the model's own (ctrl), on the 70 dev-sound and the 39 rich-sound tasks. They run on A100s in the
# gpu partition, 4 tasks at once on one GPU (the main track's eval_gpu.slurm in resume mode). So they leave alone the 2
# L40S slots of the sharing partition, where the main track's evals run. Nothing is queued without GO=1: by default it
# prints the jobs and what blocks them. Run it on the login node from this repo's checkout. It only uses ls, cat,
# grep, curl, scontrol, sacctmgr, squeue and sbatch there.
#   bash loops/submit_loops.sh                      dry run: the plan, and what blocks it
#   GO=1 bash loops/submit_loops.sh                 queue the chain below
#   GO=1 STEPS=report bash loops/submit_loops.sh    the report alone, e.g. after finishing a run's leftovers
# The chain, each job waiting for the one before:
#   1. template (CPU, short, ~5 min): make_template.py writes $R/paper/loops/always_block.jinja from the model's own
#      template, after its checks on probe conversations. A failed check cancels every job after it.
#   2. ctrl and block (one A100 80 GB each, side by side): the pinned zip of loops/arms.tsv, PINS=1, unpaced, LANES
#      tasks at once; block with CHAT_TEMPLATE set. Each arm is a chain of JOBS jobs: a later one finishes what an
#      earlier one left, or exits before starting vLLM. Expect ~1.7 h per arm for the 109 tasks.
#   3. report (CPU, minutes): baseline_report.py with loops/arms.tsv, for the dev and the rich tasks, then
#      loop_report.py on both arms' runs, into $R/paper/loops/report_p<pass>_<time>/.
# Env:
#   PASS=k     1 (default): the arms' first runs; k > 1: their _r<k> runs, a second seed of both arms
#   STEPS      any of "template ctrl block report" (default: all four)
#   SERIAL=1   block waits for ctrl to end: one A100 at a time, ~3.5 h in all instead of ~1.7 h
#   TIME (02:00:00), JOBS (2 per arm), LANES (4), GPU (a100), PARTITION (gpu), CONSTRAINT (a100@80g; empty for none)
#   AFTER=<job id>  hold the chain until that job has ended
#   MAIN       the main-track checkout (default ~/Gemma4_kaggle_repo); its eval_gpu.slurm must know CHAT_TEMPLATE
#   FORCE=1    queue even though loops jobs of ours are queued or running
#   For tests: GEMMA4_ROOT=<dir> replaces /scratch/$USER/gemma4, and KAGGLE_WH=<n> fakes Kaggle's wheelhouse version.
set -euo pipefail
cd "$(dirname "$0")/.."
LOGS=$(pwd)/logs
GO=${GO:-0}
ME=${USER:-$(id -un)}
R=${GEMMA4_ROOT:-/scratch/$ME/gemma4}
MAIN=${MAIN:-$HOME/Gemma4_kaggle_repo}
PASS=${PASS:-1}
STEPS=${STEPS:-template ctrl block report}
TIME=${TIME:-02:00:00}; JOBS=${JOBS:-2}; LANES=${LANES:-4}
GPU=${GPU:-a100}; PARTITION=${PARTITION:-gpu}; CONSTRAINT=${CONSTRAINT-a100@80g}
CON=(); [ -n "$CONSTRAINT" ] && CON=(--constraint="$CONSTRAINT")
has() { [[ " $STEPS " == *" $1 "* ]]; }
lines() { if [ -f "$1" ]; then grep -c . "$1" || true; else echo 0; fi; }
WH=$(cat "$R/wheelhouse/.version" 2>/dev/null || true)
V=v${WH:-unknown}
OUT=$R/paper/loops
TPL=$OUT/always_block.jinja
IDS_DEV=$R/train/rft0/dev_sound_ids.txt
IDS_RICH=$R/train/rich_sound_ids.txt   # the 39 rich tasks of v30's grader check (the main track's submit_ab.sh)
TASKS=$OUT/tasks_dev70_rich39.txt
kaggle_version() {   # Kaggle's wheelhouse version, as baseline/opsd_baselines.sh reads it; "" if unknown
  if [ -n "${KAGGLE_WH:-}" ]; then echo "$KAGGLE_WH"; return 0; fi
  curl -s -m 15 https://www.kaggle.com/api/v1/datasets/view/metric/gemma-4-developer-agent-wheelhouse 2>/dev/null \
    | grep -o '"currentVersionNumber":[0-9]*' | cut -d: -f2 || true
}
run_of() {   # run_of <arm> <pass>: the arm's run for that pass, from loops/arms.tsv
  local first
  first=$(awk -F'\t' -v a="$1" '!/^#/ && $1 == a {print $4}' loops/arms.tsv)
  first=${first//\{v\}/$V}
  if [ "$2" = 1 ]; then echo "$first"; else echo "${first}_r$2"; fi
}
zip_of() { awk -F'\t' -v a="$1" '!/^#/ && $1 == a {print $3}' loops/arms.tsv; }

blocked=()
[[ $PASS =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: PASS=$PASS is not a pass number"; exit 1; }
for s in $STEPS; do
  case $s in template|ctrl|block|report) ;; *) echo "FAIL: unknown step '$s' (template ctrl block report)"; exit 1 ;; esac
done
RUN_C=$(run_of ctrl "$PASS"); RUN_B=$(run_of block "$PASS")
ZIP_C=$R/submissions/$(zip_of ctrl); ZIP_B=$R/submissions/$(zip_of block)
[ -f "$ZIP_C" ] || blocked+=("no zip $ZIP_C")
[ "$ZIP_B" = "$ZIP_C" ] || blocked+=("loops/arms.tsv gives the arms different zips; they must differ only in the template")
[ -s "$R/models/gemma-4-31b-it-qat-w4a16-ct/model.safetensors" ] || blocked+=("the model is missing: (cd $MAIN && sbatch slurm/fetch_model.slurm)")
[ -x "$R/venvs/vllm_host/bin/python" ] || blocked+=("no venv $R/venvs/vllm_host")
[ "$(lines "$IDS_DEV")" = 70 ] || blocked+=("$IDS_DEV should hold the 70 dev-sound ids")
[ "$(lines "$IDS_RICH")" = 39 ] || blocked+=("$IDS_RICH should hold the 39 rich-sound ids")
grep -q CHAT_TEMPLATE "$MAIN/baseline/eval_gpu.slurm" 2>/dev/null \
  || blocked+=("$MAIN/baseline/eval_gpu.slurm doesn't know CHAT_TEMPLATE yet: git pull in $MAIN")
[ "$V" != vunknown ] || blocked+=("no wheelhouse version in $R/wheelhouse/.version")
kv=$(kaggle_version)
if [ -n "$kv" ] && [ "v$kv" != "$V" ]; then
  blocked+=("Kaggle's wheelhouse is v$kv, ours $V: re-align first ((cd $MAIN && bash baseline/submit_realign.sh))")
fi
if has block && ! has template && [ ! -s "$TPL" ]; then blocked+=("no $TPL yet: add template to STEPS"); fi
running=$(squeue -h -u "$ME" -o "%i %j %T" 2>/dev/null | awk '$2 ~ /^loops_/' || true)
if [ -n "$running" ] && [ "${FORCE:-0}" != 1 ]; then
  blocked+=("loops jobs of ours are queued or running ($(echo "$running" | awk '{print $1"("$2")"}' | paste -sd' ')); FORCE=1 queues anyway")
fi
n_gpu=0
has ctrl && n_gpu=$((n_gpu + JOBS))
has block && n_gpu=$((n_gpu + JOBS))
QOS=$(scontrol show partition "$PARTITION" 2>/dev/null | grep -oE 'QoS=[^ ]+' | cut -d= -f2 || true)
LIMIT=
if [ -n "$QOS" ] && [ "$QOS" != N/A ]; then LIMIT=$(sacctmgr -nP show qos "$QOS" format=maxsubmitpu 2>/dev/null | head -1 || true); fi
HAVE=$(squeue -h -u "$ME" -p "$PARTITION" 2>/dev/null | wc -l)
if [ -n "$LIMIT" ] && [ $((HAVE + n_gpu)) -gt "$LIMIT" ]; then
  blocked+=("$PARTITION lets you have $LIMIT jobs queued; $HAVE are, and this needs $n_gpu")
fi

echo "loop study, pass $PASS on $V: ctrl -> $RUN_C, block -> $RUN_B"
echo "  zip $(basename "$ZIP_C"); $(lines "$IDS_DEV") dev + $(lines "$IDS_RICH") rich tasks; template $TPL; steps: $STEPS"
for r in "$RUN_C" "$RUN_B"; do
  if [ -d "$R/results/$r" ]; then echo "  note: $R/results/$r exists, so its arm resumes it (finished tasks are skipped)"; fi
done
if [ ${#blocked[@]} -gt 0 ] && [ "$GO" = 1 ]; then
  printf 'blocked: %s\n' "${blocked[@]}"
  echo "FAIL: nothing queued"
  exit 1
fi
if [ "$GO" = 1 ]; then
  mkdir -p "$OUT" "$LOGS"
  cat "$IDS_DEV" "$IDS_RICH" > "$TASKS"
fi

J=
submit() {   # submit <label> <dir> <env assignments...> -- <sbatch args...>: prints it, and with GO=1 queues it from <dir>
  local label=$1 dir=$2 envs=()
  shift 2
  while [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  echo "  $label: (cd $dir && env ${envs[*]} sbatch $*)"
  if [ "$GO" = 1 ]; then
    J=$(cd "$dir" && env "${envs[@]}" sbatch --parsable "$@")
    J=${J%%;*}
    echo "    -> job $J; log $LOGS/loops_${label%% *}_$J.out"
  else
    J="<${label// /_}>"   # a placeholder without spaces, so a printed --dependency stays one flag
  fi
}
dep() {   # dep <mode:job> ...: a --dependency for the given ones that are set, or nothing
  local d=() x
  for x in "$@"; do [ -n "${x#*:}" ] && d+=("$x"); done
  [ ${#d[@]} -gt 0 ] && printf -- '--dependency=%s --kill-on-invalid-dep=yes' "$(IFS=,; echo "${d[*]}")"
  return 0
}
LOG_ARGS=(-o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err")
TJ=
if has template; then
  # shellcheck disable=SC2046  # dep prints flags, or nothing
  submit template "$(pwd)" STEP=template OUT="$TPL" -- -J loops_template -p short -c 2 --mem=8G -t 00:20:00 \
    "${LOG_ARGS[@]}" $(dep "afterany:${AFTER:-}") loops/loops.slurm
  TJ=$J
fi
arm_chain() {   # arm_chain <arm> <run> <chat template or ""> <extra dependency>: JOBS chained A100 jobs; sets LAST
  local name=$1 run=$2 tpl=$3 extra=$4 prev= k first
  for k in $(seq 1 "$JOBS"); do
    if [ "$k" = 1 ]; then
      first=$(dep "afterok:$TJ" "$extra" "afterany:$([ -z "$TJ" ] && echo "${AFTER:-}" || true)")
    else
      first=$(dep "afterany:$prev")
    fi
    # shellcheck disable=SC2086  # $first is a list of flags, or nothing
    submit "$name $k/$JOBS" "$MAIN" LANES="$LANES" PINS=1 PACE=0 CHAT_TEMPLATE="$tpl" -- -J "loops_$name" \
      -p "$PARTITION" -t "$TIME" --gres=gpu:"$GPU":1 "${CON[@]}" --cpus-per-task=$((4 * LANES)) --mem=128G \
      "${LOG_ARGS[@]}" $first baseline/eval_gpu.slurm "$ZIP_C" "$TASKS" resume "$run"
    prev=$J
    ALL+=("$J")
  done
  LAST=$prev
}
ALL=(); LAST=
CJ=
if has ctrl; then arm_chain ctrl "$RUN_C" "" ""; CJ=$LAST; fi
if has block; then
  if [ "${SERIAL:-0}" = 1 ] && [ -n "$CJ" ]; then arm_chain block "$RUN_B" "$TPL" "afterany:$CJ"
  else arm_chain block "$RUN_B" "$TPL" ""; fi
fi
if has report; then
  runs=()
  for k in $(seq 1 "$PASS"); do runs+=("$R/results/$(run_of ctrl "$k")"); done
  for k in $(seq 1 "$PASS"); do runs+=("$R/results/$(run_of block "$k")"); done
  deps=("afterany:${AFTER:-}")
  [ ${#ALL[@]} -gt 0 ] && deps=("afterany:$(IFS=:; echo "${ALL[*]}")")
  deps+=("afterok:$TJ")   # a failed template check cancels the report too
  # shellcheck disable=SC2046
  submit report "$(pwd)" STEP=report DIR="$OUT/report_p${PASS}_$(date +%m%d-%H%M)" IDS_DEV="$IDS_DEV" \
    IDS_RICH="$IDS_RICH" RUNS="${runs[*]}" -- -J loops_report -p short -c 2 --mem=8G -t 00:30:00 "${LOG_ARGS[@]}" \
    $(dep "${deps[@]}") loops/loops.slurm
fi
if [ "$GO" = 1 ]; then
  echo "watch: squeue -u \$USER; the runs: $R/results/$RUN_C and $R/results/$RUN_B; the report: $OUT/report_p${PASS}_*/"
else
  [ ${#blocked[@]} -eq 0 ] || printf 'blocked: %s\n' "${blocked[@]}"
  echo "dry run: nothing queued. GO=1 queues these jobs."
fi
exit 0
