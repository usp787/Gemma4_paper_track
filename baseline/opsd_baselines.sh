#!/bin/bash
# The OPSD paper's baseline runs (baseline/README.md), through the main-track repo's own eval tooling. Every run is
# scored the way the main track scores: the 70 dev-sound tasks, PINS=1, unpaced, one task at a time on an L40S.
#   bash baseline/opsd_baselines.sh status        prerequisites, then each arm's runs and what comes next
#   bash baseline/opsd_baselines.sh run <arm>     queue the arm's next run (arms.tsv)
#   bash baseline/opsd_baselines.sh split         write the train/dev split files (make_split.py, a 1-min job)
#   bash baseline/opsd_baselines.sh check-train   gold and empty patches on the train split (train_check.slurm)
#   bash baseline/opsd_baselines.sh report [N]    the cross-arm tables on wheelhouse vN, ours by default (a short job)
# Nothing is queued without GO=1. By default each command prints what it would run, and what blocks it.
# - An arm's first run goes through the main track's chain, post_train/v28/submit_baseline.sh: check the zip, smoke,
#   gate, the run itself, and a report against the reference arm. That script refuses while any eval job of ours is
#   queued, so start a first run on an empty queue.
# - A later run needs the zip's passed smoke gate. It goes through baseline/submit_eval.sh, then a diagnose_run.py
#   job against the arm's first run. AFTER=<job id> holds it until that job ends, e.g. the previous run's driver.
# Run it on the Explorer login node, from this repo's checkout. It runs only light commands there (ls, grep, stat,
# squeue, sbatch); Python runs inside sbatch jobs. MAIN=<dir> sets the main-track checkout
# (default ~/Gemma4_kaggle_repo). For tests only: GEMMA4_ROOT=<dir> replaces /scratch/$USER/gemma4, and
# MODEL_BYTES the model's expected size.
# Logs: every job this script queues writes <job name>_<id>.out to this repo's logs/, which git ignores. The main
# track's chain writes its jobs' logs to $MAIN/logs; this script links that folder in as logs/main_track.
# Versions: runs, smokes and gates are named after our wheelhouse version ($R/wheelhouse/.version, which the main
# track's build_vllm_env.slurm writes), and it must equal Kaggle's. arms.tsv's {v} becomes that tag, v29 say, a gate
# passed on another version never counts, and the report counts one version's runs. KAGGLE_WH=<n> fakes Kaggle's
# version, for tests.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOGS=$(dirname "$HERE")/logs
MAIN=${MAIN:-$HOME/Gemma4_kaggle_repo}
R=${GEMMA4_ROOT:-/scratch/$USER/gemma4}
PAPER=$R/paper
IDS=$R/train/rft0/dev_sound_ids.txt
PY=$R/venvs/harness/bin/python
MODEL=$R/models/gemma-4-31b-it-qat-w4a16-ct/model.safetensors
MODEL_BYTES=${MODEL_BYTES:-23265352448}   # slurm/fetch_model.slurm WANT_BYTES
GO=${GO:-0}
WH=$(cat "$R/wheelhouse/.version" 2>/dev/null || true)
V=v${WH:-unknown}                           # the tag in run, smoke and gate names

fail() { echo "FAIL: $*" >&2; exit 1; }
lines() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
arms() { grep -v '^#' "$HERE/arms.tsv" | awk -F'\t' 'NF >= 8 {print $1}'; }
arm_row() {   # arm_row <arm>: sets AGENT ZIP FIRST EXTRA REF KAGGLE ABOUT from arms.tsv, for wheelhouse $V
  local line
  line=$(grep -v '^#' "$HERE/arms.tsv" | awk -F'\t' -v a="$1" '$1 == a' | head -1)
  [ -n "$line" ] || fail "no arm '$1' in arms.tsv; the arms are: $(arms | paste -sd' ')"
  IFS=$'\t' read -r _ AGENT ZIP FIRST EXTRA REF KAGGLE ABOUT <<<"$line"
  FIRST=${FIRST//"{v}"/$V}; EXTRA=${EXTRA//"{v}"/$V}
  KAGGLE=$(tr ',' '\n' <<<"$KAGGLE" | sed -n "s/^$V://p" | head -1)   # this version's Kaggle score, if any
  return 0
}
kaggle_version() {   # Kaggle's current wheelhouse version, read as baseline/wheelhouse_check.sh reads it; "" if unknown
  if [ -n "${KAGGLE_WH:-}" ]; then echo "$KAGGLE_WH"; return 0; fi
  curl -s -m 15 https://www.kaggle.com/api/v1/datasets/view/metric/gemma-4-developer-agent-wheelhouse 2>/dev/null \
    | grep -o '"currentVersionNumber":[0-9]*' | cut -d: -f2 || true
}
runs_of() {   # runs_of <first_run> <extra_runs>: the arm's run dirs that exist, as baseline_report.py orders them
  local r
  [ -d "$R/results/$1" ] && echo "$1"
  if [ "$2" != - ]; then for r in ${2//,/ }; do [ -d "$R/results/$r" ] && echo "$r"; done; fi
  for r in "$R/results/$1"_r*; do [ -d "$r" ] && basename "$r"; done | grep -E "^$1_r[0-9]+\$" \
    | sed -E 's/^(.*_r)([0-9]+)$/\2 \1\2/' | sort -n | cut -d' ' -f2
  return 0
}
resolved_of() {   # "k/n" from a run's summary_all.json, written by eval_driver.slurm once the run has ended
  local f=$R/results/$1/summary_all.json s=
  if [ -f "$f" ]; then
    s=$(tr -d ' \n' < "$f" | grep -oE '"all":\[[0-9]+,[0-9]+\]' | head -1 | grep -oE '[0-9]+,[0-9]+' | tr , /) || true
    echo "${s:-?}"
  else
    echo "no summary"
  fi
}
gate_of() {   # a zip's passed smoke gate on this wheelhouse, if any: submit_baseline.sh names the smoke
  local sha    # smoke_zip_<sha>_<v>_<time>, and smoke_check.py writes GATE_OK into it. v28's smokes have no <v>.
  sha=$(sha256sum "$1" | cut -c1-12)
  ls -d "$R"/results/smoke_zip_"${sha}"_"$V"_*/GATE_OK 2>/dev/null | tail -1 || true
}
model_ok() { [ "$(stat -c %s "$MODEL" 2>/dev/null || echo 0)" = "$MODEL_BYTES" ]; }
evals_queued() {   # our jobs that make submit_baseline.sh refuse a new chain
  squeue -h -u "$USER" -o "%i %j %T" 2>/dev/null \
    | awk '$2 ~ /^(eval_l40s|eval_driver|smoke_gate|package_submission|diagnose|build_vllm_env|build_harness_env|lora_speed)$/' || true
}
ensure_logs() {   # this repo's logs/, with the main track's logs/ linked in as logs/main_track
  mkdir -p "$LOGS"
  if [ -d "$MAIN/logs" ] && [ ! -e "$LOGS/main_track" ] && [ ! -L "$LOGS/main_track" ]; then
    ln -s "$MAIN/logs" "$LOGS/main_track" || true
  fi
}
short_job() {   # short_job <job name> <sbatch args...>: with GO=1, queue a short CPU job, wait until it ends, print its log
  local name=$1 j rc=0
  shift
  echo "  sbatch --wait -J $name -o $LOGS/%x_%j.out $*"
  [ "$GO" = 1 ] || { echo "dry run: nothing queued. GO=1 queues it, waits for it to end and prints its log."; return 0; }
  ensure_logs
  j=$(sbatch --parsable --wait -J "$name" -o "$LOGS/%x_%j.out" "$@") || rc=$?
  j=${j%%;*}
  [ -n "$j" ] || fail "$name was not queued"
  echo "job $j ended with exit code $rc. Its log, $LOGS/${name}_$j.out:"
  cat "$LOGS/${name}_$j.out" 2>/dev/null || echo "(not there yet; look again in a minute)"
  return "$rc"
}

cmd_status() {
  local a runs n r line gate next kv
  if model_ok; then echo "model:   complete"; else
    echo "model:   INCOMPLETE ($(stat -c %s "$MODEL" 2>/dev/null || echo 0) of $MODEL_BYTES bytes): every GPU job fails until"
    echo "         (cd $MAIN && sbatch slurm/fetch_model.slurm), then: ls -la $(dirname "$MODEL")"
  fi
  kv=$(kaggle_version)
  echo "harness: wheelhouse $V, Kaggle's v${kv:-?}; the runs below are those on $V"
  if [ -n "$kv" ] && [ "v$kv" != "$V" ]; then
    echo "         WARN: Kaggle is on v$kv. Re-align before measuring: (cd $MAIN && bash baseline/submit_realign.sh)"
  fi
  echo "dev ids: $(lines "$IDS") in $IDS (should be 70)"
  if [ -s "$PAPER/split/train59.txt" ]; then echo "split:   $(lines "$PAPER/split/train59.txt") train ids in $PAPER/split"
  else echo "split:   not written yet (opsd_baselines.sh split)"; fi
  if [ -d "$MAIN/.git" ]; then
    echo "main:    $MAIN at $(git -C "$MAIN" rev-parse --short HEAD)$([ -n "$(git -C "$MAIN" status --porcelain --untracked-files=no)" ] && echo ', with local changes' || true)"
  else echo "main:    no checkout at $MAIN (set MAIN=...)"; fi
  echo "logs:    $LOGS; the main track's chain logs to $MAIN/logs, linked in as logs/main_track"
  echo "queue:   $(squeue -h -u "$USER" 2>/dev/null | wc -l) of our jobs queued or running"
  squeue -h -u "$USER" -o "         %i %j %T %M" 2>/dev/null || true
  echo
  for a in $(arms); do
    arm_row "$a"
    mapfile -t runs < <(runs_of "$FIRST" "$EXTRA")
    n=${#runs[@]}
    line=""
    for r in "${runs[@]}"; do line+="${line:+, }$r $(resolved_of "$r")"; done
    gate=none; [ -f "$R/submissions/$ZIP" ] && [ -n "$(gate_of "$R/submissions/$ZIP")" ] && gate=passed
    [ -f "$R/submissions/$ZIP" ] || gate="no zip"
    if [ "$n" -gt 0 ] && [ ! -f "$R/results/${runs[-1]}/summary_all.json" ]; then next="wait: ${runs[-1]} has no summary yet"
    elif [ ! -d "$R/results/$FIRST" ]; then next="first run: chain (check, smoke, gate, run, report) as $FIRST"
    else next="run $((n + 1)): ${FIRST}_r$((n + 1))"; fi
    echo "$a: ${line:-no runs}"
    echo "    gate $gate; next $next"
  done
}

cmd_run() {
  local a=${1:?usage: opsd_baselines.sh run <arm>} zip runs n first kv ref_run next gate out d j blocked=()
  arm_row "$a"
  zip=$R/submissions/$ZIP
  [ -f "$zip" ] || blocked+=("no zip $zip")
  model_ok || blocked+=("the model is incomplete: (cd $MAIN && sbatch slurm/fetch_model.slurm) first")
  [ "$(lines "$IDS")" = 70 ] || blocked+=("$IDS should hold the 70 dev-sound ids")
  [ -d "$MAIN/baseline" ] || blocked+=("no main-track checkout at $MAIN (set MAIN=...)")
  [ "$V" != vunknown ] || blocked+=("no wheelhouse version in $R/wheelhouse/.version (slurm/build_vllm_env.slurm writes it)")
  kv=$(kaggle_version)
  if [ -n "$kv" ] && [ "v$kv" != "$V" ]; then
    blocked+=("Kaggle's wheelhouse is v$kv, ours $V: re-align first, (cd $MAIN && bash baseline/submit_realign.sh), and settle what the update invalidates")
  fi
  mapfile -t runs < <(runs_of "$FIRST" "$EXTRA")
  n=${#runs[@]}
  if [ -d "$R/results/$FIRST" ]; then first=0; else first=1; fi   # an extra run (on's base_<v>_l40s) can come first
  if [ "$n" -gt 0 ] && [ ! -f "$R/results/${runs[-1]}/summary_all.json" ] && [ "${FORCE:-0}" != 1 ]; then
    blocked+=("${runs[-1]} has no summary yet: it is still running, or its leftovers need a rerun (FORCE=1 queues the next run anyway)")
  fi

  if [ "$first" = 1 ]; then
    # submit_baseline.sh compares the new run with REF: the reference arm's first run on this wheelhouse, or else
    # baseline/submit_realign.sh's base score on it. It always compares the host's default zip (on's) with the latter.
    ref_run=base_${V}_l40s
    if [ "$REF" != - ] && [ "$ZIP" != baseline_noadapter_20260925-2342_fd03bc4.zip ]; then
      arm_row "$REF"; [ -d "$R/results/$FIRST" ] && ref_run=$FIRST
      arm_row "$a"
    fi
    [ -d "$R/results/$ref_run" ] || blocked+=("no run $ref_run to compare with yet (baseline/submit_realign.sh makes base_${V}_l40s)")
    [ -n "$(evals_queued)" ] && blocked+=("eval jobs of ours are queued, and submit_baseline.sh refuses a new chain until they end: $(evals_queued | awk '{print $1"("$2")"}' | paste -sd' ')")
    echo "arm $a: $ABOUT"
    echo "first run: $FIRST, through the main track's chain, compared with $ref_run"
    echo "  (cd $MAIN && ZIP=$zip REF=$ref_run${KAGGLE:+ KAGGLE=$KAGGLE} PACED=0 bash post_train/v28/submit_baseline.sh)"
  else
    next=${FIRST}_r$((n + 1))
    [ -d "$R/results/$next" ] && blocked+=("$R/results/$next exists already")
    [ -f "$zip" ] && gate=$(gate_of "$zip") || gate=
    [ -n "$gate" ] || blocked+=("no passed smoke gate for $zip; the arm's first run through submit_baseline.sh makes one")
    echo "arm $a: $ABOUT"
    echo "run $((n + 1)) of the arm: $next${AFTER:+, after job $AFTER}; its report compares it with ${runs[0]}"
    echo "  (cd $MAIN && PINS=1 PACE=0${AFTER:+ DEPEND=afterany:$AFTER} bash baseline/submit_eval.sh $zip $IDS $next)"
    echo "  then, after its driver: diagnose_run.py $R/results/$next --vs $R/results/${runs[0]}${KAGGLE:+ --kaggle $KAGGLE}"
  fi
  if [ ${#blocked[@]} -gt 0 ]; then
    printf 'blocked: %s\n' "${blocked[@]}"
    [ "$GO" = 1 ] && fail "not queued"
    return 0
  fi
  [ "$GO" = 1 ] || { echo "dry run: nothing queued. GO=1 queues it."; return 0; }

  ensure_logs
  cd "$MAIN"
  if [ "$first" = 1 ]; then
    # PACED=0: only the unpaced run counts here; the default zip's chain would also queue a paced one
    ZIP=$zip REF=$ref_run KAGGLE=$KAGGLE PACED=0 bash post_train/v28/submit_baseline.sh
    echo "(that logs/ is the main track's, $MAIN/logs: from this repo, $LOGS/main_track)"
    return 0
  fi
  if ! out=$(PINS=1 PACE=0 DEPEND=${AFTER:+afterany:$AFTER} bash baseline/submit_eval.sh "$zip" "$IDS" "$next" 2>&1); then
    echo "$out"; fail "the driver of $next was not queued"
  fi
  echo "$out"
  d=$(grep -oE 'driver job [0-9]+' <<<"$out" | head -1 | grep -oE '[0-9]+$') || true
  [ -n "$d" ] || fail "no driver job id in submit_eval.sh's output; queue the report by hand once $next has ended"
  j=$(sbatch --parsable -p short -t 00:30:00 --mem=4G -J diagnose -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" \
      --dependency="afterany:$d" \
      --wrap "$PY baseline/diagnose_run.py $R/results/$next --vs $R/results/${runs[0]}${KAGGLE:+ --kaggle $KAGGLE}")
  echo "report: job ${j%%;*}, once driver $d ends -> $LOGS/diagnose_${j%%;*}.out and $R/results/$next/diagnosis.md"
  echo "the run's driver and shards log to the main track's logs/: $LOGS/main_track/eval_driver_$d.out, eval_l40s_<id>.out"
}

cmd_split() {
  short_job opsd_split -p short -t 00:05:00 --mem=1G --wrap "$PY $HERE/make_split.py --out $PAPER/split"
}

cmd_check_train() {
  local j cmd=(sbatch --parsable --export=ALL,MAIN="$MAIN" -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" "$HERE/train_check.slurm")
  echo "  ${cmd[*]}"
  [ -s "$PAPER/split/train59.txt" ] || { echo "blocked: no $PAPER/split/train59.txt yet (opsd_baselines.sh split)"; [ "$GO" = 1 ] && fail "not queued"; return 0; }
  [ "$GO" = 1 ] || { echo "dry run: nothing queued. GO=1 queues it (CPU; it took 6 min on 2026-10-08)."; return 0; }
  ensure_logs
  j=$("${cmd[@]}")
  echo "queued job ${j%%;*}; its log: $LOGS/opsd_train_check_${j%%;*}.out (and .err)"
}

cmd_report() {   # cmd_report [N]: the tables of the runs on wheelhouse vN (default: ours), e.g. 28 for the v28 record
  local wh=${1:-$WH} out
  [ -n "$wh" ] || fail "no wheelhouse version in $R/wheelhouse/.version; name one: opsd_baselines.sh report <N>"
  out=$PAPER/report_v${wh}_$(date +%m%d-%H%M)
  if [ "$GO" = 1 ]; then mkdir -p "$PAPER"; fi
  short_job opsd_report -p short -t 00:30:00 --mem=8G \
    --wrap "$PY $HERE/baseline_report.py --results $R/results --ids $IDS --wheelhouse $wh --json $out.json > $out.md" \
    || fail "the report job failed; its log is above"
  [ "$GO" = 1 ] || return 0
  echo "report: $out.md and $out.json"
  cat "$out.md"
}

case ${1:-status} in
  status) cmd_status ;;
  run) shift; cmd_run "$@" ;;
  split) cmd_split ;;
  check-train) cmd_check_train ;;
  report) cmd_report "${2:-}" ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
