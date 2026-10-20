#!/bin/bash
# Queue the loop study's closed-loop runs (loops/README.md: experiment 2, and round 2's G3 and G4): off_args with the
# model's own chat template (ctrl), with the always-block template (block), at temperature 1.0 (ctrl_t1), and with the
# main track's token proxy collapsing runs of identical tool turns in what the model sees (dedup, G4), on the 70
# dev-sound and the 39 rich-sound tasks. They run on A100s in the gpu partition, 4 tasks at once on one GPU (the main
# track's eval_gpu.slurm in resume mode). So they leave alone the 2 L40S slots of the sharing partition, where the main
# track's evals run. Nothing is queued without GO=1: by default it prints the jobs and what blocks them. Run it on the
# login node from this repo's checkout. It only uses ls, cat, grep, curl, scontrol, sacctmgr, squeue and sbatch there.
#   bash loops/submit_loops.sh                      dry run: the plan, and what blocks it
#   GO=1 bash loops/submit_loops.sh                 queue pass 1 of G3's three arms: the chain below
#   GO=1 PASS=2 bash loops/submit_loops.sh          pass 2, once pass 1's GPU jobs have left the queue
#   GO=1 STEPS=report bash loops/submit_loops.sh    the report alone, e.g. after finishing a run's leftovers
#   GO=1 STEPS=smoke bash loops/submit_loops.sh                    G4's smoke test first (below)
#   GO=1 STEPS="dedup report" bash loops/submit_loops.sh           G4: pass 1 of dedup, reported against G3's arms
#   GO=1 PASS=2 STEPS="dedup report" bash loops/submit_loops.sh    G4's pass 2
#   GO=1 STEPS="modelzip ctrl_12b report" bash loops/submit_loops.sh          the 12B control: its zip, then pass 1
#   GO=1 PASS=2 STEPS="modelzip ctrl_12b report" bash loops/submit_loops.sh   its pass 2 (modelzip only checks the zip)
# G4's smoke test (STEPS=smoke): one A100 job, the dedup arm on the 4 tasks of loops/g4_smoke_ids.txt (each loops on one
# well-formed call with the same result in every T = 0.2 control run), into loops_dedup_smoke_<v>_a100, with no report.
# It shows, for ~0.5 A100-hour, that the proxy collapses the harness's real requests; the command it prints checks that.
# The chain:
#   1. checks (CPU, short, minutes), side by side:
#      - template: make_template.py writes $R/paper/loops/always_block.jinja from the model's own template, after its
#        checks on probe conversations;
#      - t1zip: make_t1_zip.py derives the T = 1 zip from the pinned one (only the temperature line of
#        configs/sampling.yaml differs), or checks it if it exists; then the main track's
#        slurm/package_submission.slurm validates it with the host's compiler (ZIPCHECK=0 skips that);
#      - modelzip: make_model_zip.py derives the 12B control's zip from the pinned one (only its two `model:` lines
#        differ), or checks it if it exists; then package_submission.slurm validates it, as for t1zip.
#      A failed check cancels every job after it, in every arm.
#   2. the arms (one A100 80 GB each): the zip of loops/arms.tsv, PINS=1, unpaced, LANES tasks at once; block with
#      CHAT_TEMPLATE set, dedup with DEDUP=DEDUP_RUN (every other arm DEDUP=0); ctrl_12b with the 12B zip, which makes
#      eval_gpu.slurm serve the 12B. Each arm is a chain of JOBS jobs: a later one finishes what an earlier one left,
#      or exits before starting vLLM. WIDTH arms run side by side (default: all of them). Expect ~1.5 h per arm.
#   3. report (CPU, minutes): baseline_report.py with loops/arms.tsv, for the dev and the rich tasks, then
#      loop_report.py on every pass of every arm (passes 1 to PASS, and any later pass that already exists), pooled per
#      arm and compared in pairs (block and ctrl_t1 against ctrl; dedup, when it is queued or has run, against ctrl and
#      block; ctrl_12b, likewise, against ctrl), with round 1's runs on wheelhouse OLD as a replicate, into
#      $R/paper/loops/report_[g4_|12b_]p<pass>_<time>/ (g4_ when dedup is queued, 12b_ when ctrl_12b is).
# The gpu partition lets one user have 8 jobs queued and 4 running (QoS gpu, 2026-10-15). One pass of three arms is
# 3 x JOBS = 6 GPU jobs, so queue the passes one at a time.
# Env:
#   PASS=k     1 (default): the arms' first runs; k > 1: their _r<k> runs, another pass of every arm
#   STEPS      any of "template t1zip modelzip ctrl block ctrl_t1 dedup ctrl_12b smoke report" (default: G3's six)
#   DEDUP_RUN  the dedup arm's DEDUP: the run length the proxy starts collapsing at (default 2: no copy is ever seen)
#   WIDTH=n    arms side by side (default: all queued); SERIAL=1 means WIDTH=1: one A100 at a time, ~4.5 h a pass
#   TIME (02:00:00), JOBS (2 per arm), LANES (4), GPU (a100), PARTITION (gpu), CONSTRAINT (a100@80g; empty for none)
#   SMOKE_TIME (00:45:00): the smoke job's limit
#   ZIPCHECK=0 skip the host compiler's check of the derived zips; OLD (v30): round 1's wheelhouse tag, empty for none
#   AFTER=<job id>  hold the chain until that job has ended
#   MAIN       the main-track checkout (default ~/Gemma4_kaggle_repo); its eval_gpu.slurm must know CHAT_TEMPLATE, for
#              dedup DEDUP, with a token_proxy.py that has --dedup, and for ctrl_12b how to serve the 12B
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
STEPS=${STEPS:-template t1zip ctrl block ctrl_t1 report}
TIME=${TIME:-02:00:00}; JOBS=${JOBS:-2}; LANES=${LANES:-4}
GPU=${GPU:-a100}; PARTITION=${PARTITION:-gpu}; CONSTRAINT=${CONSTRAINT-a100@80g}
OLD=${OLD-v30}; ZIPCHECK=${ZIPCHECK:-1}; DEDUP_RUN=${DEDUP_RUN:-2}
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
SMOKE_IDS=$(pwd)/loops/g4_smoke_ids.txt
SMOKE_RUN=loops_dedup_smoke_${V}_a100
M31=gemma-4-31b-it-qat-w4a16-ct; M12=gemma-4-12b-it-qat-w4a16-ct   # the pinned zip's model, and the 12B control's
ALL_ARMS=(ctrl block ctrl_t1 dedup ctrl_12b)
LATE_ARMS=" dedup ctrl_12b "   # reported only once queued or run, so the earlier reports stay as they were
tag=; has dedup && tag=g4_; has ctrl_12b && tag=12b_   # the report dir's prefix
kaggle_version() {   # Kaggle's wheelhouse version, as baseline/opsd_baselines.sh reads it; "" if unknown
  if [ -n "${KAGGLE_WH:-}" ]; then echo "$KAGGLE_WH"; return 0; fi
  curl -s -m 15 https://www.kaggle.com/api/v1/datasets/view/metric/gemma-4-developer-agent-wheelhouse 2>/dev/null \
    | grep -o '"currentVersionNumber":[0-9]*' | cut -d: -f2 || true
}
field() { awk -F'\t' -v a="$1" -v f="$2" '!/^#/ && $1 == a {print $f}' loops/arms.tsv; }
run_of() {   # run_of <arm> <pass> [<version tag>]: the arm's run for that pass, from loops/arms.tsv
  local first
  first=$(field "$1" 4)
  first=${first//\{v\}/${3:-$V}}
  if [ "$2" = 1 ]; then echo "$first"; else echo "${first}_r$2"; fi
}

blocked=()
[[ $PASS =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: PASS=$PASS is not a pass number"; exit 1; }
for s in $STEPS; do
  case $s in template|t1zip|modelzip|ctrl|block|ctrl_t1|dedup|ctrl_12b|smoke|report) ;;
    *) echo "FAIL: unknown step '$s' (template t1zip modelzip ctrl block ctrl_t1 dedup ctrl_12b smoke report)"; exit 1 ;; esac
done
[[ $DEDUP_RUN =~ ^[1-9][0-9]*$ ]] && [ "$DEDUP_RUN" -ge 2 ] || { echo "FAIL: DEDUP_RUN=$DEDUP_RUN is not a run length of 2 or more"; exit 1; }
ARMS=(); for a in "${ALL_ARMS[@]}"; do has "$a" && ARMS+=("$a"); done
[ "${SERIAL:-0}" = 1 ] && WIDTH=1
n_arms=${#ARMS[@]}; [ "$n_arms" -gt 0 ] || n_arms=1   # STEPS=report queues no arm
WIDTH=${WIDTH:-$n_arms}
[[ $WIDTH =~ ^[1-9][0-9]*$ ]] || { echo "FAIL: WIDTH=$WIDTH is not a number of arms"; exit 1; }
for a in "${ALL_ARMS[@]}"; do [ -n "$(field "$a" 3)" ] || { echo "FAIL: loops/arms.tsv has no arm $a"; exit 1; }; done
ZIP_C=$R/submissions/$(field ctrl 3); ZIP_B=$R/submissions/$(field block 3); ZIP_T=$R/submissions/$(field ctrl_t1 3)
ZIP_D=$R/submissions/$(field dedup 3); ZIP_M=$R/submissions/$(field ctrl_12b 3)
[ -f "$ZIP_C" ] || blocked+=("no zip $ZIP_C")
[ "$ZIP_B" = "$ZIP_C" ] || blocked+=("loops/arms.tsv gives block another zip than ctrl; the two must differ only in the template")
[ "$ZIP_D" = "$ZIP_C" ] || blocked+=("loops/arms.tsv gives dedup another zip than ctrl; the two must differ only in the proxy's edit")
[ "$ZIP_T" = "${ZIP_C%.zip}_t1.zip" ] || blocked+=("loops/arms.tsv's ctrl_t1 zip should be ctrl's with _t1, the one make_t1_zip.py derives")
[ "$ZIP_M" = "${ZIP_C%.zip}_12b.zip" ] || blocked+=("loops/arms.tsv's ctrl_12b zip should be ctrl's with _12b, the one make_model_zip.py derives")
if has ctrl_t1 && ! has t1zip && [ ! -s "$ZIP_T" ]; then blocked+=("no $ZIP_T yet: add t1zip to STEPS"); fi
if has ctrl_12b && ! has modelzip && [ ! -s "$ZIP_M" ]; then blocked+=("no $ZIP_M yet: add modelzip to STEPS"); fi
if { has t1zip || has modelzip; } && [ "$ZIPCHECK" = 1 ] && [ ! -f "$MAIN/slurm/package_submission.slurm" ]; then
  blocked+=("no $MAIN/slurm/package_submission.slurm to validate the derived zips (or ZIPCHECK=0)")
fi
if has template || has ctrl || has block || has ctrl_t1 || has dedup || has smoke; then   # every step on the 31B
  [ -s "$R/models/$M31/model.safetensors" ] || blocked+=("the model is missing: (cd $MAIN && sbatch slurm/fetch_model.slurm)")
fi
if has ctrl_12b; then
  [ -s "$R/models/$M12/model.safetensors" ] || blocked+=("the 12B is missing: (cd $MAIN && sbatch slurm/fetch_model.slurm $M12)")
  grep -q "$M12" "$MAIN/baseline/eval_gpu.slurm" 2>/dev/null \
    || blocked+=("$MAIN/baseline/eval_gpu.slurm doesn't serve the 12B yet: git pull in $MAIN")
fi
[ -x "$R/venvs/vllm_host/bin/python" ] || blocked+=("no venv $R/venvs/vllm_host")
[ "$(lines "$IDS_DEV")" = 70 ] || blocked+=("$IDS_DEV should hold the 70 dev-sound ids")
[ "$(lines "$IDS_RICH")" = 39 ] || blocked+=("$IDS_RICH should hold the 39 rich-sound ids")
grep -q CHAT_TEMPLATE "$MAIN/baseline/eval_gpu.slurm" 2>/dev/null \
  || blocked+=("$MAIN/baseline/eval_gpu.slurm doesn't know CHAT_TEMPLATE yet: git pull in $MAIN")
if has dedup || has smoke; then
  grep -q 'DEDUP_ARGS' "$MAIN/baseline/eval_gpu.slurm" 2>/dev/null && grep -q -- '--dedup' "$MAIN/baseline/token_proxy.py" 2>/dev/null \
    || blocked+=("$MAIN's eval_gpu.slurm and token_proxy.py don't know DEDUP yet: git pull in $MAIN")
fi
if has smoke && [ "$(lines "$SMOKE_IDS")" != 4 ]; then blocked+=("$SMOKE_IDS should hold the smoke test's 4 task ids"); fi
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
n_gpu=$(( ${#ARMS[@]} * JOBS ))
has smoke && n_gpu=$((n_gpu + 1))
QOS=$(scontrol show partition "$PARTITION" 2>/dev/null | grep -oE 'QoS=[^ ]+' | cut -d= -f2 || true)
LIMIT=
if [ -n "$QOS" ] && [ "$QOS" != N/A ]; then LIMIT=$(sacctmgr -nP show qos "$QOS" format=maxsubmitpu 2>/dev/null | head -1 || true); fi
HAVE=$(squeue -h -u "$ME" -p "$PARTITION" 2>/dev/null | wc -l)
if [ -n "$LIMIT" ] && [ $((HAVE + n_gpu)) -gt "$LIMIT" ]; then
  blocked+=("$PARTITION lets you have $LIMIT jobs queued; $HAVE are, and this needs $n_gpu (queue the passes one at a time, or fewer arms or JOBS)")
fi

echo "loop study, pass $PASS on $V: $(for a in "${ARMS[@]}"; do printf '%s -> %s; ' "$a" "$(run_of "$a" "$PASS")"; done)width $WIDTH"
echo "  zips $(basename "$ZIP_C"), $(basename "$ZIP_T") and $(basename "$ZIP_M"); $(lines "$IDS_DEV") dev + $(lines "$IDS_RICH") rich tasks; template $TPL; steps: $STEPS"
for a in "${ARMS[@]}"; do
  r=$(run_of "$a" "$PASS")
  if [ -d "$R/results/$r" ]; then echo "  note: $R/results/$r exists, so $a resumes it (finished tasks are skipped)"; fi
done
if has t1zip && [ -s "$ZIP_T" ]; then echo "  note: $ZIP_T exists, so t1zip only checks it"; fi
if has modelzip && [ -s "$ZIP_M" ]; then echo "  note: $ZIP_M exists, so modelzip only checks it"; fi
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
    echo "    -> job $J"
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
PREP=()   # the check jobs: every arm and the report need them to succeed
if has template; then
  # shellcheck disable=SC2046  # dep prints flags, or nothing
  submit template "$(pwd)" STEP=template OUT="$TPL" -- -J loops_template -p short -c 2 --mem=8G -t 00:20:00 \
    "${LOG_ARGS[@]}" $(dep "afterany:${AFTER:-}") loops/loops.slurm
  PREP+=("$J")
fi
if has t1zip; then
  # shellcheck disable=SC2046
  submit t1zip "$(pwd)" STEP=t1zip SRC="$ZIP_C" OUT="$ZIP_T" REPORT="$OUT/t1_zip.check.md" -- -J loops_t1zip \
    -p short -c 2 --mem=8G -t 00:20:00 "${LOG_ARGS[@]}" $(dep "afterany:${AFTER:-}") loops/loops.slurm
  ZJ=$J
  if [ "$ZIPCHECK" = 1 ]; then
    # The main track's validator, on the zip as it is (its log: $MAIN/logs/package_submission_<job>.out). It keeps
    # its singleton dependency, so it waits for any other package job of ours.
    submit zipcheck "$MAIN" -- --dependency="afterok:$ZJ,singleton" --kill-on-invalid-dep=yes \
      slurm/package_submission.slurm "$ZIP_T"
    PREP+=("$J")
  else
    PREP+=("$ZJ")
  fi
fi
if has modelzip; then
  # shellcheck disable=SC2046
  submit modelzip "$(pwd)" STEP=modelzip SRC="$ZIP_C" OUT="$ZIP_M" MODEL_ID="$M12" REPORT="$OUT/12b_zip.check.md" -- \
    -J loops_modelzip -p short -c 2 --mem=8G -t 00:20:00 "${LOG_ARGS[@]}" $(dep "afterany:${AFTER:-}") loops/loops.slurm
  MJ=$J
  if [ "$ZIPCHECK" = 1 ]; then   # as for the T = 1 zip: the host's compiler on the zip as it is
    submit "modelzip check" "$MAIN" -- --dependency="afterok:$MJ,singleton" --kill-on-invalid-dep=yes \
      slurm/package_submission.slurm "$ZIP_M"
    PREP+=("$J")
  else
    PREP+=("$MJ")
  fi
fi
prep_deps() {   # the dependency strings on every check job, or on AFTER when there is none
  local d=() j
  for j in "${PREP[@]}"; do d+=("afterok:$j"); done
  [ ${#d[@]} -eq 0 ] && d=("afterany:${AFTER:-}")
  printf '%s\n' "${d[@]}"
}
ALL=(); LASTS=()
arm_chain() {   # arm_chain <arm> <run> <chat template or ""> <dedup run length or 0> <extra dependency>: JOBS chained A100
                # jobs; sets LAST
  local name=$1 run=$2 tpl=$3 ded=$4 extra=$5 prev= k first zip pd=()
  zip=$R/submissions/$(field "$name" 3)
  for k in $(seq 1 "$JOBS"); do
    if [ "$k" = 1 ]; then
      mapfile -t pd < <(prep_deps)
      first=$(dep "${pd[@]}" "$extra")
    else
      first=$(dep "afterany:$prev")
    fi
    # shellcheck disable=SC2086  # $first is a list of flags, or nothing
    submit "$name $k/$JOBS" "$MAIN" LANES="$LANES" PINS=1 PACE=0 CHAT_TEMPLATE="$tpl" DEDUP="$ded" -- -J "loops_$name" \
      -p "$PARTITION" -t "$TIME" --gres=gpu:"$GPU":1 "${CON[@]}" --cpus-per-task=$((4 * LANES)) --mem=128G \
      "${LOG_ARGS[@]}" $first baseline/eval_gpu.slurm "$zip" "$TASKS" resume "$run"
    prev=$J
    ALL+=("$J")
  done
  LAST=$prev
}
i=0
for a in "${ARMS[@]}"; do
  tpl=; [ "$a" = block ] && tpl=$TPL
  ded=0; [ "$a" = dedup ] && ded=$DEDUP_RUN
  extra=; [ "$i" -ge "$WIDTH" ] && extra="afterany:${LASTS[$((i - WIDTH))]}"
  arm_chain "$a" "$(run_of "$a" "$PASS")" "$tpl" "$ded" "$extra"
  LASTS+=("$LAST")
  i=$((i + 1))
done
if has smoke; then   # G4's smoke test: one job, no report; it needs none of the checks
  # shellcheck disable=SC2046
  submit smoke "$MAIN" LANES="$LANES" PINS=1 PACE=0 CHAT_TEMPLATE= DEDUP="$DEDUP_RUN" -- -J loops_smoke -p "$PARTITION" \
    -t "${SMOKE_TIME:-00:45:00}" --gres=gpu:"$GPU":1 "${CON[@]}" --cpus-per-task=$((4 * LANES)) --mem=128G \
    "${LOG_ARGS[@]}" $(dep "afterany:${AFTER:-}") baseline/eval_gpu.slurm "$ZIP_C" "$SMOKE_IDS" resume "$SMOKE_RUN"
  echo "  after it, on the login node: python3 loops/loop_report.py --arm dedup=$R/results/$SMOKE_RUN | grep -A6 'dedup proxy did'"
  echo "    (expected = seen, and both > 0: the proxy collapsed the real requests; then queue STEPS=\"dedup report\")"
fi
if has report; then
  specs=(); runs_now=()
  for a in "${ALL_ARMS[@]}"; do
    # G4's arm and the 12B control only once queued or run, so the earlier reports stay as they were
    if [[ $LATE_ARMS == *" $a "* ]] && ! has "$a" && [ ! -d "$R/results/$(run_of "$a" 1)" ]; then continue; fi
    dirs=()
    for k in $(seq 1 "$PASS"); do dirs+=("$R/results/$(run_of "$a" "$k")"); done
    k=$((PASS + 1))   # later passes that already exist: G4's pass 1 is reported against both of G3's control passes
    while [ -d "$R/results/$(run_of "$a" "$k")" ]; do dirs+=("$R/results/$(run_of "$a" "$k")"); k=$((k + 1)); done
    specs+=("$a=$(IFS=,; echo "${dirs[*]}")")
  done
  pairs=(block:ctrl ctrl_t1:ctrl)
  [[ " ${specs[*]} " == *" dedup="* ]] && pairs+=(dedup:ctrl dedup:block)
  [[ " ${specs[*]} " == *" ctrl_12b="* ]] && pairs+=(ctrl_12b:ctrl)
  if [ -n "$OLD" ] && [ "$OLD" != "$V" ]; then   # round 1's runs, as a replicate: only those that exist
    for a in ctrl block; do
      first=$R/results/$(run_of "$a" 1 "$OLD")
      dirs=()
      for d in "$first" "$first"_r*; do [ -d "$d" ] && dirs+=("$d"); done
      [ ${#dirs[@]} -gt 0 ] && specs+=("${a}_$OLD=$(IFS=,; echo "${dirs[*]}")")
    done
    have_old() { [[ " ${specs[*]} " == *" ${1}_$OLD="* ]]; }
    have_old ctrl && have_old block && pairs+=("block_$OLD:ctrl_$OLD")
    have_old ctrl && pairs+=("ctrl:ctrl_$OLD")
    have_old block && pairs+=("block:block_$OLD")
  fi
  deps=("afterany:${AFTER:-}")
  [ ${#ALL[@]} -gt 0 ] && deps=("afterany:$(IFS=:; echo "${ALL[*]}")")
  for j in "${PREP[@]}"; do deps+=("afterok:$j"); done   # a failed check cancels the report too
  # shellcheck disable=SC2046
  submit report "$(pwd)" STEP=report DIR="$OUT/report_${tag}p${PASS}_$(date +%m%d-%H%M)" IDS_DEV="$IDS_DEV" \
    IDS_RICH="$IDS_RICH" ARMSPEC="${specs[*]}" PAIRS="${pairs[*]}" -- -J loops_report -p short -c 2 --mem=8G \
    -t 00:30:00 "${LOG_ARGS[@]}" $(dep "${deps[@]}") loops/loops.slurm
fi
if [ "$GO" = 1 ]; then
  echo "watch: squeue -u \$USER; logs: $LOGS/loops_<step>_<job>.out; the runs: $R/results/loops_*_${V}_a100*; the report: $OUT/report_${tag}p${PASS}_*/"
else
  [ ${#blocked[@]} -eq 0 ] || printf 'blocked: %s\n' "${blocked[@]}"
  echo "dry run: nothing queued. GO=1 queues these jobs."
fi
exit 0
