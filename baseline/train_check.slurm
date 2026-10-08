#!/bin/bash -l
#SBATCH --job-name=opsd_train_check
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Do the 59 OPSD training tasks still run in today's env? CPU only, no model.
# - The grader check behind the sound set (10686757, 2026-09-29) ran before wheelhouse v28 rebuilt the harness.
#   OPSD's rollouts need every training task's sandbox to start, gradable or not.
# - So this reruns that check on the train split only: gold and empty patches (baseline/run_eval.py --gold/--empty)
#   in the PINS=1 env, as baseline/grader_check.slurm does for all 129 tasks.
# - A task whose sandbox can't start shows up with an error in both modes. Sound = gold resolves and empty doesn't.
#   Expected: 39 sound (the rich holdout), and the 20 tasks known to be unsound.
# Needs the split files (opsd_baselines.sh split). opsd_baselines.sh check-train submits it, with both logs in this
# repo's logs/. Submitted by hand, it must start from the repo root, where logs/ is:
#   sbatch --export=ALL,MAIN=$HOME/Gemma4_kaggle_repo baseline/train_check.slurm

MAIN=${MAIN:-$HOME/Gemma4_kaggle_repo}
ROOT=/scratch/$USER/gemma4
SPLIT=${SPLIT:-$ROOT/paper/split}
HARNESS=$ROOT/venvs/harness
OUT=$ROOT/results/opsd_train_check_$SLURM_JOB_ID
SBX=/tmp/$USER/sbx_$SLURM_JOB_ID          # node-local: small-file I/O on NFS scratch is slow
[ -s "$SPLIT/train59.txt" ] || { echo "FAIL: no $SPLIT/train59.txt; run opsd_baselines.sh split first"; exit 1; }
mkdir -p "$SBX" "$OUT"; trap 'rm -rf "$SBX"' EXIT
cd "$MAIN" || { echo "FAIL: no main-track checkout at $MAIN"; exit 1; }
module load explorer
export LITELLM_LOCAL_MODEL_COST_MAP=True
echo "job=$SLURM_JOB_ID node=$(hostname) main=$(git rev-parse --short HEAD) tasks=$(wc -l < "$SPLIT/train59.txt") start=$(date -Is)"
cp --no-preserve=mode "$ROOT/sandbox/testenv/.sig" "$OUT/testenv.sig"
echo "testenv signature $(cat "$OUT/testenv.sig")"

for mode in gold empty; do
  echo "== $mode =="
  "$HARNESS/bin/python" baseline/run_eval.py --$mode --task-ids-file "$SPLIT/train59.txt" --out "$OUT/$mode" \
    --concurrency 8 --sandbox-root "$SBX" --pins
done
echo "== summary =="
for mode in gold empty; do "$HARNESS/bin/python" baseline/summarize_eval.py "$OUT/$mode" --sound "$SPLIT/train59_sound.txt"; done

echo "== against the grader check of 2026-09-29 (10686757) =="
"$HARNESS/bin/python" - "$OUT" "$ROOT/results/grader_check_pins_10686757" <<'EOF'
import json, sys
from pathlib import Path

def rows(d):
    out = {}
    for f in Path(d).glob("**/task_results.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["instance_id"]] = r
    return out

now, then = Path(sys.argv[1]), Path(sys.argv[2])
res = {m: rows(now / m) for m in ("gold", "empty")}
for m, a in res.items():
    b = rows(then / m)
    changed = sorted(t for t in a if t in b and bool(a[t]["resolved"]) != bool(b[t]["resolved"]))
    errors = sorted(t for t in a if a[t].get("error"))
    print(f"{m}: {sum(bool(r['resolved']) for r in a.values())}/{len(a)} resolved; "
          f"changed since then: {' '.join(changed) or 'none'}; with an error: {' '.join(errors) or 'none'}")
sound = sorted(t for t, r in res["gold"].items() if r["resolved"] and not res["empty"].get(t, {}).get("resolved"))
(now / "sound_ids.txt").write_text("".join(f"{t}\n" for t in sound), encoding="utf-8")
print(f"sound now: {len(sound)} of {len(res['gold'])} -> {now / 'sound_ids.txt'}")
EOF
echo "end=$(date -Is)"
