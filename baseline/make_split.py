"""The OPSD paper's train/dev split of the 129 public tasks (baseline/README.md).

- dev, 70 tasks: the main track's dev repos (fastapi, requests, httpx) whose gold patch passes and empty patch fails
  in the training-reward env (PINS=1, grader check 10686757). Every v28 baseline ran on exactly these. The main
  track keeps them in /scratch/$USER/gemma4/train/rft0/dev_sound_ids.txt.
- train, 59 tasks: all the others. That is rich 48 (39 sound, 9 not) and the 11 fastapi/requests tasks the grader
  can't score. OPSD needs no passing test, so the unsound ones are training data too.

The ids go to --out on scratch, never into this repo: they come from the competition's tasks.jsonl.
Checks: dev has 70 tasks, all sound and all from the dev repos; dev and train are disjoint and cover every task.
stdlib only (opsd_baselines.sh split runs it through srun):
    python make_split.py [--tasks FILE] [--dev FILE] [--sound FILE] [--out DIR]
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(f"/scratch/{os.environ.get('USER', '')}/gemma4")
DEV_REPOS = {"fastapi/fastapi", "psf/requests", "encode/httpx"}   # baseline/run_eval.py SPLITS["dev"]


def ids_in(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, ids) -> str:
    text = "".join(f"{i}\n" for i in sorted(ids))
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, default=ROOT / "data/tasks.jsonl")
    ap.add_argument("--dev", type=Path, default=ROOT / "train/rft0/dev_sound_ids.txt")
    ap.add_argument("--sound", type=Path, default=ROOT / "results/grader_check_pins_10686757/sound_ids.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "paper/split")
    a = ap.parse_args()

    repo = {}
    for line in a.tasks.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            repo[t["instance_id"]] = t["repo"]
    dev, sound = set(ids_in(a.dev)), set(ids_in(a.sound))
    train = set(repo) - dev

    problems = []
    if len(dev) != 70:
        problems.append(f"dev has {len(dev)} tasks, not 70")
    if dev - set(repo):
        problems.append(f"{len(dev - set(repo))} dev ids are not in {a.tasks}: {sorted(dev - set(repo))[:5]}")
    if dev - sound:
        problems.append(f"{len(dev - sound)} dev tasks are not in the sound set: {sorted(dev - sound)[:5]}")
    if any(repo.get(i) not in DEV_REPOS for i in dev):
        problems.append("some dev tasks are not from the dev repos")
    if len(repo) != 129 or len(train) != 59:
        problems.append(f"{len(repo)} tasks in all, {len(train)} for training; expected 129 and 59")
    for p in problems:
        print(f"FAIL: {p}")
    if problems:
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    files = {
        "dev70.txt": dev,
        "train59.txt": train,
        "train59_sound.txt": train & sound,
        "train59_unsound.txt": train - sound,
    }
    summary = {"tasks": str(a.tasks), "dev_from": str(a.dev), "sound_from": str(a.sound), "files": {}}
    for name, ids in files.items():
        sha = write_ids(a.out / name, ids)
        by_repo = dict(sorted(Counter(repo[i] for i in ids).items()))
        summary["files"][name] = {"n": len(ids), "by_repo": by_repo, "sha256": sha}
        label = ", ".join(f"{r.split('/')[1]} {n}" for r, n in by_repo.items())
        print(f"{name:20s} {len(ids):3d}  {label}  sha256 {sha[:12]}")
    (a.out / "split.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
