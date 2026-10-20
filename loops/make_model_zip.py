"""The 12B control's zip (README.md section 3): off_args's pinned zip with the model it declares changed, in every
`model:` line of its YAML files (agent.yaml and sub_agents/code_analyzer.yaml). Every other line and entry keeps its
bytes, name, order and timestamp, so the arms differ only in the model; every agent and sub-agent declares the same
one, as a submission must. stdlib only.

    python make_model_zip.py --src <pinned zip> --out <new zip> --model gemma-4-12b-it-qat-w4a16-ct [--report <check.md>]

An --out that already exists is only checked, never rewritten: its sha256 is in the run_meta.json of every run made
with it, so a second pass must use the same bytes. A failed check exits 1 and leaves no new zip behind. loops.slurm
runs it as STEP=modelzip; the main track's slurm/package_submission.slurm then validates the zip with the host's
compiler, and baseline/eval_gpu.slurm serves the model the zip declares.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

# The models a submission may declare (the main track's scripts/package_submission.py, ALLOWED_MODELS, 2026-10-11)
ALLOWED = ("gemma-4-31b-it-qat-w4a16-ct", "gemma-4-12b-it-qat-w4a16-ct", "gemma-4-e4b-it-qat-w4a16-ct",
           "gemma-4-e2b-it-qat-w4a16-ct")
PREFIXES = ("openai/", "google/", "hosted_vllm/", "custom/")   # the scorer's normalize_model_name strips these
MODEL_LINE = re.compile(r"^(?P<lead>[ \t]*(?:-[ \t]+)?model:[ \t]*)(?P<q>[\"']?)(?P<name>[A-Za-z0-9_./-]+)(?P=q)"
                        r"(?P<tail>[ \t]*(?:#.*)?)$")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_yaml(name: str) -> bool:
    return name.endswith((".yaml", ".yml"))


def norm(name: str) -> str:
    name = name.strip().lower()
    for p in PREFIXES:
        name = name[len(p):] if name.startswith(p) else name
    return name


def models_in(text: str) -> list[tuple[int, str]]:
    """(line index, normalized model) of every `model:` line."""
    out = []
    for i, line in enumerate(text.split("\n")):
        m = MODEL_LINE.match(line)
        if m:
            out.append((i, norm(m.group("name"))))
    return out


def edit_models(text: str, model: str) -> str:
    """The file with the model of every `model:` line set to `model`; a prefix, quotes and a comment stay."""
    lines = text.split("\n")
    for i, _ in models_in(text):
        m = MODEL_LINE.match(lines[i])
        name = m.group("name")
        prefix = next((p for p in PREFIXES if name.lower().startswith(p)), "")
        lines[i] = f"{m.group('lead')}{m.group('q')}{name[:len(prefix)]}{model}{m.group('q')}{m.group('tail')}"
    return "\n".join(lines)


def declared(z: zipfile.ZipFile) -> dict[str, list[tuple[int, str]]]:
    """YAML entry -> its `model:` lines, for the entries that have one."""
    out = {}
    for n in z.namelist():
        if is_yaml(n):
            hits = models_in(z.read(n).decode("utf-8"))
            if hits:
                out[n] = hits
    return out


def build(src: Path, out: Path, model: str) -> None:
    with zipfile.ZipFile(src) as zs:
        found = declared(zs)
        names = {m for hits in found.values() for _, m in hits}
        if not found:
            raise ValueError("the source zip declares no model (no `model:` line in its YAML)")
        if len(names) != 1:
            raise ValueError(f"the source zip declares {len(names)} models ({', '.join(sorted(names))}); expected 1")
        with zipfile.ZipFile(out, "w") as zo:
            for info in zs.infolist():
                data = zs.read(info)
                if info.filename in found:
                    data = edit_models(data.decode("utf-8"), model).encode("utf-8")
                new = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                new.compress_type, new.external_attr, new.create_system = info.compress_type, info.external_attr, info.create_system
                zo.writestr(new, data)


def check(src: Path, out: Path, model: str) -> tuple[list[tuple[str, bool, str]], dict]:
    """(name, pass, detail) per check, and facts for the report."""
    res, facts = [], {}
    res.append((f"{model} is a model a submission may declare", model in ALLOWED, ", ".join(ALLOWED)))
    with zipfile.ZipFile(src) as zs, zipfile.ZipFile(out) as zo:
        a, b = zs.infolist(), zo.infolist()
        res.append(("the same entries in the same order", [i.filename for i in a] == [i.filename for i in b],
                    ", ".join(i.filename for i in b)))
        res.append(("the same timestamps", [i.date_time for i in a] == [i.date_time for i in b], ""))
        old, new = declared(zs), declared(zo)
        old_names = sorted({m for hits in old.values() for _, m in hits})
        facts["old_model"] = old_names[0] if len(old_names) == 1 else None
        res.append(("the source declares one model", len(old_names) == 1,
                    ", ".join(f"{n}:{i + 1} {m}" for n, hits in old.items() for i, m in hits) or "no `model:` line"))
        res.append(("the source's model differs from the new one", facts["old_model"] not in (None, model),
                    str(facts["old_model"])))
        res.append(("the same entries declare a model, in the same lines",
                    {n: [i for i, _ in h] for n, h in old.items()} == {n: [i for i, _ in h] for n, h in new.items()},
                    ", ".join(f"{n} line {i + 1}" for n, h in new.items() for i, _ in h)))
        res.append((f"every `model:` line in the new zip names {model}",
                    bool(new) and all(m == model for h in new.values() for _, m in h),
                    ", ".join(sorted({m for h in new.values() for _, m in h}))))
        same = [i.filename for i in a if i.filename not in old and i.filename in zo.namelist()
                and zs.read(i.filename) == zo.read(i.filename)]
        others = [i.filename for i in a if i.filename not in old]
        res.append(("every entry without a `model:` line has the same bytes", same == others,
                    f"{len(same)} of {len(others)}" + ("" if same == others else
                                                       f"; differ: {sorted(set(others) - set(same))}")))
        changed = []
        for n, hits in old.items():
            if n not in zo.namelist():
                res.append((f"{n} differs only in its `model:` lines", False, "missing from the new zip"))
                continue
            ol, nl = zs.read(n).decode("utf-8").split("\n"), zo.read(n).decode("utf-8").split("\n")
            diff = [k for k, (x, y) in enumerate(zip(ol, nl)) if x != y] if len(ol) == len(nl) else None
            ok_lines = diff is not None and diff == [i for i, _ in hits]
            res.append((f"{n} differs only in its `model:` lines", ok_lines,
                        "; ".join(f"line {k + 1}: {ol[k]} -> {nl[k]}" for k in diff) if ok_lines else
                        "line counts differ" if diff is None else f"lines {[k + 1 for k in diff]} differ"))
            changed += [f"{n}:{k + 1}" for k in (diff or [])]
        facts["changed"] = changed
    facts["src_sha"], facts["out_sha"] = sha(src), sha(out)
    return res, facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the pinned zip")
    ap.add_argument("--out", type=Path, required=True, help="the new zip: built if missing, else only checked")
    ap.add_argument("--model", required=True, help="the model the new zip declares, e.g. gemma-4-12b-it-qat-w4a16-ct")
    ap.add_argument("--report", type=Path, help="also write the checks here, as markdown")
    a = ap.parse_args()
    model = norm(a.model)
    if not a.src.is_file():
        print(f"FAIL: no source zip {a.src}")
        return 1
    if a.out.resolve() == a.src.resolve():
        print("FAIL: --out is the source zip")
        return 1
    if model not in ALLOWED:
        print(f"FAIL: {model} is not a model a submission may declare ({', '.join(ALLOWED)})")
        return 1
    built = not a.out.exists()
    target = a.out.with_name(a.out.name + ".tmp") if built else a.out
    if built:
        try:
            build(a.src, target, model)
        except (ValueError, zipfile.BadZipFile, KeyError, UnicodeDecodeError) as e:
            target.unlink(missing_ok=True)
            print(f"FAIL: {e}")
            return 1
    try:
        res, facts = check(a.src, target, model)
    except (zipfile.BadZipFile, UnicodeDecodeError) as e:
        res, facts = [("both zips open", False, str(e))], {}
    ok = all(p for _, p, _ in res)
    if built:
        if ok:
            target.replace(a.out)
        else:
            target.unlink(missing_ok=True)
    lines = [f"# The {model} zip", "",
             f"- source: `{a.src}` sha256 `{facts.get('src_sha', '?')}`, declaring {facts.get('old_model') or '?'}",
             f"- {'built' if built else 'checked, as it already existed'}: `{a.out}`"
             + (f" sha256 `{facts['out_sha']}`" if ok else " (nothing written)" if built else ""),
             f"- changed lines: {', '.join(facts.get('changed') or []) or 'none'}", "",
             "## Checks", ""]
    lines += [f"- {'PASS' if p else 'FAIL'} {n}{': ' + d if d else ''}" for n, p, d in res]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(text, encoding="utf-8")
    if not ok:
        print("FAIL: a check failed" + ("; nothing was written" if built else f"; {a.out} is not the {model} variant of the source"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
