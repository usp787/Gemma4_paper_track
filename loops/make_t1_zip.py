"""G3's T = 1 zip (README.md section 3): off_args's pinned zip with one line changed, the temperature in
configs/sampling.yaml. Every other entry keeps its bytes, name, order and timestamp, so the two arms differ only in the
sampling temperature (the main agent's and the analyzer sub-agent's: both include configs/sampling.yaml). stdlib only.

    python make_t1_zip.py --src <pinned zip> --out <T=1 zip> [--temperature 1.0] [--report <check.md>]

An --out that already exists is only checked, never rewritten: its sha256 is in the run_meta.json of every run made
with it, so a second pass must use the same bytes. A failed check exits 1 and leaves no new zip behind. loops.slurm
runs it as STEP=t1zip; the main track's slurm/package_submission.slurm then validates the zip with the host's compiler.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

SAMPLING = "configs/sampling.yaml"
TEMP_LINE = re.compile(r"^temperature:[ \t]*([0-9]*\.?[0-9]+)[ \t]*(#.*)?$")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(t: float) -> str:
    return repr(float(t))   # 1.0 -> "1.0", 0.2 -> "0.2"


def edit_sampling(text: str, temperature: float) -> tuple[str, str]:
    """The file with its one temperature line set to `temperature`, and the old value. Raises ValueError unless the
    file has exactly one top-level temperature line."""
    lines = text.split("\n")
    hits = [i for i, line in enumerate(lines) if TEMP_LINE.match(line)]
    if len(hits) != 1:
        raise ValueError(f"{SAMPLING} has {len(hits)} top-level temperature lines; expected 1")
    m = TEMP_LINE.match(lines[hits[0]])
    lines[hits[0]] = f"temperature: {fmt(temperature)}" + (f"  {m.group(2)}" if m.group(2) else "")
    return "\n".join(lines), m.group(1)


def build(src: Path, out: Path, temperature: float) -> None:
    with zipfile.ZipFile(src) as zs, zipfile.ZipFile(out, "w") as zo:
        for info in zs.infolist():
            data = zs.read(info)
            if info.filename == SAMPLING:
                data = edit_sampling(data.decode("utf-8"), temperature)[0].encode("utf-8")
            new = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new.compress_type, new.external_attr, new.create_system = info.compress_type, info.external_attr, info.create_system
            zo.writestr(new, data)


def check(src: Path, out: Path, temperature: float) -> tuple[list[tuple[str, bool, str]], dict]:
    """(name, pass, detail) per check, and facts for the report."""
    res, facts = [], {}
    with zipfile.ZipFile(src) as zs, zipfile.ZipFile(out) as zo:
        a, b = zs.infolist(), zo.infolist()
        res.append(("the same entries in the same order", [i.filename for i in a] == [i.filename for i in b],
                    ", ".join(i.filename for i in b)))
        res.append(("the same timestamps", [i.date_time for i in a] == [i.date_time for i in b], ""))
        res.append((f"{SAMPLING} is in the zip", SAMPLING in zs.namelist() and SAMPLING in zo.namelist(), ""))
        same = [i.filename for i in a if i.filename != SAMPLING and i.filename in zo.namelist()
                and zs.read(i.filename) == zo.read(i.filename)]
        others = [i.filename for i in a if i.filename != SAMPLING]
        res.append(("every other entry has the same bytes", same == others,
                    f"{len(same)} of {len(others)}" + ("" if same == others else
                                                       f"; differ: {sorted(set(others) - set(same))}")))
        if SAMPLING in zs.namelist() and SAMPLING in zo.namelist():
            old, new = zs.read(SAMPLING).decode("utf-8"), zo.read(SAMPLING).decode("utf-8")
            ol, nl = old.split("\n"), new.split("\n")
            diff = [(x, y) for x, y in zip(ol, nl) if x != y] if len(ol) == len(nl) else None
            ok_lines = diff is not None and len(diff) == 1 and all(TEMP_LINE.match(s) for s in diff[0])
            res.append((f"{SAMPLING} differs in exactly one line, the temperature", bool(ok_lines),
                        " -> ".join(diff[0]) if ok_lines else f"{'line counts differ' if diff is None else diff}"))
            if ok_lines:
                facts["old_t"] = float(TEMP_LINE.match(diff[0][0]).group(1))
                facts["new_t"] = float(TEMP_LINE.match(diff[0][1]).group(1))
                res.append((f"the new temperature is {fmt(temperature)}", facts["new_t"] == temperature,
                            fmt(facts["new_t"])))
                res.append(("the source's temperature differs from it", facts["old_t"] != temperature,
                            fmt(facts["old_t"])))
            facts["sampling_new"] = new
    facts["src_sha"], facts["out_sha"] = sha(src), sha(out)
    return res, facts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="the pinned zip")
    ap.add_argument("--out", type=Path, required=True, help="the T=1 zip: built if missing, else only checked")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--report", type=Path, help="also write the checks here, as markdown")
    a = ap.parse_args()
    if not a.src.is_file():
        print(f"FAIL: no source zip {a.src}")
        return 1
    if a.out.resolve() == a.src.resolve():
        print("FAIL: --out is the source zip")
        return 1
    built = not a.out.exists()
    target = a.out.with_name(a.out.name + ".tmp") if built else a.out
    if built:
        try:
            build(a.src, target, a.temperature)
        except (ValueError, zipfile.BadZipFile, KeyError) as e:
            target.unlink(missing_ok=True)
            print(f"FAIL: {e}")
            return 1
    try:
        res, facts = check(a.src, target, a.temperature)
    except zipfile.BadZipFile as e:
        res, facts = [("both zips open", False, str(e))], {}
    ok = all(p for _, p, _ in res)
    if built:
        if ok:
            target.replace(a.out)
        else:
            target.unlink(missing_ok=True)
    lines = [f"# The T = {fmt(a.temperature)} zip", "",
             f"- source: `{a.src}` sha256 `{facts.get('src_sha', '?')}`",
             f"- {'built' if built else 'checked, as it already existed'}: `{a.out}`"
             + (f" sha256 `{facts['out_sha']}`" if ok else " (nothing written)" if built else ""), "",
             "## Checks", ""]
    lines += [f"- {'PASS' if p else 'FAIL'} {n}{': ' + d if d else ''}" for n, p, d in res]
    if "sampling_new" in facts:
        lines += ["", f"## {SAMPLING} in the new zip", "", "```yaml", facts["sampling_new"].rstrip("\n"), "```"]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(text, encoding="utf-8")
    if not ok:
        print("FAIL: a check failed" + ("; nothing was written" if built else f"; {a.out} is not the T=1 variant of the source"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
