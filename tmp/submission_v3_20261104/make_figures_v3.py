"""Render only the paper's two code-native figures with the accepted wording."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/figures/make_figures.py"
OUT = ROOT / "paper/submission_v3/figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SOURCE.parent))
source = SOURCE.read_text(encoding="utf-8")
changes = {
    "budget warning in the result:\\nthe run ends from outside": "budget warning in the result:\\nobserved exit after warning",
    "copies seal the escapes": "observed arm rates",
    '"With the request fixed, each step is a fresh draw from the same prompt: the chance to continue can "\n            "rise with k only\\nthrough which states keep looping (selection). The gap to the control is what the copies "\n            "add (reinforcement)."': '"Fixed requests remove accumulating copies; differences between arms can also reflect\\n"\n            "the states and survivors they reach."',
    "request held fixed, still 0.78 → 0.99:\\nmost of the height is selection": "request held fixed, still 0.78 → 0.99:\\npersistence rises without new copies",
    "copies close\\nthe last escapes": "observed arm\\ncurves differ",
}
for old, new in changes.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
namespace = {"__name__": "submission_v3_figures", "__file__": str(SOURCE)}
exec(compile(source, str(SOURCE), "exec"), namespace)
namespace["OUT"] = OUT
namespace["fig1a"]()
namespace["fig1b"]()
(Path(__file__).parent / "figure_wording_changes.json").write_text(
    json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({"figure_directory": str(OUT), "files": sorted(x.name for x in OUT.iterdir())}, indent=2))
