from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/submission_v2/submission_v2.docx"
OUT = ROOT / "paper/submission_v3/submission_v3.docx"
QA = Path(__file__).parent
SOURCE_HASH = "9f1fed9c32270eab4cc345bb597943cf2c750904d8df1fc89faeabc163f24159"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH, "The source v2 changed."
doc = Document(SOURCE)
old_text = [p.text for p in doc.paragraphs]
ai_text = next(p.text for p in doc.paragraphs if p.text.startswith("Claude (Anthropic)"))
applied = []


def find(prefix):
    matches = [p for p in doc.paragraphs if p.style.name != "Review Suggestion" and p.text.startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]


def rewrite(p, text, issues):
    previous = p.text
    p.clear()
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10 if text.startswith(("Figure 1.", "Table 1.")) else 11)
    r.font.color.rgb = RGBColor(0, 0, 0)
    applied.append({"issues": issues, "before": previous, "after": text})
    return p


# Apply the suggestions as manuscript text, rather than retaining their instructions.
p = find("We study why Gemma")
prefix = p.text.split("Using one-step replays", 1)[0]
rewrite(p, prefix + "Using one-step replays at fixed histories and whole-task runs under three levers, we obtain three findings. "
        "(1) The chance of one more identical call rises from 78% after two calls to 99% after ten even with a fixed request, "
        "showing that accumulating copies is unnecessary for increasing persistence. Fixed-history edits independently show that "
        "identical copies reinforce immediate repetition. (2) One-step escapes can be induced: where the model, free to think, chose "
        "not to, a forced thought lets the next call escape 58% of the time, against 11%. (3) None of the three whole-task "
        "interventions produced a detectable improvement in resolution under the tested budget.", "Q2, Q3")

rewrite(find("(1) Lock-in is mostly"), "(1) Persistence rises even with a fixed request. Continuation increases from 78% to 99% "
        "under the proxy, consistent with selection of persistent states; fixed-history edits separately establish reinforcement "
        "by identical copies. The two model sizes show similar continuation curves once copying has begun (§4).", "Q2, Q3")
rewrite(find("(2) Loops end from outside"), "(2) In the 31B control at T = 0.2, no spontaneous exit occurred in 1,342 observed "
        "clean transitions after five or more identical, well-formed calls (§5).", "Q4")
rewrite(find("(4) No lever"), "(4) The three whole-task interventions alter repetition, but none produces a detectable improvement "
        "in resolution under the tested budget (§7).", "Q3")

p = find("A call is stale")
prefix, tail = p.text.split("After k identical calls,", 1)
ending = "An escape," + tail.split("An escape,", 1)[1]
rewrite(p, prefix + "After k identical calls, h(k) is the probability that the next call is identical. Figure 1b uses transitions "
        "following well-formed calls on clean steps: no budget warning in the last result and no compaction before the next call. "
        "For the proxy at k ≥ 2, the figure additionally requires that the entire current run was collapsed. Paired differences "
        "reported for all well-formed steps use a separate population. The conventional exit hazard is 1 − h(k). " + ending, "Q1, Q4, Q5")

p = find("In the control, a fresh call")
prefix = p.text.split("Under reinforcement,", 1)[0].replace("the hazard is", "the continuation probability is")
rewrite(p, prefix + "Reinforcement predicts that added copies increase the chance of another copy at a fixed history. "
        "Selection also predicts increasing continuation: under a fixed request, states with higher repeat probabilities are more "
        "likely to survive to longer runs. The proxy tests whether the curve can rise without accumulating copies; the fixed-history "
        "edits test reinforcement directly.", "Q2, Q4, Q5")

# Preserve the two native Word equations while applying the revised interpretation.
p = find("A rising hazard")
equations = [deepcopy(x) for x in p._p.xpath("./m:oMath")]
assert len(equations) == 2
old_math_paragraph = p.text
p.clear()
p.add_run("A rising continuation probability alone does not establish reinforcement. Suppose each state s draws the same call "
          "again with a fixed probability p, as it does when the request is held fixed. Over the states where runs start, ")
p._p.append(equations[0])
p.add_run(", and since ")
p._p.append(equations[1])
p.add_run(" (Cauchy–Schwarz), h(k) never falls with k, whatever the mix of states. This is analogous to frailty selection in "
          "survival analysis (Vaupel et al., 1979): more persistent states become enriched among survivors, so continuation rises "
          "and the corresponding exit hazard falls. Under a fixed request and stationary, independent draws within a state, "
          "the rise can arise through selection alone. Because the two arms reach different states and retain different "
          "survivors, their gap does not identify the fraction of the control curve caused by accumulating copies.")
applied.append({"issues": "Q2, Q5", "before": old_math_paragraph, "after": p.text, "native_equations_preserved": 2})

p = find("With the request fixed, the hazard")
first = p.text.split("Most of the curve's height", 1)[0].replace("the hazard still rises", "the continuation probability still rises")
last = "One step ahead," + p.text.split("One step ahead,", 1)[1]
rewrite(p, first + "High continuation persists without accumulating copies. In a separate analysis of all well-formed steps, "
        "paired differences over tasks (proxy minus control) are +7.6 points (−8.9 to +22.4) at k = 2, −7.0 (−13.3 to −0.8) "
        "at k = 3–4, and −4.1 (−7.3 to −1.1) at k = 5–9. These estimates include warning and compaction transitions and "
        "are not differences between the clean, fully collapsed values in Figure 1b. The between-arm differences are consistent "
        "with a contribution from copies, which the fixed-history edits test directly. Among runs of three or more calls that "
        "ended before the task, own exits occurred in 43 of 91 proxy runs (47%) and 16 of 107 control runs (15%; difference "
        "+32 points, +20 to +45). Runs of ten or more occurred 0.25 times per task under the proxy versus 0.33 under the "
        "control, about three quarters as often. This ratio compares incidence across arms, not matched survival of the same runs. "
        + last, "Q1, Q2")

p12 = rewrite(find("Gemma 4 12B starts runs"), "On clean steps, Gemma 4 12B copies a fresh call 5.8% of the time, versus "
              "2.4% for 31B. A separate all-step comparison gives +3.2 points (+2.2 to +4.2). Once copying begins, the observed "
              "continuation curves are similar; the all-step paired intervals include zero, which does not establish equivalence. "
              "We did not run the proxy at 12B, so the fixed-request selection test is limited to 31B.", "Q1, Q3")

rewrite(find("5 Long runs end"), "5 Observed exits of long runs", "Q4")
p = find("At T = 0.2 the agent does not")
middle = "Of the 58" + p.text.split("Of the 58", 1)[1].split("Past five calls,", 1)[0]
rewrite(p, "In the 31B control at T = 0.2, no spontaneous exit was observed after five or more identical, well-formed calls "
        "on clean steps: 290 of 290 transitions continued at k = 5–9 and 1,052 of 1,052 at k ≥ 10. " + middle +
        "These observed long control runs ended following a budget warning, at compaction, or with task termination; "
        "this descriptive classification does not by itself identify the causal effect of either intervention. Malformed runs "
        "and the 12B control can end spontaneously.", "Q4")

rewrite(find("Table 1."), "Table 1. Whole-task levers against the control, Gemma 4 31B on 109 tasks unless noted. "
        "Differences are lever minus control, paired over tasks, with 95% intervals from 2,000 task resamples. Resolution "
        "and share differences are in percentage points; run-length and frequency differences use calls and runs per task, "
        "respectively. The 12B comparison provides an observed large contrast, rather than a formal estimate of the minimum "
        "detectable effect.", "Q3")

p = find("None of them measurably")
middle = "T = 1 un-looped" + p.text.split("T = 1 un-looped", 1)[1].split("The design detects", 1)[0]
rewrite(p, "None of the three interventions produced a detectable improvement in resolution under the tested budget. " + middle +
        "The resolution intervals retain uncertainty about small and moderate benefits; the two-pass design does not establish "
        "equivalence. The large 12B contrast is readily observed, while the effects of improving a small number of task runs "
        "remain uncertain. The cross-run rescue count describes the tested runs and does not bound what a new intervention "
        "could recover.", "Q3")

p = find("Repetition in text.")
prefix = p.text.split("Our tool-call loops", 1)[0]
rewrite(p, prefix + "Our tool-call loops show a similar rising curve, and holding the request fixed demonstrates that such "
        "a curve can arise without accumulating copies; fixed-history edits separately show reinforcement by identical copies.", "Q2")

p = find("Repeats as symptoms.")
text = p.text.replace("Our three levers test that reading, and none raises resolution.",
        "Our three whole-task interventions provide further evidence that reducing repetition alone need not improve resolution, "
        "although their confidence intervals do not exclude useful benefits.")
text = text.replace("removing the copies does not help it solve more.",
                    "removing the copies produced no detectable increase in resolution in our experiment.")
rewrite(p, text, "Q3")

p = find("The evidence comes from two sizes")
rewrite(p, p.text.replace("Two passes cannot detect a change of one or two tasks.",
        "With two passes, the resolution estimates remain too uncertain to establish equivalence or exclude benefits affecting "
        "only a small number of tasks."), "Q3")

p = find("Three things follow for agent builders.")
start = p.text.split("Raising the temperature,", 1)[0]
future = "A lever worth testing next" + p.text.split("A lever worth testing next", 1)[1].split("Breaking a loop frees", 1)[0]
rewrite(p, start + "Raising the temperature, breaking the pattern, and removing copies altered repetition without producing "
        "a detectable resolution improvement. The results are consistent with loops marking difficult states, while leaving "
        "open a causal contribution of repetition to failure. " + future + "Breaking a loop can free budget, but these interventions "
        "did not demonstrate improved task recovery under the tested conditions.", "Q2, Q3, Q4")

caption = rewrite(find("Figure 1."), "Figure 1. (a) A tool-call loop and the fixed-request test, on a synthetic call c. Before "
        "the next call at k = 4, the control shows every copy; the proxy collapses the run to its first turn. The rates illustrate "
        "pooled three-to-four-call bins, not probabilities of the synthetic call. (b) Continuation on clean transitions following "
        "well-formed calls; proxy values at k ≥ 2 additionally require complete collapse of the current run. Binomial 95% bars "
        "are descriptive and do not account for task clustering.", "Q1, Q2, Q4")

review_paragraphs = [p for p in doc.paragraphs if p.style.name == "Review Suggestion"]
assert len(review_paragraphs) == 19
for p in review_paragraphs:
    p._p.getparent().remove(p._p)

# The original source uses a local caption style; normalize its size explicitly.
for p in doc.paragraphs:
    if p.text.startswith(("Figure 1.", "Table 1.")):
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_together = True
        for r in p.runs:
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(0, 0, 0)
    elif p.style.name == "Heading 1":
        for r in p.runs:
            r.font.size = Pt(12)
            r.font.bold = True

# Replace the embedded PNGs with the updated code-native figures.
picture_paragraphs = [p for p in doc.paragraphs if p._p.xpath(".//w:drawing")]
assert len(picture_paragraphs) == 2
for p, name in zip(picture_paragraphs, ("fig1a_design.png", "fig1b_hazard.png")):
    p.clear()
    p.add_run().add_picture(str(OUT.parent / "figures" / name), width=Inches(5.8))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(3)

# Restore figure placement after the §4 introduction. The introduction and both
# panels share a page; the following equations begin a new page.
intro4 = find("In the control, a fresh call")
labela = next(p for p in doc.paragraphs if p.text == "(a)")
labelb = next(p for p in doc.paragraphs if p.text == "(b)")
figure_nodes = [labela._p, picture_paragraphs[0]._p, labelb._p, picture_paragraphs[1]._p, caption._p]
for node in figure_nodes:
    node.getparent().remove(node)
cursor = intro4._p
for node in figure_nodes:
    cursor.addnext(node)
    cursor = node
labela.paragraph_format.page_break_before = False
find("4 Selected").paragraph_format.page_break_before = True
intro4.paragraph_format.keep_with_next = True
find("A rising continuation probability").paragraph_format.page_break_before = True
p12.paragraph_format.page_break_before = False
caption.paragraph_format.keep_with_next = False
find("Table 1.").paragraph_format.keep_with_next = True

doc.core_properties.subject = "Paper with the accepted revisions from submission_v2 applied"
doc.core_properties.comments = ""
doc.core_properties.title = doc.paragraphs[0].text
if "Review Suggestion" in doc.styles:
    doc.styles["Review Suggestion"].delete()

assert next(p.text for p in doc.paragraphs if p.text.startswith("Claude (Anthropic)")) == ai_text
assert len(doc.element.xpath(".//m:oMath")) == 2
assert len(applied) == 18, len(applied)
assert not any("Suggested replacement" in p.text or "[Q" in p.text or "Blue text proposes" in p.text for p in doc.paragraphs)
assert all(r.font.color.rgb is None or str(r.font.color.rgb) != "1F5AA6" for p in doc.paragraphs for r in p.runs)
OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH

# Text-based counts are a cross-check. Word's native counter is captured at export.
def words(text):
    return len(text.split())

counted = []
for item in doc.element.body:
    # Word text runs must be concatenated within each paragraph, then separated.
    paragraphs = item.xpath(".//w:p") if item.tag == qn("w:tbl") else [item]
    text = "\n".join("".join(p.xpath(".//w:t/text() | .//m:t/text()")) for p in paragraphs)
    if text.strip() == "References":
        break
    counted.append(text)
report = {"source": str(SOURCE), "source_sha256": SOURCE_HASH, "output": str(OUT), "applied_blue_suggestions": 18,
          "removed_review_paragraphs": 19, "user_ai_disclosure_preserved": True, "Q6_deferred": True,
          "native_equations": 2, "whitespace_count_before_references": words("\n".join(counted))}
(QA / "applied_changes.json").write_text(json.dumps(applied, ensure_ascii=False, indent=2), encoding="utf-8")
(QA / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
(QA / "counted_text.txt").write_text("\n".join(counted), encoding="utf-8")
print(json.dumps(report, indent=2))
