from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/submission_v1/submission_v1.docx"
OUT = ROOT / "paper/submission_v2/submission_v2.docx"
QA = Path(__file__).parent
BLUE = "1F5AA6"
SOURCE_HASH = "57c055ff4ca3fed1476285398256e51f77b85c6d4a544985efdff0131b578ea6"


def element(tag: str, attrs: dict | None = None):
    node = OxmlElement(tag)
    for key, value in (attrs or {}).items():
        node.set(qn(key), str(value))
    return node


def after(anchor: Paragraph, style: str) -> Paragraph:
    node = element("w:p")
    anchor._p.addnext(node)
    p = Paragraph(node, anchor._parent)
    p.style = style
    return p


def before(anchor: Paragraph, text: str, style: str) -> Paragraph:
    return anchor.insert_paragraph_before(text, style)


def set_font(font, size: float, *, bold=False, color="000000"):
    font.name = "Times New Roman"
    font.size = Pt(size)
    font.bold = bold
    font.color.rgb = RGBColor.from_string(color)


def clear_direct_paragraph_format(p: Paragraph):
    props = p._p.get_or_add_pPr()
    for child in list(props):
        if child.tag not in (qn("w:pStyle"), qn("w:sectPr")):
            props.remove(child)


def annotate(anchor: Paragraph, issues: str, text: str, *, lead="Suggested replacement"):
    p = after(anchor, "Review Suggestion")
    label = p.add_run(f"{lead} [{issues}]. ")
    label.bold = True
    p.add_run(text)
    notes.append({"issues": issues, "anchor": anchor.text, "text": p.text})
    if len(anchor.text) < 420:
        anchor.paragraph_format.keep_with_next = True
    return p


def math_run(text: str):
    r = element("m:r")
    rp = element("w:rPr")
    rp.append(element("w:rFonts", {"w:ascii": "Cambria Math", "w:hAnsi": "Cambria Math"}))
    rp.append(element("w:sz", {"w:val": "22"}))
    r.append(rp)
    t = element("m:t")
    t.text = text
    r.append(t)
    return r


def superscript(base_nodes: list, exponent: str):
    s = element("m:sSup")
    base = element("m:e")
    base.extend(base_nodes)
    exp = element("m:sup")
    exp.append(math_run(exponent))
    s.extend([base, exp])
    return s


def moment(exponent: str):
    return [math_run("E["), superscript([math_run("p")], exponent), math_run("]")]


def append_equations(p: Paragraph):
    original = p.text
    start = original.index("h(k) =")
    end = original.index(" (Cauchy")
    p.clear()
    p.add_run(original[:start])
    equation = element("m:oMath")
    equation.append(math_run("h(k) = "))
    fraction = element("m:f")
    numerator, denominator = element("m:num"), element("m:den")
    numerator.extend(moment("k"))
    denominator.extend(moment("k − 1"))
    fraction.extend([numerator, denominator])
    equation.append(fraction)
    p._p.append(equation)
    p.add_run(", and since ")
    inequality = element("m:oMath")
    inequality.append(superscript(moment("k"), "2"))
    inequality.append(math_run(" ≤ "))
    inequality.extend(moment("k − 1"))
    inequality.append(math_run(" "))
    inequality.extend(moment("k + 1"))
    p._p.append(inequality)
    p.add_run(original[end:])


assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH, "Source changed; review before editing."
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = Document(SOURCE)
original = list(doc.paragraphs)
original_text = [p.text for p in original]
notes: list[dict] = []

# Normalize the page system, including the inherited Chinese line grid.
for section in doc.sections:
    section.page_width, section.page_height = Inches(8.2677), Inches(11.6929)
    section.top_margin = section.bottom_margin = section.left_margin = section.right_margin = Inches(1)
    section.header_distance = section.footer_distance = Inches(0.45)
    grid = section._sectPr.find(qn("w:docGrid"))
    if grid is not None:
        section._sectPr.remove(grid)

normal = doc.styles["Normal"]
set_font(normal.font, 11)
normal.paragraph_format.line_spacing = 1.0
normal.paragraph_format.space_before = Pt(0)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.widow_control = True
normal.paragraph_format.keep_with_next = False
normal.paragraph_format.keep_together = False
normal.paragraph_format.left_indent = Inches(0)
normal.paragraph_format.right_indent = Inches(0)
normal.paragraph_format.first_line_indent = Inches(0)

for name in ("Title", "Subtitle", "Heading 1", "Caption"):
    if name not in doc.styles:
        doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)

for name, size, bold in [("Title", 17, True), ("Subtitle", 11, False), ("Heading 1", 12, True), ("Caption", 10, False)]:
    s = doc.styles[name]
    s.base_style = normal
    set_font(s.font, size, bold=bold)
    pf = s.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(10 if name == "Heading 1" else 0)
    pf.space_after = Pt(5)
    pf.keep_with_next = name in ("Title", "Subtitle", "Heading 1")
    pf.keep_together = True
    pf.widow_control = True
    for child in list(s.element.get_or_add_pPr()):
        if child.tag in (qn("w:pBdr"), qn("w:shd")):
            s.element.get_or_add_pPr().remove(child)

review = doc.styles.add_style("Review Suggestion", WD_STYLE_TYPE.PARAGRAPH)
review.base_style = normal
set_font(review.font, 10.5, color=BLUE)
review.paragraph_format.space_before = Pt(3)
review.paragraph_format.space_after = Pt(7)
review.paragraph_format.left_indent = Inches(0.12)
review.paragraph_format.keep_together = True
review.paragraph_format.keep_with_next = False

ref = doc.styles.add_style("Paper Reference", WD_STYLE_TYPE.PARAGRAPH)
ref.base_style = normal
set_font(ref.font, 10)
ref.paragraph_format.space_after = Pt(4)
ref.paragraph_format.left_indent = Inches(0.2)
ref.paragraph_format.first_line_indent = Inches(-0.2)
ref.paragraph_format.keep_together = True

panel = doc.styles.add_style("Figure Panel", WD_STYLE_TYPE.PARAGRAPH)
panel.base_style = normal
set_font(panel.font, 10, bold=True)
panel.paragraph_format.space_before = Pt(0)
panel.paragraph_format.space_after = Pt(2)
panel.paragraph_format.keep_with_next = True

# Explicit formatting prevents the original theme and direct styles overriding sizes.
for i, p in enumerate(original):
    clear_direct_paragraph_format(p)
    p.style = normal
    for r in p.runs:
        set_font(r.font, 11)
        r.font.italic = False
        if r._r.rPr is not None:
            for prop in list(r._r.rPr):
                if prop.tag in (qn("w:spacing"), qn("w:w"), qn("w:position")):
                    r._r.rPr.remove(prop)
    if i == 0:
        p.style = doc.styles["Title"]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            set_font(r.font, 17, bold=True)
    elif i == 1:
        p.style = doc.styles["Subtitle"]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif i == 2:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        for r in p.runs:
            set_font(r.font, 10.5)
    elif i in (3, 5, 13, 17, 19, 27, 29, 32, 36, 41, 45, 48, 50):
        p.style = doc.styles["Heading 1"]
        for r in p.runs:
            set_font(r.font, 12, bold=True)
    elif i in (23, 34):
        p.style = doc.styles["Caption"]
        for r in p.runs:
            set_font(r.font, 10)
    elif i >= 51:
        p.style = ref
        for r in p.runs:
            set_font(r.font, 10)

# Existing hyperlinks remain usable, but blue is reserved for proposed wording.
for rpr in doc.element.xpath(".//w:hyperlink//w:rPr"):
    for tag in ("w:color", "w:rFonts", "w:sz", "w:szCs"):
        for n in rpr.findall(qn(tag)):
            rpr.remove(n)
    rpr.append(element("w:color", {"w:val": "000000"}))
    rpr.append(element("w:rFonts", {"w:ascii": "Times New Roman", "w:hAnsi": "Times New Roman"}))
    rpr.append(element("w:sz", {"w:val": "21"}))

legend = after(original[2], "Review Suggestion")
legend.add_run("Blue text proposes replacement wording for review points 1–5. ").bold = True
legend.add_run("The original prose is retained for comparison; the blue suggestions are alternatives, not additional manuscript claims.")

annotate(original[4], "Q2, Q3", "For finding (1): The chance of one more identical call rises from 78% after two calls to 99% after ten even with a fixed request, showing that accumulating copies is unnecessary for increasing persistence. Fixed-history edits independently show that identical copies reinforce immediate repetition. For finding (2), replace ‘loops can be broken but not removed’ with ‘one-step escapes can be induced’. For finding (3): None of the three whole-task interventions produced a detectable improvement in resolution under the tested budget.")
annotate(original[9], "Q2, Q3", "(1) Persistence rises even with a fixed request. Continuation increases from 78% to 99% under the proxy, consistent with selection of persistent states; fixed-history edits separately establish reinforcement by identical copies. The two model sizes show similar continuation curves once copying has begun (§4).")
annotate(original[10], "Q4", "(2) In the 31B control at T = 0.2, no spontaneous exit occurred in 1,342 observed clean transitions after five or more identical, well-formed calls (§5).")
annotate(original[12], "Q3", "(4) The three whole-task interventions alter repetition, but none produces a detectable improvement in resolution under the tested budget (§7).")

annotate(original[16], "Q1, Q4, Q5", "Replace the continuation-measure definition with: After k identical calls, h(k) is the probability that the next call is identical. Figure 1b uses transitions following well-formed calls on clean steps: no budget warning in the last result and no compaction before the next call. For the proxy at k ≥ 2, the figure additionally requires that the entire current run was collapsed. Paired differences reported for all well-formed steps use a separate population. The conventional exit hazard is 1 − h(k).")

annotate(original[20], "Q2, Q4", "Replace the final two sentences with: Reinforcement predicts that added copies increase the chance of another copy at a fixed history. Selection also predicts increasing continuation: under a fixed request, states with higher repeat probabilities are more likely to survive to longer runs. The proxy tests whether the curve can rise without accumulating copies; the fixed-history edits test reinforcement directly.")

# Native Office Math preserves the original argument while removing literal caret notation.
append_equations(original[24])
annotate(original[24], "Q2, Q5", "Replace the opening sentence with: A rising continuation probability alone does not establish reinforcement. Replace the final two sentences with: This is analogous to frailty selection in survival analysis (Vaupel et al., 1979): more persistent states become enriched among survivors, so continuation rises and the corresponding exit hazard falls. Under a fixed request and stationary, independent draws within a state, the rise can arise through selection alone. Because the two arms reach different states and retain different survivors, their gap does not identify the fraction of the control curve caused by accumulating copies.")

annotate(original[25], "Q1, Q2", "Replace the sentences from ‘Most of the curve’s height’ through the five-to-nine-call difference with: High continuation persists without accumulating copies. In a separate analysis of all well-formed steps, paired differences over tasks (proxy minus control) are +7.6 points (−8.9 to +22.4) at k = 2, −7.0 (−13.3 to −0.8) at k = 3–4, and −4.1 (−7.3 to −1.1) at k = 5–9. These estimates include warning and compaction transitions and are not differences between the clean, fully collapsed values in Figure 1b. The between-arm differences are consistent with a contribution from copies, which G1 tests directly at fixed histories.")
note = annotate(original[25], "Q1", "For the own-exit and long-run comparison: Among runs of three or more calls that ended before the task, own exits occurred in 43 of 91 proxy runs (47%) and 16 of 107 control runs (15%; difference +32 points, +20 to +45). Runs of ten or more occurred 0.25 times per task under the proxy versus 0.33 under the control, about three quarters as often. This ratio compares incidence across arms, not matched survival of the same runs.", lead="Additional replacement")
# Keep the two notes in reading order after the original paragraph.
note._p.getparent().remove(note._p)
first_note = original[25]._p.getnext()
first_note.addnext(note._p)

annotate(original[26], "Q1, Q3", "On clean steps, Gemma 4 12B copies a fresh call 5.8% of the time, versus 2.4% for 31B. A separate all-step comparison gives +3.2 points (+2.2 to +4.2). Once copying begins, the observed continuation curves are similar; the all-step paired intervals include zero, which does not establish equivalence. We did not run the proxy at 12B, so the fixed-request selection test is limited to 31B.")

annotate(original[28], "Q4", "Suggested section heading: Observed exits of long runs. Replace the opening two sentences with: In the 31B control at T = 0.2, no spontaneous exit was observed after five or more identical, well-formed calls on clean steps: 290 of 290 transitions continued at k = 5–9 and 1,052 of 1,052 at k ≥ 10. Replace the final sentence with: These observed long control runs ended following a budget warning, at compaction, or with task termination; this descriptive classification does not by itself identify the causal effect of either intervention. Malformed runs and the 12B control can end spontaneously.")

table_note = annotate(original[34], "Q3", "Replace the final sentence with: The 12B comparison provides an observed large contrast, rather than a formal estimate of the minimum detectable effect. Also qualify the units: resolution and share differences are in percentage points; run-length and frequency differences use calls and runs per task, respectively.")
annotate(original[35], "Q3", "Replace the opening sentence with: None of the three interventions produced a detectable improvement in resolution under the tested budget. Replace the final two sentences with: The resolution intervals retain uncertainty about small and moderate benefits; the two-pass design does not establish equivalence. The large 12B contrast is readily observed, while the effects of improving a small number of task runs remain uncertain. The cross-run rescue count describes the tested runs and does not bound what a new intervention could recover.")

annotate(original[37], "Q2", "Replace the final sentence with: Our tool-call loops show a similar rising curve, and holding the request fixed demonstrates that such a curve can arise without accumulating copies; fixed-history edits separately show reinforcement by identical copies.")
annotate(original[39], "Q3", "Replace ‘Our three levers test that reading, and none raises resolution’ with: Our three whole-task interventions provide further evidence that reducing repetition alone need not improve resolution, although their confidence intervals do not exclude useful benefits. Replace the final clause with: removing the copies produced no detectable increase in resolution in our experiment.")
annotate(original[42], "Q3", "Replace ‘Two passes cannot detect a change of one or two tasks’ with: With two passes, the resolution estimates remain too uncertain to establish equivalence or exclude benefits affecting only a small number of tasks.")
annotate(original[43], "Q2, Q3, Q4", "Replace the sentences beginning ‘Raising the temperature’ and ending ‘causing the failure’ with: Raising the temperature, breaking the pattern, and removing copies altered repetition without producing a detectable resolution improvement. The results are consistent with loops marking difficult states, while leaving open a causal contribution of repetition to failure. Replace the final sentence with: Breaking a loop can free budget, but these interventions did not demonstrate improved task recovery under the tested conditions.")

# A dedicated figure page keeps both panels, their labels, and the caption together.
figure_a, figure_b, caption = original[21], original[22], original[23]
label_a = before(figure_a, "(a)", "Figure Panel")
label_a.paragraph_format.page_break_before = True
label_b = before(figure_b, "(b)", "Figure Panel")
for p in (figure_a, figure_b):
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
for shape in doc.inline_shapes:
    ratio = shape.height / shape.width
    shape.width = Inches(6.1)
    shape.height = int(shape.width * ratio)
caption.paragraph_format.keep_with_next = True
caption_note = annotate(caption, "Q1, Q2, Q4", "For panel (b): Continuation on clean transitions following well-formed calls; proxy values at k ≥ 2 additionally require complete collapse of the current run. Binomial bars are descriptive and do not account for task clustering. For panel (a), replace the image’s claim that the arm gap is the copies’ contribution with: Fixed requests remove accumulating copies; differences between arms can also reflect the states and survivors they reach.", lead="Suggested caption and figure wording")
caption_note.paragraph_format.keep_with_next = False
# Place the full-page figure after the main 31B discussion, so its introduction
# does not occupy an otherwise empty preceding page.
figure_nodes = [label_a._p, figure_a._p, label_b._p, figure_b._p, caption._p, caption_note._p]
for node in figure_nodes:
    node.getparent().remove(node)
for node in figure_nodes:
    original[26]._p.addprevious(node)
original[24].paragraph_format.page_break_before = False
original[26].paragraph_format.page_break_before = True

# Table 1: fit the usable width with readable type and repeatable header.
table = doc.tables[0]
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = False
widths = [1.40, 0.60, 2.20, 0.75, 1.31]
for col, width in zip(table.columns, widths):
    col.width = Inches(width)
for row_index, row in enumerate(table.rows):
    trpr = row._tr.get_or_add_trPr()
    trpr.append(element("w:cantSplit"))
    if row_index == 0:
        trpr.append(element("w:tblHeader"))
    for col_index, (cell, width) in enumerate(zip(row.cells, widths)):
        cell.width = Inches(width)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        cp = cell._tc.get_or_add_tcPr()
        for old in list(cp):
            if old.tag in (qn("w:shd"), qn("w:tcMar"), qn("w:tcBorders")):
                cp.remove(old)
        shade = "E9EDF1" if row_index == 0 else "FFFFFF"
        cp.append(element("w:shd", {"w:fill": shade, "w:val": "clear"}))
        margins = element("w:tcMar")
        for side in ("top", "bottom", "left", "right"):
            margins.append(element(f"w:{side}", {"w:w": "85", "w:type": "dxa"}))
        cp.append(margins)
        for p in cell.paragraphs:
            clear_direct_paragraph_format(p)
            p.style = normal
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.keep_with_next = row_index < len(table.rows) - 1
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_index in (1, 3) else WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                set_font(r.font, 10, bold=row_index == 0)

tp = table._tbl.tblPr
for n in list(tp):
    if n.tag == qn("w:tblBorders"):
        tp.remove(n)
borders = element("w:tblBorders")
for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
    borders.append(element(f"w:{side}", {"w:val": "single", "w:sz": "4", "w:color": "D9D9D9"}))
tp.append(borders)
original[34].paragraph_format.keep_with_next = True
table_note._p.getparent().remove(table_note._p)
table._tbl.addnext(table_note._p)
original[35].paragraph_format.space_before = Pt(6)

# Keep the bibliography as one readable block instead of leaving a short tail.
for p in original[51:-1]:
    p.paragraph_format.keep_with_next = True

# Page numbers are genuine Word fields and are refreshed during PDF export.
for section in doc.sections:
    footer = section.footer.paragraphs[0]
    footer.clear()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_after = Pt(0)
    r = footer.add_run()
    set_font(r.font, 10)
    r._r.append(element("w:fldChar", {"w:fldCharType": "begin"}))
    field = element("w:instrText")
    field.set(qn("xml:space"), "preserve")
    field.text = " PAGE "
    r._r.append(field)
    r._r.append(element("w:fldChar", {"w:fldCharType": "separate"}))
    t = element("w:t")
    t.text = "1"
    r._r.append(t)
    r._r.append(element("w:fldChar", {"w:fldCharType": "end"}))

settings = doc.settings.element
update = settings.find(qn("w:updateFields"))
if update is None:
    update = element("w:updateFields")
    settings.append(update)
update.set(qn("w:val"), "true")

# Remove inherited blank paragraphs and their manual break residue.
for p in original:
    if not p.text and not p._p.xpath(".//w:drawing"):
        p._p.getparent().remove(p._p)

doc.core_properties.title = original_text[0]
doc.core_properties.subject = "Formatted paper with blue proposed wording for review points 1–5"
doc.core_properties.comments = "Blue suggestions are alternatives for author review."
doc.save(OUT)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_HASH

(QA / "annotations.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
(QA / "original_paragraphs.json").write_text(json.dumps(original_text, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"output": str(OUT), "annotations": len(notes), "source_sha256": SOURCE_HASH, "math_objects": len(doc.element.xpath(".//m:oMath")), "table_font_pt": 10, "body_font_pt": 11}, indent=2))
