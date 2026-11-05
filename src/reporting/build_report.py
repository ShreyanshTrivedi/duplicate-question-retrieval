"""Builds the Project 8 IR course report (.docx), 7 pages, with figures.

Page setup and named styles are inherited from the course template
(docs/IR_report formatt.docx); its body, header text and metadata are
discarded. Every number comes from reports/comparison_table.csv,
reports/segmented_metrics.csv, reports/pr_curve_data.json, the
data/processed parquet statistics, the evaluation run log, or the
documented intermediate-round figures (DECISIONS.md, gap_analysis.md,
comparison_table_before_fix.csv, scoped_full_evaluation.log).
"""

import csv
import datetime as dt
import json
import os
import re
import sys
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import _Cell

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.reporting import make_figures  # noqa: E402
from src.retrieval.hybrid_rrf import reciprocal_rank_fusion  # noqa: E402

FORMAT_DOC = os.path.join(ROOT, "docs", "IR_report formatt.docx")
REPORTS = os.path.join(ROOT, "reports")
FIG = os.path.join(REPORTS, "figures")
OUTPUT = os.environ.get("REPORT_OUT") or os.path.join(REPORTS, "IR_project8_report.docx")
LATENCY_LOG = os.path.join(REPORTS, "ui_redesign_evaluation.err.log")

NAVY = "1B365D"
ROW_TINT = "F9FAFB"
RULE = "E0E0E0"
ABSTRACT_FILL = "F4F6F8"
TEXT_WIDTH_IN = 6.5
BODY_PT = 11
TABLE_PT = 9
CAPTION_PT = 9
BODY_AFTER = Pt(6)
BULLET_AFTER = Pt(3)
BODY_LINE = 1.15
TABLE_GAP = Pt(8)
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


# ------------------------------------------------------------------ document

def fresh_document_from_format() -> Document:
    """Opens the course template and strips everything except styles and page setup."""
    doc = Document(FORMAT_DOC)
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    for section in doc.sections:
        for part in (section.header, section.footer):
            for p in part.paragraphs:
                for r in list(p.runs):
                    r._r.getparent().remove(r._r)
    cp = doc.core_properties
    cp.author = ""
    cp.last_modified_by = ""
    cp.title = "Project 8: Duplicate Question Retrieval"
    cp.subject = ""
    cp.comments = ""
    cp.keywords = ""
    return doc


# --------------------------------------------------------------- text helpers

_INLINE = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|~.+?~)")


def add_runs(paragraph, text, size=None, color=None):
    """Adds text with **bold**, *italic*, `code` and ~subscript~ inline markup."""
    for chunk in _INLINE.split(text):
        if not chunk:
            continue
        if chunk.startswith("**"):
            run = paragraph.add_run(chunk[2:-2])
            run.bold = True
        elif chunk.startswith("~"):
            run = paragraph.add_run(chunk[1:-1])
            run.font.subscript = True
        elif chunk.startswith("`"):
            run = paragraph.add_run(chunk[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(size if size else BODY_PT - 1)
            if color:
                run.font.color.rgb = RGBColor.from_string(color)
            continue
        elif chunk.startswith("*"):
            run = paragraph.add_run(chunk[1:-1])
            run.italic = True
        else:
            run = paragraph.add_run(chunk)
        run.font.size = Pt(size if size else BODY_PT)
        if color:
            run.font.color.rgb = RGBColor.from_string(color)
    return paragraph


def new_par(container):
    """Appends a paragraph to a document or table cell."""
    return container.add_paragraph()


def set_outline_level(paragraph, level):
    el = OxmlElement("w:outlineLvl")
    el.set(qn("w:val"), str(level))
    paragraph._p.get_or_add_pPr().append(el)


def heading(doc, text, size=15, level=0):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 0 else 10)
    p.paragraph_format.space_after = Pt(6 if level == 0 else 4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)
    set_outline_level(p, level)
    return p


def subheading(doc, text):
    return heading(doc, text, size=13, level=1)


def para(container, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, after=BODY_AFTER, keep_next=False):
    p = new_par(container)
    p.alignment = align
    p.paragraph_format.space_after = after
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = BODY_LINE
    p.paragraph_format.keep_with_next = False
    return add_runs(p, text, size=size)


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = BULLET_AFTER
    p.paragraph_format.line_spacing = BODY_LINE
    p._p.get_or_add_pPr().append(parse_xml(f'<w:contextualSpacing {nsdecls("w")} w:val="0"/>'))
    return add_runs(p, text)


def caption(container, text, kind="table", width_in=None):
    """kind='table' is bold and above its table; kind='figure' is centred and below its figure."""
    p = new_par(container)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    if kind == "table":
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.bold = True
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
    run.font.size = Pt(CAPTION_PT)
    return p


def picture(container, filename, width_in, cap):
    p = new_par(container)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = not isinstance(container, _Cell)
    p.add_run().add_picture(os.path.join(FIG, filename), width=Inches(width_in))
    caption(container, cap, kind="figure")


# -------------------------------------------------------------- table helpers

def shade(cell, fill):
    cell._tc.get_or_add_tcPr().append(
        parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill}"/>')
    )


def cell_borders(cell, **edges):
    """edges: side=(size_eighths, colour); sides not given are nil. Call before shade()."""
    xml = "".join(
        f'<w:{s} w:val="single" w:sz="{edges[s][0]}" w:space="0" w:color="{edges[s][1]}"/>'
        if s in edges else f'<w:{s} w:val="nil"/>'
        for s in ("top", "left", "bottom", "right")
    )
    cell._tc.get_or_add_tcPr().append(parse_xml(f"<w:tcBorders {nsdecls('w')}>{xml}</w:tcBorders>"))


def set_col_widths(table, widths_in):
    table.autofit = False
    for row in table.rows:
        for cell, w in zip(row.cells, widths_in):
            cell.width = Inches(w)


def zero_cell_margins(table):
    tbl_pr = table._tbl.tblPr
    mar = parse_xml(
        f'<w:tblCellMar {nsdecls("w")}><w:left w:w="0" w:type="dxa"/><w:right w:w="0" w:type="dxa"/></w:tblCellMar>'
    )
    look = tbl_pr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)
    else:
        tbl_pr.append(mar)


def set_cell_margins(table, top_bottom_pt=3, left_right_pt=5):
    """Cell padding (tblCellMar), in points."""
    tb, lr = int(top_bottom_pt * 20), int(left_right_pt * 20)
    mar = parse_xml(
        f'<w:tblCellMar {nsdecls("w")}><w:top w:w="{tb}" w:type="dxa"/><w:left w:w="{lr}" w:type="dxa"/>'
        f'<w:bottom w:w="{tb}" w:type="dxa"/><w:right w:w="{lr}" w:type="dxa"/></w:tblCellMar>'
    )
    tbl_pr = table._tbl.tblPr
    look = tbl_pr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)
    else:
        tbl_pr.append(mar)


def finish_rows(table, glue):
    """Rows never split. Only short tables (glue=True) are held on one page."""
    for i, row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if glue and i < len(table.rows) - 1:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True


def cell_par(cell, text, bold=False, size=TABLE_PT, color=None, align=WD_ALIGN_PARAGRAPH.CENTER, first=True):
    p = cell.paragraphs[0] if first else cell.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    add_runs(p, f"**{text}**" if bold and text else text, size=size, color=color)
    return p


def data_table(container, headers, rows, widths_in, bold_best_cols=(), left_cols=(0,), size=TABLE_PT,
               spacer=True, together=True):
    """Navy-header, lightly ruled table in the course template's style (9 pt text)."""
    size = TABLE_PT
    table = container.add_table(len(rows) + 1, len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_col_widths(table, widths_in)
    set_cell_margins(table)
    for c, h in enumerate(headers):
        cell = table.rows[0].cells[c]
        shade(cell, NAVY)
        cell_par(cell, h, bold=True, size=size, color="FFFFFF",
                 align=WD_ALIGN_PARAGRAPH.LEFT if c in left_cols else WD_ALIGN_PARAGRAPH.CENTER)
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    best = {c: max(float(r[c]) for r in rows) for c in bold_best_cols}
    for r_i, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = table.rows[r_i].cells[c]
            cell_borders(cell, top=(4, RULE), bottom=(4, RULE))
            if r_i % 2 == 1:
                shade(cell, ROW_TINT)
            is_best = c in best and float(value) == best[c]
            cell_par(cell, value, bold=is_best, size=size,
                     align=WD_ALIGN_PARAGRAPH.LEFT if c in left_cols else WD_ALIGN_PARAGRAPH.CENTER)
    finish_rows(table, glue=len(table.rows) <= 6)
    if spacer and not isinstance(container, _Cell):
        gap = container.add_paragraph()
        gap.paragraph_format.space_before = Pt(0)
        gap.paragraph_format.space_after = Pt(0)
        gap.paragraph_format.line_spacing = TABLE_GAP
    return table


def side_by_side(doc, left_w, right_w):
    """Borderless one-row, two-cell table; returns (left_cell, right_cell)."""
    table = doc.add_table(1, 2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_col_widths(table, [left_w, right_w])
    zero_cell_margins(table)
    return table.cell(0, 0), table.cell(0, 1)


def drop_leading_empty(cell):
    """Removes the cell's initial empty paragraph once other content has been added."""
    first = cell.paragraphs[0]
    if (not first.text.strip() and not first._p.xpath(".//w:drawing")
            and not first._p.xpath('.//*[local-name()="oMath"]')) and len(cell._tc.findall(qn("w:p"))) + len(
            cell._tc.findall(qn("w:tbl"))) > 1:
        cell._tc.remove(first._p)


def shrink_trailing_empty(cell):
    last = cell.paragraphs[-1]
    if (not last.text.strip() and not last._p.xpath(".//w:drawing")
            and not last._p.xpath('.//*[local-name()="oMath"]')):
        last.paragraph_format.space_after = Pt(0)
        last.paragraph_format.line_spacing = Pt(2)


def boxed(doc, fill, left_rule=False):
    table = doc.add_table(1, 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_col_widths(table, [TEXT_WIDTH_IN])
    cell = table.rows[0].cells[0]
    if left_rule:
        cell_borders(cell, left=(24, NAVY))
    else:
        cell_borders(cell)
    shade(cell, fill)
    set_cell_margins(table, top_bottom_pt=5, left_right_pt=8)
    return cell


# ----------------------------------------------------------- native equations

def mr(text, plain=False):
    rpr = '<m:rPr><m:sty m:val="p"/></m:rPr>' if plain else ""
    return f'<m:r>{rpr}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def msub(base, sub):
    return f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>"


def mfrac(num, den):
    return f"<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>"


def mparen(x, beg="(", end=")"):
    return (f'<m:d><m:dPr><m:begChr m:val="{beg}"/><m:endChr m:val="{end}"/></m:dPr>'
            f"<m:e>{x}</m:e></m:d>")


def mnary(sub, sup, body):
    hide = '<m:supHide m:val="1"/>' if not sup else ""
    return (f'<m:nary><m:naryPr><m:chr m:val="&#8721;"/><m:limLoc m:val="undOvr"/>{hide}</m:naryPr>'
            f"<m:sub>{sub}</m:sub><m:sup>{sup}</m:sup><m:e>{body}</m:e></m:nary>")


def mfunc(name, arg):
    return f"<m:func><m:fName>{mr(name, True)}</m:fName><m:e>{arg}</m:e></m:func>"


def equation(container, inner, inline=False, after=BODY_AFTER):
    p = new_par(container)
    p.paragraph_format.space_after = after
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = BODY_LINE
    if inline:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p._p.append(parse_xml(f'<m:oMath xmlns:m="{M_NS}">{inner}</m:oMath>'))
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p._p.append(parse_xml(f'<m:oMathPara xmlns:m="{M_NS}"><m:oMath>{inner}</m:oMath></m:oMathPara>'))
    return p


def eq_tfidf_weight():
    frac = mfrac(mr("1 + N"), mr("1 + ") + msub(mr("df", True), mr("t")))
    return (msub(mr("w"), mr("t,d")) + mr(" = ") + msub(mr("tf", True), mr("t,d")) + mr(" · ")
            + mparen(mfunc("ln", mparen(frac)) + mr(" + 1")))


def eq_cosine():
    norms = mparen(mr("q"), "‖", "‖") + mparen(mr("d"), "‖", "‖")
    return mfunc("cos", mparen(mr("q, d"))) + mr(" = ") + mfrac(mr("q · d"), norms)


def eq_bm25():
    k1 = msub(mr("k"), mr("1"))
    f = msub(mr("f"), mr("t,d"))
    body = (mr("idf", True) + mparen(mr("t")) + mr(" · ")
            + mfrac(f + mr(" (") + k1 + mr(" + 1)"),
                    f + mr(" + ") + k1 + mparen(mr("1 − b + b ") + mfrac(mr("|d|"), mr("avgdl", True)))))
    return mr("BM25", True) + mparen(mr("q, d")) + mr(" = ") + mnary(mr("t ∈ q"), "", body)


def eq_rrf():
    body = mfrac(mr("1"), mr("k + ") + msub(mr("rank", True), mr("m")) + mparen(mr("d")))
    return (mr("RRF", True) + mparen(mr("d")) + mr(" = ") + mnary(mr("m ∈ M"), "", body)
            + mr(",     k = 60"))


# -------------------------------------------------------------- data loading

def load_csv(name):
    with open(os.path.join(REPORTS, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f4(x):
    return f"{float(x):.4f}"


def latency_from_log():
    """Fit and per-query times from the 2026-09-27 evaluation run's timestamped log."""
    stamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.*)$")
    events = []
    with open(LATENCY_LOG, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = stamp.match(line.rstrip("\n"))
            if m:
                events.append((dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f"), m.group(2)))

    def at(prefix):
        return next(t for t, msg in events if msg.startswith(prefix))

    sec = lambda a, b: (b - a).total_seconds()  # noqa: E731
    out = {
        "fit_tfidf": sec(at("Fitting TF-IDF"), at("Fitting BM25")),
        "fit_bm25": sec(at("Fitting BM25"), at("Fitting Dense")),
        "fit_dense": sec(at("Fitting Dense"), at("Fitting Hybrid RRF")),
        "startup": sec(at("Loading datasets"), at("Fitting Hybrid RRF")),
    }
    for name in ("TF-IDF", "BM25", "Dense", "Hybrid-RRF"):
        out[name] = sec(at(f"Evaluating {name}"), at(f"{name} progress: 400/400")) / 400
    return out


def rrf_worked_example():
    """Five documents, two illustrative rank lists, scored by the project's own RRF function."""
    bm25 = [("A", 0), ("B", 0), ("C", 0), ("x", 0), ("E", 0)]
    dense = [("B", 0), ("D", 0), ("A", 0), ("y", 0), ("E", 0)]
    scores = dict(reciprocal_rank_fusion([bm25, dense], 10, 60))
    rank = lambda lst, d: next((i for i, (x, _) in enumerate(lst, 1) if x == d), None)  # noqa: E731
    rows = []
    for d in "ABCDE":
        rb, rd = rank(bm25, d), rank(dense, d)
        rows.append((d, rb, rd, scores[d]))
    order = sorted(rows, key=lambda r: -r[3])
    final_rank = {r[0]: i for i, r in enumerate(order, 1)}
    return [[d, str(rb) if rb else "-", str(rd) if rd else "-", f"{s:.4f}", str(final_rank[d])]
            for d, rb, rd, s in rows]


# ------------------------------------------------------------------- content

def build_title_block(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("Duplicate Question Retrieval on Quora Question Pairs: Sparse, Dense and "
                    "Rank-Fusion Retrieval with Short-Text Vector Indexing")
    run.bold = True
    run.font.size = Pt(20)

    students = doc.add_table(5, 3)
    students.style = doc.styles["Table Grid"]
    students.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_col_widths(students, [1.85, 1.7, 2.95])
    for c, h in enumerate(["Students Name", "Registration No.", "Email-id"]):
        cell = students.rows[0].cells[c]
        cell.paragraphs[0].add_run(h).bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(14)
        cell.paragraphs[0].paragraph_format.space_after = Pt(2)
    for row in students.rows:
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(parse_xml(f'<w:trHeight {nsdecls("w")} w:val="380" w:hRule="atLeast"/>'))
        for cell in row.cells:
            cell._tc.get_or_add_tcPr().append(parse_xml(
                f'<w:vAlign {nsdecls("w")} w:val="center"/>'))
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            cell.paragraphs[0].paragraph_format.space_before = Pt(0)
    set_cell_margins(students, top_bottom_pt=2, left_right_pt=6)
    gap = doc.add_paragraph()
    gap.paragraph_format.space_after = Pt(0)
    gap.paragraph_format.line_spacing = Pt(6)

    cell = boxed(doc, ABSTRACT_FILL, left_rule=True)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = BODY_AFTER
    p.paragraph_format.line_spacing = BODY_LINE
    add_runs(p, (
        "**Abstract — **This report treats duplicate-question detection as a search task on Quora "
        "Question Pairs. The collection has 537,929 unique questions, the answer key (qrels) has "
        "298,526 duplicate judgements, and 400 randomly chosen questions are the test queries. We "
        "compare four methods: TF-IDF, BM25, a dense retriever built on a Sentence Transformer, and "
        "a hybrid that merges the BM25 and dense rankings with Reciprocal Rank Fusion (RRF). "
        "Standard text cleaning shrank some short questions to a single word, which gave many tied "
        "cosine scores of 1.0. A dual-space indexing scheme, which merges light, heavy and "
        "character-level rankings, removes these ties. Dense retrieval scores highest on all four "
        "metrics (nDCG@10 0.9431, MAP@10 0.9087), the hybrid is second (0.9251), and BM25 (0.7755) "
        "beats TF-IDF (0.7314). We made design choices by looking at these same queries, so the "
        "figures are development-set figures and are probably optimistic."
    ))
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = BULLET_AFTER
    p2.paragraph_format.line_spacing = BODY_LINE
    add_runs(p2, (
        "**Keywords: ***Information Retrieval · Duplicate Question Detection · Short-Text Vector "
        "Indexing · BM25 · Sentence Transformers · Reciprocal Rank Fusion · Quora Question Pairs*"
    ))


def build_introduction(doc):
    heading(doc, "1. Introduction")
    para(doc, (
        "Question-answering sites collect many versions of the same question, which splits the "
        "answers across several threads. Finding a repeat is a search problem: the new question is "
        "the query, the earlier questions are the collection, and a document is relevant if "
        "annotators labelled it a duplicate. This differs from ordinary web search because queries "
        "and documents are both single short sentences, and relevance means the same intent, not "
        "just the same topic. For example, “Which smartphone should I buy under INR 15k?” and "
        "“…under INR 40000?” share almost every word but are not duplicates."
    ))
    para(doc, (
        "*Sparse lexical retrieval* scores each document by how many weighted words it shares with "
        "the query; TF-IDF [5] and BM25 [1] are the standard methods. It cannot see a paraphrase "
        "that shares few words, and text cleaning can remove most of a very short question. "
        "*Dense retrieval* turns each question into a vector using a bi-encoder such as "
        "Sentence-BERT [2] and ranks by vector similarity. It handles re-worded questions well, but "
        "needs an encoding step and can miss rare exact words. *Hybrid retrieval* merges the two "
        "rankings. Reciprocal Rank Fusion (RRF) [3] does this using ranks alone, so the scores "
        "need no adjusting."
    ))
    para(doc, (
        "This report turns QQP [4] into a search benchmark, compares four rankers on four metrics, "
        "describes a dual-space indexing scheme for short text that evaluation by query length "
        "showed to be necessary, and provides a Streamlit search interface."
    ))


def build_dataset(doc):
    heading(doc, "2. About Dataset")
    para(doc, (
        "QQP [4] is a set of question pairs, each with a yes/no label `is_duplicate` assigned by "
        "people. The file we used has 404,290 pairs, of which 149,263 are duplicates. It has no "
        "ready-made collection or relevance judgements, so we built them (counts in Table 1):"
    ))
    bullet(doc, "**Corpus:** every distinct question, 537,929 in all.")
    bullet(doc, (
        "**Qrels:** each duplicate pair is stored in both directions, giving 298,526 (query, "
        "relevant document) rows over 149,650 query IDs. Non-duplicate pairs add no rows; those "
        "questions stay in the collection as distractors. A true duplicate that nobody labelled "
        "counts as not relevant."
    ))
    bullet(doc, (
        "**Evaluation queries:** 400 questions picked at random (seed 42) from those that have a "
        "qrel; 40 have at most five words and 360 have more. They stay in the collection; only a "
        "document whose text is identical to the query is removed from its own results."
    ))
    caption(doc, "Table 1. Dataset statistics.")
    data_table(doc, ["Quantity", "Value", "Quantity", "Value"], [
        ["Raw question pairs", "404,290", "Query IDs with a qrel", "149,650"],
        ["Labelled duplicates", "149,263", "Evaluation queries", "400"],
        ["Corpus (unique questions)", "537,929", "Short / long (≤5 / >5 words)", "40 / 360"],
        ["Qrel rows", "298,526", "Dense index (questions × dims)", "100,000 × 384"],
    ], [1.9, 0.95, 2.25, 1.4], left_cols=(0, 2), size=9)
    subheading(doc, "2.1 Dataset visualisation and examples")
    para(doc, (
        "Questions are short (Figure 1a): the median is 10 words, and 8.2% have at most five. "
        "This is where text cleaning can erase most of the question (Section 3.4). Most evaluation "
        "queries have just one relevant document (294 of 400; mean 2.19, maximum 33; Figure 1b), "
        "which limits Precision@10 (Section 4). In Table 2, pair 1 is a paraphrase with little "
        "word overlap, pair 2 is a three-word question, and pair 3, which is not a duplicate, "
        "shares almost every word."
    ), keep_next=True)
    caption(doc, "Table 2. Example pairs from the QQP file.")
    data_table(doc, ["#", "Question 1", "Question 2", "Dup."], [
        ["1", "How do I speak English fluently?",
         "How can I take my spoken English to a higher level when I speak good enough but not fluently?", "Yes"],
        ["2", "The loneliest number?", "What is the loneliest number?", "Yes"],
        ["3", "Which smartphone should I buy under INR 15k?",
         "Which smartphone should I buy under INR 40000?", "No"],
    ], [0.3, 2.3, 3.3, 0.6], left_cols=(1, 2), size=9)
    picture(doc, "fig_dataset.png", 6.5, (
        "Figure 1: (a) Question length in the corpus, questions of at most five words in dark. "
        "(b) Relevant documents per evaluation query (log scale)."
    ))


def design_rationale_rows():
    return [
        ["TF-IDF", "Spec", "scikit-learn TfidfVectorizer, cosine",
         "Own inverted index (written, unused)",
         "Specified. Sparse matrices give exact cosine over every question."],
        ["BM25", "Spec", "rank_bm25 BM25Okapi, defaults",
         "Tuned parameters, other variants",
         "Specified. A maintained library saves us rewriting the scoring."],
        ["Dense encoder", "Spec + ours", "all-MiniLM-L6-v2, 384-d", "Other encoders",
         "Encoder type was specified; this model is small, free and runs on a CPU."],
        ["Vector index", "Ours", "FAISS IndexFlatIP on unit vectors", "Approximate indexes",
         "Exact cosine, no approximation error; 100,000 × 384 numbers take about 150 MB."],
        ["Fusion", "Spec", "RRF on ranks, k = 60", "Weighted fusion, learned re-ranker, other k",
         "Ranks need no score adjusting between BM25 and cosine; a learned re-ranker needs labels, and our only labels are the ones used for testing."],
        ["Preprocessing", "Spec + ours", "NLTK, Porter; light and heavy modes",
         "Other stemmers, lemmatisers",
         "Steps were specified; heavy mode alone shrank short questions too much (Section 3.4)."],
        ["Interface", "Ours", "Streamlit, three tabs", "Other front ends",
         "A Python app that gives search, a comparison table and a PR graph."],
        ["Metrics", "Spec", "P@10, Recall, MAP, nDCG at 10", "-",
         "Specified; P@10 has a low maximum (Section 4), so the other three metrics carry the comparison."],
    ]


def build_methodology(doc):
    heading(doc, "3. Methodology")
    para(doc, (
        "All four rankers use the same interface and none is trained: the qrels are used only for "
        "scoring. Figure 2 shows the pipeline. The Search tab runs the retrievers live, while the "
        "comparison and precision-recall tabs read the saved evaluation results."
    ))
    picture(doc, "fig_pipeline.png", 6.5, (
        "Figure 2: Pipeline. TF-IDF and BM25 each merge their internal spaces with RRF; Hybrid-RRF "
        "merges the BM25 and Dense lists (TF-IDF is not part of it)."
    ))

    subheading(doc, "3.1 Preprocessing and Indexing")
    para(doc, (
        "Preprocessing makes the text lowercase, replaces punctuation with spaces, and splits it "
        "into words with NLTK `word_tokenize`. *Heavy* mode then removes common English "
        "stop-words and shortens words with the Porter stemmer [6]; *light* mode stops after "
        "splitting. A text counts as short if it has at most five words separated by spaces. "
        "TF-IDF and BM25 are built once over all 537,929 questions. TF-IDF keeps sparse "
        "document-term matrices (scikit-learn) and scores a query against every row. BM25 uses "
        "`rank_bm25`, which goes through every document's word counts for each query word. "
        "Neither uses an inverted index at query time. The dense retriever encodes the raw "
        "question into a 384-number unit vector stored in a FAISS `IndexFlatIP`. To keep encoding "
        "manageable, the dense index holds only 100,000 questions: the 400 queries and their "
        "relevant documents (1,276 questions in the run log) plus a random fill chosen with a "
        "fixed seed."
    ))

    subheading(doc, "3.2 Ranking Models")
    para(doc, "Table 3 lists the design choices; the four models are described below.", keep_next=True)
    caption(doc, "Table 3. Design rationale. Spec = specified by the assignment; Ours = our own choice. None of the listed alternatives was evaluated.")
    data_table(doc, ["Component", "Origin", "Chosen", "Alternatives", "Why"],
               design_rationale_rows(), [1.05, 0.6, 1.25, 1.15, 2.45], left_cols=(0, 1, 2, 3, 4), size=8.5, together=False)

    para(doc, (
        "**TF-IDF** [5] turns a text into a vector over words, using the weight below. Here tf is "
        "the word count, N is the number of questions, and df~t~ is the number of questions that "
        "contain word t (scikit-learn's smoothed form). Each vector is scaled to length 1 "
        "(L2-normalised), so similarity is the cosine. Rare words get more weight, which helps when "
        "a duplicate shares an unusual word."
    ), keep_next=True)
    left, right = side_by_side(doc, 4.1, 2.4)
    equation(left, eq_tfidf_weight(), after=Pt(0))
    equation(right, eq_cosine(), after=Pt(0))
    for cell in (left, right):
        drop_leading_empty(cell)
        shrink_trailing_empty(cell)
    para(doc, (
        "**BM25** [1] scores a document with the formula below. Here idf(t) = ln((N − n~t~ + "
        "0.5)/(n~t~ + 0.5)), f~t,d~ is the word count, |d| is the document length and avgdl is "
        "the average length. k~1~ limits the gain from repeated words, and b controls how much "
        "length is penalised. We use `rank_bm25`'s BM25Okapi with its default values k~1~ = 1.5 "
        "and b = 0.75 (not tuned); the library replaces a negative idf with 0.25 times the mean "
        "idf. BM25 scores have no upper limit, so they cannot be added to cosine scores."
    ), keep_next=True)
    equation(doc, eq_bm25())
    para(doc, (
        "**Dense retrieval** turns each question into a unit vector f(x) with all-MiniLM-L6-v2 "
        "[2, 8] and ranks by cosine, sim(q, d) = f(q)·f(d), using exact FAISS search [7]. Because "
        "it compares meaning instead of words, it can match a re-worded duplicate. "
        "**Hybrid-RRF** applies RRF (Section 3.3) to two lists of 100 documents: the output of "
        "the BM25 ensemble and the dense list. TF-IDF is not included. The BM25 list is itself an "
        "RRF merge of its light and heavy spaces, so the hybrid is RRF over an RRF-merged list, "
        "and its dense side searches the same 100,000-question index."
    ))

    subheading(doc, "3.3 Reciprocal Rank Fusion Formulation")
    para(doc, (
        "RRF gives each document a score by adding up, over every ranked list M that contains it, "
        "one divided by (k plus its rank). Here rank~m~(d) is the position of d in list m, counting "
        "from 1; a list that does not contain d adds nothing:"
    ), keep_next=True)
    equation(doc, eq_rrf())
    para(doc, (
        "Only positions are used, so lists from different scorers need no adjusting. The constant "
        "k = 60 (the value given in the assignment, also used in [3]) stops one first place from "
        "outweighing agreement between lists. Table 4 scores example ranks with the project's own "
        "function: document E, at rank 5 in both lists, beats D, at rank 2 in only one. The same "
        "function does the merging inside TF-IDF, inside BM25 and in the hybrid."
    ), keep_next=True)
    caption(doc, "Table 4. RRF worked example (example ranks, k = 60).")
    data_table(doc, ["Doc", "BM25 rank", "Dense rank", "RRF score", "Fused rank"], rrf_worked_example(),
               [0.8, 1.0, 1.0, 1.0, 1.0], left_cols=(), size=9)

    subheading(doc, "3.4 Short-Text Vector Indexing")
    para(doc, (
        "Short-text vector indexing means building representations that still carry enough "
        "information when a question has only a few words. Heavy preprocessing can shrink a short "
        "question to one word or none: “What is the google?” becomes [googl] (Figure 3, top lane), "
        "and six of the 40 short evaluation queries shrank to zero or one word. Many documents "
        "were reduced to the same one-dimensional vector, so different questions tied at cosine "
        "1.0 and their order was arbitrary."
    ))
    para(doc, (
        "We tried two designs and rejected both, based on evaluation split by query length "
        "(Table 7 and Figure 4, Section 4). *Global light mode* (no stop-word removal or stemming "
        "for any text) removed the ties but lowered TF-IDF on long queries below its original "
        "level, even though long queries had not been affected by the problem. *Per-text mode "
        "selection* (light for short texts, heavy for long ones, inside one shared vocabulary) "
        "fixed long queries but broke the vocabulary: a short query kept “geographic” while its "
        "longer duplicate was stemmed to “geograph”, and bigrams (word pairs) existed only on "
        "the query side."
    ))
    para(doc, (
        "The design we adopted is *dual-space indexing with rank fusion*. Every question is "
        "indexed in separate spaces over the whole collection: light (plain words plus word "
        "pairs), heavy (stemmed single words, stop-words removed) and, for TF-IDF only, groups of "
        "3 to 5 characters within words, used only for queries of at most five words. A query is "
        "processed in each space in exactly the same way as that space's documents. Each space "
        "returns up to 100 documents, and the lists are merged with the RRF formula above "
        "(Figure 3). BM25 uses its light and heavy spaces for every query. Inside one space, "
        "query and documents are split into words in the same way, which removes the mismatch. "
        "The five-word threshold was chosen by looking at the data, not tuned."
    ))
    picture(doc, "fig_dual_space.png", 5.2, (
        "Figure 3: Dual-space indexing for the TF-IDF query “What is the google?”; BM25 uses the "
        "first two lanes."
    ))


def build_experiments(doc, comparison, segmented, pr, lat):
    heading(doc, "4. Experimental Setup and Performance Analysis")
    para(doc, (
        "**Setup and metrics.** The 400 evaluation queries are run through all four rankers, "
        "looking at the top 10 results (top 50 for the PR curves). Nothing is trained on the "
        "qrels. The queries stay in the collection, and only a document with identical text is "
        "removed. The short-text design (Section 3.4) was chosen by looking at these same queries, "
        "with no separate development split, so all figures are development-set figures and are "
        "probably optimistic. In Table 5, R is the set of relevant documents for the query, T~10~ "
        "is its top ten, and rel~i~ ∈ {0, 1} says whether the document at rank i is relevant. AP "
        "is divided by |R| instead of min(10, |R|), which gives a lower value than the usual "
        "convention for the 13 queries with more than ten relevant documents."
    ))
    ceiling = 0.184
    caption(doc, "Table 5. Metric definitions as implemented (cutoff 10).")
    data_table(doc, ["Metric", "Definition"], [
        ["P@10", "|R ∩ T~10~| / 10"],
        ["Recall@10", "|R ∩ T~10~| / |R|"],
        ["AP@10 (MAP@10 = mean over queries)", "(1 / |R|) · Σ~i=1..10~ P@i · rel~i~"],
        ["nDCG@10 [9]", "DCG@10 / IDCG@10, with DCG@10 = Σ~i=1..10~ rel~i~ / log~2~(i + 1); IDCG@10 is the DCG@10 of "
                    "the ideal ranking (min(10, |R|) relevant first)"],
    ], [2.3, 4.2], left_cols=(0, 1), size=9)
    para(doc, (
        f"**Precision ceiling.** The qrels limit Precision@10. A perfect ranker scores "
        f"min(|R|, 10)/10 on each query, and the mean of this over the 400 queries is "
        f"{ceiling:.3f}. The looser bound 2.19/10 ≈ 0.22 ignores that 13 queries have more than "
        f"ten relevant documents. Dense retrieval's 0.1738 is {0.1738 / ceiling * 100:.0f}% of the "
        f"exact ceiling."
    ))

    caption(doc, "Table 6. Retrieval performance on the 400 evaluation queries (cutoff 10). Best per column in bold.")
    data_table(doc, ["Ranking Model", "P@10", "Recall@10", "MAP@10", "nDCG@10"], [
        [r["Method"], f4(r["P@10"]), f4(r["Recall@10"]), f4(r["MAP@10"]), f4(r["nDCG@10"])]
        for r in comparison
    ], [2.0, 1.1, 1.1, 1.1, 1.1], bold_best_cols=(1, 2, 3, 4), size=9.5)
    last = {m: pr[m]["recall"][49] for m in pr}
    first = {m: pr[m]["precision"][0] for m in pr}
    para(doc, (
        "**Overall results** (Table 6, Figure 5a; precision-recall curves in Figure 5b). Dense "
        "retrieval is best on every metric (nDCG@10 0.9431, MAP@10 0.9087, Recall@10 0.9659). "
        "Hybrid-RRF is second on every metric (nDCG@10 0.9251), and BM25 beats TF-IDF on all four "
        "(nDCG@10 0.7755 against 0.7314). At k = 1 precision is "
        f"{first['Dense']:.4f} (Dense), {first['Hybrid-RRF']:.4f} (Hybrid-RRF), {first['BM25']:.4f} "
        f"(BM25) and {first['TF-IDF']:.4f} (TF-IDF), so Dense puts a duplicate first for about "
        "nine of ten queries. In Figure 5b the order Dense, Hybrid-RRF, BM25, TF-IDF holds for "
        "both precision and recall at every k from 1 to 50, and the k = 10 markers match the "
        "P@10 and Recall@10 columns of Table 6. At k = 50 recall is "
        f"{last['Dense']:.3f} (Dense) but only {last['BM25']:.3f} (BM25) and {last['TF-IDF']:.3f} (TF-IDF)."
    ))
    para(doc, (
        "A likely reason for Dense's lead is that duplicates are paraphrases and the encoder "
        "captures meaning; no ablation tested this. Dense also searches an index 5.4 times "
        "smaller (100,000 against 537,929 questions), which favours it and the hybrid by an "
        "unmeasured amount. Hybrid-RRF is below Dense, perhaps because equal-weight RRF lets the "
        "weaker BM25 list pull down the dense list; this was not tested."
    ))

    para(doc, (
        "**Segmented analysis.** Table 7 splits the lexical models by query length; the TF-IDF and "
        "BM25 rows of Table 6 are the query-weighted averages of these rows. TF-IDF does better on "
        "short queries (nDCG@10 0.7908) than on long ones (0.7247), while BM25 does the opposite "
        "(0.7314 short, 0.7804 long). We did not test why, and with only 40 short queries the "
        "difference may be noise. Figure 4 follows TF-IDF nDCG@10 through the designs of Section "
        "3.4; the original figure is inflated by the 1.0 ties. The per-text design's overall "
        "score (0.6784) hid a collapse to 0.3508 on short queries. The final design beats the "
        "per-text design on both groups; its overall score of 0.7314 is within 0.004 of the "
        "original 0.7274, but without the 1.0 ties behind it."
    ))
    caption(doc, "Table 7. Lexical models by query length (cutoff 10).")
    data_table(doc, ["Model", "Group", "n", "P@10", "Recall@10", "MAP@10", "nDCG@10"], [
        [r["method"], r["group"].capitalize(), r["query_count"], f4(r["precision_at_10"]),
         f4(r["recall_at_10"]), f4(r["map_at_10"]), f4(r["ndcg_at_10"])]
        for r in segmented
    ], [1.0, 0.9, 0.6, 0.9, 1.0, 1.0, 1.1], left_cols=(0, 1), size=9)
    picture(doc, "fig_results.png", 6.5, (
        "Figure 5: (a) The four metrics for each method. (b) Precision-recall curves for k = 1 to "
        "50, averaged over the 400 queries; large markers show k = 10."
    ))

    picture(doc, "fig_indexing_designs.png", 6.5,
            "Figure 4: TF-IDF nDCG@10 by indexing design and query group; missing bars were not recorded.")


def build_conclusion(doc, lat):
    heading(doc, "5. Conclusion")
    para(doc, (
        "On the 400 evaluation queries, similarity of meaning was the strongest single signal: "
        "Dense led on every metric, Hybrid-RRF came within 0.0180 nDCG@10 of it (0.9251 against "
        "0.9431) while keeping an exact-match component, and BM25 beat TF-IDF. Dual-space indexing "
        "removed the 1.0 ties and gave TF-IDF a short-query nDCG@10 of 0.7908 (0.3508 under the "
        "per-text design)."
    ))
    para(doc, "**Limitations.**", after=Pt(1), keep_next=True)
    bullet(doc, (
        "**Development-set figures.** We made design choices by looking at these 400 queries, "
        "with no separate development split, so the figures are probably optimistic. The rankers "
        "are unsupervised: nothing is trained on the qrels."
    ))
    bullet(doc, (
        "**Unequal pools.** Dense, and the dense side of the hybrid, search 100,000 questions; "
        "TF-IDF and BM25 search 537,929. This favours Dense and Hybrid-RRF by an unmeasured amount."
    ))
    bullet(doc, (
        "**Untuned and unswept.** BM25 uses library defaults. We did not test other values of "
        "k (60) or of the five-word threshold. The analysis by query length covered only the "
        "lexical models, and the short group has just 40 queries."
    ))
    bullet(doc, (
        "**Encoder training data.** The model card of all-MiniLM-L6-v2 lists Quora Question "
        "Triplets (103,663) among its fine-tuning data, so Dense may have seen related pairs; we "
        "did not measure the overlap with our queries."
    ))
    bullet(doc, (
        "**Labels and cutoff.** True duplicates that nobody labelled count as not relevant, and "
        "13 queries have more than ten relevant documents, so Recall@10 cannot reach 1 for them."
    ))
    bullet(doc, (
        f"**Speed.** Approximate figures from the log of one CPU run (2026-09-27; Dense used saved "
        f"embeddings; per-query times include ranking the top 50): fitting took "
        f"{lat['fit_tfidf']:.0f} s for TF-IDF, {lat['fit_bm25']:.0f} s for BM25 and "
        f"{lat['fit_dense']:.1f} s for Dense, and the mean time per query was "
        f"{lat['TF-IDF']:.2f} s (TF-IDF), {lat['BM25']:.2f} s (BM25), {lat['Dense']:.3f} s (Dense) "
        f"and {lat['Hybrid-RRF']:.2f} s (Hybrid-RRF); start-up took "
        f"{lat['startup'] / 60:.1f} minutes, which is slow for interactive use."
    ))
    para(doc, (
        "**Future work:** a separate development split, equal-sized pools, tests of other values "
        "of k, a re-ranker and tuned BM25."
    ))


REFERENCES = [
    "Robertson, S., Zaragoza, H.: The Probabilistic Relevance Framework: BM25 and Beyond. "
    "Found. Trends Inf. Retr. 3(4), 333–389 (2009)",
    "Reimers, N., Gurevych, I.: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. "
    "In: Proceedings of EMNLP-IJCNLP, pp. 3982–3992. ACL (2019)",
    "Cormack, G.V., Clarke, C.L.A., Buettcher, S.: Reciprocal Rank Fusion Outperforms Condorcet "
    "and Individual Rank Learning Methods. In: Proceedings of SIGIR, pp. 758–759. ACM (2009)",
    "Iyer, S., Dandekar, N., Csernai, K.: First Quora Dataset Release: Question Pairs. Quora (2017)",
    "Salton, G., Buckley, C.: Term-Weighting Approaches in Automatic Text Retrieval. Information "
    "Processing & Management 24(5), 513–523 (1988)",
    "Porter, M.F.: An Algorithm for Suffix Stripping. Program 14(3), 130–137 (1980)",
    "Johnson, J., Douze, M., Jégou, H.: Billion-Scale Similarity Search with GPUs. IEEE "
    "Transactions on Big Data 7(3), 535–547 (2021)",
    "Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., Zhou, M.: MiniLM: Deep Self-Attention "
    "Distillation for Task-Agnostic Compression of Pre-Trained Transformers. In: NeurIPS 33, "
    "pp. 5776–5788 (2020)",
    "Järvelin, K., Kekäläinen, J.: Cumulated Gain-Based Evaluation of IR Techniques. ACM "
    "TOIS 20(4), 422–446 (2002)",
]


def build_references(doc):
    heading(doc, "References")
    for i, ref in enumerate(REFERENCES, start=1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = BULLET_AFTER
        p.paragraph_format.line_spacing = BODY_LINE
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        run = p.add_run(f"{i}. {ref}")
        run.font.size = Pt(10)


def body_word_count(path):
    """Words in top-level paragraphs from '1. Introduction' to 'References', minus captions."""
    doc = Document(path)
    count, inside = 0, False
    for p in doc.paragraphs:
        text = p.text.strip()
        if text.startswith("1. Introduction"):
            inside = True
        if text == "References":
            break
        if inside and text and not re.match(r"^(Table|Figure) \d", text):
            count += len(text.split())
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    text = p.text.strip()
                    if len(text.split()) > 40 and not re.match(r"^(Table|Figure) \d", text) \
                            and not text.startswith("Abstract"):
                        count += len(text.split())
    return count


def main():
    make_figures.main()
    comparison = load_csv("comparison_table.csv")
    segmented = load_csv("segmented_metrics.csv")
    with open(os.path.join(REPORTS, "pr_curve_data.json"), encoding="utf-8") as f:
        pr = json.load(f)
    lat = latency_from_log()

    doc = fresh_document_from_format()
    build_title_block(doc)
    build_introduction(doc)
    build_dataset(doc)
    build_methodology(doc)
    build_experiments(doc, comparison, segmented, pr, lat)
    build_conclusion(doc, lat)
    build_references(doc)
    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")
    print("Body words (sections 1-5, excl. captions/tables):", body_word_count(OUTPUT))


if __name__ == "__main__":
    main()

