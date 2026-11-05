"""Builds the Project 8 presentation deck from the project's own final results.

Reads only real numbers already written to reports/project_summary_for_writeup.md
(and the artifacts it cites: comparison_table.csv, pr_curve.png). Does not
invent figures; if an expected input (e.g. a UI screenshot) is missing, the
slide says so instead of fabricating one.
"""

import csv
import os

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
OUTPUT_PATH = os.path.join(REPORTS_DIR, "project8_presentation.pptx")
PR_CURVE_PATH = os.path.join(REPORTS_DIR, "pr_curve.png")
COMPARISON_CSV_PATH = os.path.join(REPORTS_DIR, "comparison_table.csv")

TEAM_PLACEHOLDER = "Your Name / Team Name"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

NAVY = RGBColor(0x0D, 0x16, 0x26)
INK = RGBColor(0x1A, 0x22, 0x33)
ACCENT_TEAL = RGBColor(0x2D, 0xD4, 0xBF)   # TF-IDF
ACCENT_BLUE = RGBColor(0x60, 0xA5, 0xFA)   # BM25
ACCENT_PURPLE = RGBColor(0xC0, 0x84, 0xFC) # Dense
ACCENT_GOLD = RGBColor(0xFB, 0xBF, 0x24)   # Hybrid-RRF
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x94, 0xA3, 0xB8)
LIGHT_BG = RGBColor(0xF4, 0xF6, 0xFA)

METHOD_COLORS = {
    "TF-IDF": ACCENT_TEAL,
    "BM25": ACCENT_BLUE,
    "Dense": ACCENT_PURPLE,
    "Hybrid-RRF": ACCENT_GOLD,
}


def load_comparison_table() -> list[dict]:
    """Reads the final held-out comparison table, failing loudly if absent."""
    if not os.path.exists(COMPARISON_CSV_PATH):
        raise FileNotFoundError(
            f"{COMPARISON_CSV_PATH} not found; run src/evaluation/run_evaluation.py first."
        )
    with open(COMPARISON_CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_background(slide, color=NAVY):
    """Fills the slide background with a solid color."""
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def add_textbox(slide, left, top, width, height, text, size=18, bold=False,
                 color=WHITE, align=PP_ALIGN.LEFT, font="Calibri"):
    """Adds a single-paragraph textbox and returns it."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return box


def add_bullets(slide, left, top, width, height, items, size=18,
                 color=WHITE, line_spacing=1.25, font="Calibri"):
    """Adds a bulleted list as one textbox with one paragraph per item."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = line_spacing
        p.space_after = Pt(10)
        run = p.add_run()
        run.text = f"•  {item}"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = font
    return box


def add_slide_title(slide, title, subtitle=None):
    """Adds a consistent title (and optional subtitle) bar to a content slide."""
    add_textbox(slide, Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.7),
                title, size=30, bold=True, color=WHITE)
    if subtitle:
        add_textbox(slide, Inches(0.5), Inches(0.95), Inches(12.3), Inches(0.5),
                    subtitle, size=16, color=GREY)


def add_box(slide, left, top, width, height, label, fill_color, text_color=WHITE,
            size=14, shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
    """Adds a filled shape with centered label text and returns the shape."""
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = fill_color
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = text_color
    return shape


def add_arrow(slide, start_shape, end_shape):
    """Connects two shapes left-to-right with a straight arrow connector."""
    connector = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        start_shape.left + start_shape.width, start_shape.top + start_shape.height // 2,
        end_shape.left, end_shape.top + end_shape.height // 2,
    )
    connector.line.color.rgb = GREY
    connector.line.width = Pt(2)


def build_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_textbox(slide, Inches(1), Inches(2.5), Inches(11.3), Inches(1.3),
                "Project 8: Duplicate Question Retrieval",
                size=40, bold=True, color=WHITE)
    add_textbox(slide, Inches(1), Inches(3.6), Inches(11.3), Inches(0.6),
                "Semantic Duplicate Detection over the Quora Question Pairs Dataset",
                size=20, color=ACCENT_GOLD)
    add_textbox(slide, Inches(1), Inches(4.3), Inches(11.3), Inches(0.5),
                "Short-Text Vector Indexing • TF-IDF • BM25 • Dense • Hybrid-RRF",
                size=16, color=GREY)
    add_textbox(slide, Inches(1), Inches(6.5), Inches(11.3), Inches(0.5),
                TEAM_PLACEHOLDER, size=16, color=GREY)
    return slide


def build_problem_statement_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "Problem Statement")
    add_bullets(slide, Inches(0.6), Inches(1.6), Inches(12.0), Inches(1.6), [
        "Goal: given a new question, retrieve semantically duplicate questions "
        "already asked on Quora — reframed from pairwise duplicate "
        "classification into a retrieval task.",
    ], size=18)
    add_textbox(slide, Inches(0.6), Inches(3.1), Inches(5.0), Inches(0.4),
                "Dataset: Quora Question Pairs", size=18, bold=True, color=ACCENT_GOLD)
    add_bullets(slide, Inches(0.8), Inches(3.6), Inches(11.5), Inches(3.0), [
        "404,290 raw question pairs; 149,263 labeled is_duplicate=1",
        "537,929 unique questions in the deduplicated retrieval corpus",
        "298,526 qrel (query_id, doc_id) pairs covering 149,650 unique query IDs",
        "400 held-out evaluation queries (40 short ≤ 5 words / 360 long)",
    ], size=18)
    return slide


def build_architecture_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "System Architecture")

    stages = [
        ("Preprocessing", ACCENT_TEAL),
        ("Dual-Space\nTF-IDF / BM25", ACCENT_BLUE),
        ("Dense\n(Sentence\nTransformer)", ACCENT_PURPLE),
        ("Hybrid-RRF\nFusion", ACCENT_GOLD),
        ("Evaluation", GREY),
        ("UI", WHITE),
    ]
    n = len(stages)
    box_w = Inches(1.75)
    box_h = Inches(1.3)
    gap = Inches(0.35)
    total_w = box_w * n + gap * (n - 1)
    start_left = Emu(int((SLIDE_W - total_w) / 2))
    top = Inches(2.9)

    boxes = []
    for i, (label, color) in enumerate(stages):
        left = Emu(int(start_left + i * (box_w + gap)))
        text_color = INK if color in (GREY, WHITE) else WHITE
        box = add_box(slide, left, top, box_w, box_h, label, color, text_color=text_color, size=13)
        boxes.append(box)
    for a, b in zip(boxes, boxes[1:]):
        add_arrow(slide, a, b)

    add_bullets(slide, Inches(0.8), Inches(4.8), Inches(11.5), Inches(2.0), [
        "Preprocessing: lowercase, strip punctuation, NLTK tokenize, stopword "
        "removal + Porter stemming — skipped for short text (≤ 5 words) to "
        "avoid token collapse.",
        "TF-IDF and BM25 each fuse light (surface-form) and heavy (stemmed) "
        "corpus-wide spaces via RRF; TF-IDF adds a character n-gram space for "
        "short queries.",
        "Dense: all-MiniLM-L6-v2 embeddings, FAISS exact search. "
        "Hybrid-RRF fuses BM25 with Dense.",
    ], size=15)
    return slide


def build_methodology_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "Methodology: 4 Ranking Models")

    models = [
        ("TF-IDF", "Dual-space (light + heavy + char n-gram) cosine similarity, fused with RRF", ACCENT_TEAL),
        ("BM25", "Dual-space (light + heavy) Okapi BM25, fused with RRF", ACCENT_BLUE),
        ("Dense", "Sentence-transformer embeddings (all-MiniLM-L6-v2) + FAISS inner-product search", ACCENT_PURPLE),
        ("Hybrid-RRF", "Reciprocal Rank Fusion of BM25 + Dense ranked lists", ACCENT_GOLD),
    ]
    top = Inches(1.6)
    row_h = Inches(0.95)
    for name, desc, color in models:
        add_box(slide, Inches(0.6), top, Inches(2.0), Inches(0.7), name, color, size=15)
        add_textbox(slide, Inches(2.9), top, Inches(9.6), Inches(0.8), desc, size=15, color=WHITE)
        top = Emu(int(top + row_h))

    add_textbox(slide, Inches(0.6), Inches(5.9), Inches(12.0), Inches(0.4),
                "Reciprocal Rank Fusion (RRF) formula:", size=16, bold=True, color=ACCENT_GOLD)
    add_textbox(slide, Inches(0.6), Inches(6.35), Inches(12.0), Inches(0.6),
                "RRF_Score(d) = Σ  1 / (k + rₘ(d))   over methods m ∈ M,  k = 60",
                size=20, color=WHITE)
    return slide


def build_engineering_story_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "Engineering Story: The Short-Text Bug Arc",
                    "Caught and corrected by the project's own evaluation methodology")
    add_bullets(slide, Inches(0.6), Inches(1.75), Inches(12.0), Inches(5.3), [
        "Bug found: aggressive stopword removal + stemming collapsed short "
        "queries to 0–1 tokens (e.g. “What is the google?” → [googl]), "
        "causing degenerate 1.0 cosine-similarity ties across unrelated documents.",
        "Fix #1 (global light mode): removed stopword/stemming everywhere. "
        "Fixed the tie bug, but was applied uniformly — full held-out TF-IDF "
        "dropped to 0.1158 P@10 from a 0.1283 buggy baseline.",
        "Caught by segmentation: splitting eval queries into short (≤5 words) "
        "vs. long revealed the fix had over-corrected long queries, which were "
        "never part of the original bug.",
        "Fix #2 (per-text scoping in one vocabulary): restored long-query "
        "metrics, but fragmented vocabulary inside the shared vectorizer "
        "(e.g. query token “geographic” vs. indexed stem “geograph”), "
        "crashing short-query TF-IDF.",
        "Final fix (dual-space RRF fusion): independent corpus-wide light + "
        "heavy (+ char n-gram for short queries) spaces, fused with the same "
        "RRF formula as the top-level hybrid — recovered short-query TF-IDF "
        "to 0.1450 P@10 and long-query to 0.1294 P@10 simultaneously.",
    ], size=16)
    return slide


def build_results_slide(prs, comparison_rows):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "Results: Held-Out Performance (400 queries, k=10)")

    columns = list(comparison_rows[0].keys())  # Method, P@10, Recall@10, MAP@10, nDCG@10
    n_rows = len(comparison_rows) + 1
    n_cols = len(columns)

    table_left, table_top = Inches(1.4), Inches(1.9)
    table_width, table_height = Inches(10.5), Inches(2.6)
    graphic_frame = slide.shapes.add_table(n_rows, n_cols, table_left, table_top, table_width, table_height)
    table = graphic_frame.table

    for c, col_name in enumerate(columns):
        cell = table.cell(0, c)
        cell.text = col_name
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for run in p.runs:
                run.font.bold = True
                run.font.color.rgb = WHITE
                run.font.size = Pt(16)

    for r, row in enumerate(comparison_rows, start=1):
        method = row["Method"]
        accent = METHOD_COLORS.get(method, GREY)
        for c, col_name in enumerate(columns):
            cell = table.cell(r, c)
            value = row[col_name]
            if col_name != "Method":
                value = f"{float(value):.4f}"
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = accent if col_name == "Method" else LIGHT_BG
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.size = Pt(15)
                    run.font.bold = (col_name == "Method")
                    run.font.color.rgb = WHITE if col_name == "Method" else INK

    add_bullets(slide, Inches(1.4), Inches(4.9), Inches(10.5), Inches(2.0), [
        "Dense leads on all 4 metrics (P@10, Recall@10, MAP@10, nDCG@10).",
        "Hybrid-RRF (BM25 + Dense) is second on all 4 metrics, ahead of both "
        "standalone lexical methods.",
        "BM25 outperforms TF-IDF on every metric within the lexical pair.",
    ], size=16)
    return slide


def build_pr_curve_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "Precision-Recall Curve")
    if os.path.exists(PR_CURVE_PATH):
        slide.shapes.add_picture(PR_CURVE_PATH, Inches(2.4), Inches(1.5), height=Inches(5.5))
    else:
        add_textbox(slide, Inches(1), Inches(3), Inches(11), Inches(1),
                    "reports/pr_curve.png not found — run src/evaluation/run_evaluation.py.",
                    size=20, color=ACCENT_GOLD)
    return slide


def build_search_interface_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "Search Interface (Streamlit UI)")

    screenshot_dir = REPORTS_DIR
    candidate_names = [
        "ui_search_screenshot.png", "ui_screenshot.png", "search_screenshot.png",
        "ui_comparison_screenshot.png", "ui_pr_curve_screenshot.png",
    ]
    found = [os.path.join(screenshot_dir, n) for n in candidate_names
             if os.path.exists(os.path.join(screenshot_dir, n))]

    if found:
        left = Inches(0.8)
        for path in found[:2]:
            slide.shapes.add_picture(path, left, Inches(1.7), height=Inches(5.0))
            left = Emu(int(left + Inches(6.0)))
    else:
        add_textbox(slide, Inches(0.6), Inches(1.6), Inches(12.0), Inches(0.5),
                    "No screenshot found in reports/ — describing the UI instead "
                    "(supply a screenshot to replace this slide):",
                    size=16, color=ACCENT_GOLD)
        add_bullets(slide, Inches(0.8), Inches(2.3), Inches(11.5), Inches(4.0), [
            "Search tab: free-text query box + method dropdown (TF-IDF / BM25 / "
            "Dense / Hybrid-RRF); returns top-10 ranked questions with scores.",
            "Performance Comparison tab: per-metric winner cards plus the full "
            "4-method × 4-metric comparison table (reports/comparison_table.csv).",
            "Precision-Recall Curve tab: interactive Plotly PR curves for all "
            "4 methods, k = 1..50, with k = 10 marked.",
        ], size=17)
    return slide


def build_conclusion_slide(prs, comparison_rows):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_slide_title(slide, "Conclusion")

    by_method = {row["Method"]: row for row in comparison_rows}
    dense = {k: float(v) for k, v in by_method["Dense"].items() if k != "Method"}
    hybrid = {k: float(v) for k, v in by_method["Hybrid-RRF"].items() if k != "Method"}

    add_bullets(slide, Inches(0.6), Inches(1.8), Inches(12.0), Inches(4.5), [
        f"Dense retrieval wins outright: P@10 {dense['P@10']:.4f}, Recall@10 "
        f"{dense['Recall@10']:.4f}, MAP@10 {dense['MAP@10']:.4f}, nDCG@10 "
        f"{dense['nDCG@10']:.4f} — best on every held-out metric.",
        f"Hybrid-RRF (BM25 + Dense) is a close second on every metric "
        f"(nDCG@10 {hybrid['nDCG@10']:.4f} vs. Dense's {dense['nDCG@10']:.4f}) "
        f"while retaining an interpretable lexical signal — the practical "
        f"production choice when exact-term matches still matter.",
        "The short-text bug arc shows the evaluation methodology itself "
        "(segmentation by query length) was what caught a real regression "
        "that the aggregate metric alone had hidden.",
    ], size=18)
    return slide


def main():
    comparison_rows = load_comparison_table()

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    build_title_slide(prs)
    build_problem_statement_slide(prs)
    build_architecture_slide(prs)
    build_methodology_slide(prs)
    build_engineering_story_slide(prs)
    build_results_slide(prs, comparison_rows)
    build_pr_curve_slide(prs)
    build_search_interface_slide(prs)
    build_conclusion_slide(prs, comparison_rows)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    prs.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
