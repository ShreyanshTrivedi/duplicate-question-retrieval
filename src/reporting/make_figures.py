"""Draws the report figures from real repo files into reports/figures/.

Reads reports/comparison_table.csv, reports/pr_curve_data.json and
data/processed/*.parquet (statistics only). No retriever is fitted.
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORTS = os.path.join(ROOT, "reports")
PROCESSED = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(REPORTS, "figures")

COLORS = {"TF-IDF": "#0F9D8A", "BM25": "#2563EB", "Dense": "#7C3AED", "Hybrid-RRF": "#D97706"}
INK = "#111827"
GREY = "#6B7280"
NAVY = "#1B365D"

plt.rcParams.update({
    "font.family": ["Calibri", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def tint(hex_color, amount=0.16):
    """Blends a colour with white; amount is the colour's share."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    out = (255 - (255 - rgb) * amount).astype(int)
    return "#%02X%02X%02X" % tuple(out)


def box(ax, x, y, w, h, title, body="", color=GREY, size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=tint(color), ec=color, lw=0.9))
    lines = body.count("\n") + 1 if body else 0
    block = 0.145 + 0.135 * lines
    top = y + h / 2 + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=size, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, top - 0.155, body, ha="center", va="top", fontsize=size, color=INK,
                linespacing=1.12)


def arrow(ax, p0, p1, ls="-", head=True):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if head else "-", mutation_scale=8,
                                 color="#4B5563", lw=0.9, ls=ls, shrinkA=0, shrinkB=0))


def canvas(width, height, scale=1.0):
    fig = plt.figure(figsize=(width * scale, height * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def fig_pipeline():
    fig, ax = canvas(6.5, 2.78)
    box(ax, 0.0, 2.12, 1.55, 0.62, "Quora Question Pairs", "404,290 labelled pairs")
    box(ax, 1.95, 2.12, 3.1, 0.62, "Corpus, qrels, evaluation queries",
        "537,929 questions · 298,526 qrels\n400 evaluation queries")
    arrow(ax, (1.55, 2.43), (1.95, 2.43))

    centres = {"TF-IDF": 0.69, "BM25": 2.39, "Hybrid-RRF": 4.09, "Dense": 5.81}
    box(ax, 0.0, 0.98, 1.38, 0.8, "TF-IDF", "light + heavy\n(+ char n-grams,\nshort queries)\nRRF-fused", COLORS["TF-IDF"])
    box(ax, 1.7, 0.98, 1.38, 0.8, "BM25", "light + heavy\nspaces\nRRF-fused", COLORS["BM25"])
    box(ax, 3.4, 0.98, 1.38, 0.8, "Hybrid-RRF", "RRF over the\nBM25 and Dense\nlists, k = 60", COLORS["Hybrid-RRF"])
    box(ax, 5.12, 0.98, 1.38, 0.8, "Dense", "MiniLM-L6 encoder\nFAISS flat index\n100,000 questions", COLORS["Dense"])

    arrow(ax, (3.5, 2.12), (3.5, 1.98), head=False)
    arrow(ax, (centres["TF-IDF"], 1.98), (centres["Dense"], 1.98), head=False)
    for key in ("TF-IDF", "BM25", "Dense"):
        arrow(ax, (centres[key], 1.98), (centres[key], 1.78))
    arrow(ax, (3.08, 1.38), (3.4, 1.38))
    arrow(ax, (5.12, 1.38), (4.78, 1.38))

    box(ax, 0.0, 0.08, 4.4, 0.52, "Evaluation against qrels",
        "P@10 · Recall@10 · MAP@10 · nDCG@10 · PR curve")
    box(ax, 4.72, 0.08, 1.78, 0.52, "Streamlit UI", "Search · Comparison · PR")
    for key, x in centres.items():
        arrow(ax, (x, 0.98), (x, 0.80), head=False)
    arrow(ax, (centres["TF-IDF"], 0.80), (centres["Dense"], 0.80), head=False)
    arrow(ax, (2.2, 0.80), (2.2, 0.60))
    arrow(ax, (5.55, 0.80), (5.55, 0.60), ls="--")
    ax.text(5.62, 0.70, "live search", fontsize=8, color=GREY, va="center")
    arrow(ax, (4.4, 0.34), (4.72, 0.34))
    return save(fig, "fig_pipeline.png")


def fig_dataset():
    corpus = pd.read_parquet(os.path.join(PROCESSED, "corpus.parquet"))
    qrels = pd.read_parquet(os.path.join(PROCESSED, "qrels.parquet"))
    queries = pd.read_parquet(os.path.join(PROCESSED, "eval_queries.parquet"))
    words = corpus["question"].astype(str).str.split().str.len()
    short_share = float((words <= 5).mean() * 100)
    short_count = int((words <= 5).sum())

    per_query = qrels.groupby("query_id").size()
    eval_ids = set(queries["qid"].astype(str))
    rel = per_query[[q for q in per_query.index if str(q) in eval_ids]]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.05), gridspec_kw={"wspace": 0.32})
    clipped = words.clip(upper=40)
    counts = np.bincount(clipped, minlength=41)[1:] / len(words) * 100
    xs = np.arange(1, 41)
    a.bar(xs, counts, width=0.85, color=["#1B365D" if x <= 5 else "#9CA3AF" for x in xs])
    a.set_xlabel("Question length (words; 40 = 40 or more)")
    a.set_ylabel("Share of corpus (%)")
    a.set_title("(a) Question length", loc="left")
    a.text(14.5, counts.max() * 0.66, f"dark bars, ≤ 5 words:\n{short_share:.1f}% of questions\n({short_count:,} of {len(words):,})",
           fontsize=8, color=NAVY, va="center")
    a.axvline(words.median(), color="#B45309", lw=0.9, ls="--")
    a.text(words.median() + 0.8, counts.max() * 1.02, f"median {int(words.median())} words",
           fontsize=8, color="#B45309", va="center")
    a.set_ylim(0, counts.max() * 1.12)

    cats = list(range(1, 11)) + ["11+"]
    vals = [int((rel == k).sum()) for k in range(1, 11)] + [int((rel >= 11).sum())]
    b.bar(range(len(cats)), vals, width=0.8, color="#1B365D")
    b.set_yscale("log")
    b.set_ylim(0.8, 900)
    b.set_yticks([1, 10, 100])
    b.set_yticklabels(["1", "10", "100"])
    b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_xticks(range(len(cats)))
    b.set_xticklabels([str(c) for c in cats])
    b.set_xlabel("Relevant documents per evaluation query")
    b.set_ylabel("Queries (log scale)")
    b.set_title("(b) Relevant documents", loc="left")
    for i, v in enumerate(vals):
        b.text(i, v * 1.12, str(v), ha="center", va="bottom", fontsize=8)
    b.text(10.4, 300, f"mean {rel.mean():.2f}\nmedian {int(rel.median())}\nmax {int(rel.max())}",
           ha="right", va="center", fontsize=8, color=GREY)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.23)
    return save(fig, "fig_dataset.png")


def fig_dual_space():
    fig, ax = canvas(6.5, 1.98, scale=0.8)
    box(ax, 0.0, 0.62, 1.0, 0.74, "Query", "“What is\nthe google?”")
    teal = COLORS["TF-IDF"]
    lanes = [
        (1.38, "Heavy: stop-words removed, stemmed", "[googl]"),
        (0.70, "Light: surface tokens + bigrams",
         "what · is · the · google\nwhat__is · is__the · the__google"),
        (0.02, "Character 3–5-grams (short queries only)",
         "' wh' · 'wha' · 'hat' · ' what' · 'what ' · …"),
    ]
    for y, title, body in lanes:
        lines = body.count("\n") + 1
        h = 0.58
        box(ax, 1.3, y, 3.35, h, title, body, teal)
        cy = y + h / 2
        arrow(ax, (1.0, 0.99), (1.3, cy))
        box(ax, 4.9, y + 0.07, 0.72, h - 0.14, "ranked", "list")
        arrow(ax, (4.65, cy), (4.9, cy))
        arrow(ax, (5.62, cy), (5.88, cy))
    box(ax, 5.88, 0.35, 0.62, 1.3, "RRF", "k = 60\n↓\ntop 10", COLORS["Hybrid-RRF"])
    return save(fig, "fig_dual_space.png")


def fig_indexing_designs():
    groups = ["All 400", "Short 40", "Long 360"]
    designs = [
        ("Original (heavy only)", "#D1D5DB", {"All 400": 0.7274}),
        ("Global light", "#9CA3AF", {"All 400": 0.6184, "Short 40": 0.5887}),
        ("Per-text, shared vocabulary", "#4B5563", {"All 400": 0.6784, "Long 360": 0.7148, "Short 40": 0.3508}),
        ("Dual-space RRF (final)", "#BE185D", {"All 400": 0.7314, "Short 40": 0.7908, "Long 360": 0.7247}),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 1.75))
    width = 0.19
    first_seen = set()
    for d, (name, color, cells) in enumerate(designs):
        for g, group in enumerate(groups):
            if group not in cells:
                continue
            x = g + (d - 1.5) * width
            ax.bar(x, cells[group], width=width * 0.95, color=color,
                   label=None if name in first_seen else name)
            first_seen.add(name)
            ax.text(x, cells[group] + 0.012, f"{cells[group]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("TF-IDF nDCG@10")
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
               columnspacing=1.2, handlelength=1.1)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.83, bottom=0.15)
    return save(fig, "fig_indexing_designs.png")


def fig_results():
    with open(os.path.join(REPORTS, "comparison_table.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(REPORTS, "pr_curve_data.json"), encoding="utf-8") as f:
        pr = json.load(f)
    metrics = ["P@10", "Recall@10", "MAP@10", "nDCG@10"]
    methods = [r["Method"] for r in rows]

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.05]})
    width = 0.19
    for i, row in enumerate(rows):
        vals = [float(row[m]) for m in metrics]
        xs = np.arange(len(metrics)) + (i - 1.5) * width
        a.bar(xs, vals, width=width * 0.94, color=COLORS[row["Method"]], label=row["Method"])
        for x, v in zip(xs, vals):
            a.text(x, v + 0.012, f"{v:.3f}", rotation=90, ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(len(metrics)))
    a.set_xticklabels(metrics)
    a.set_ylim(0, 1.28)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    a.set_ylabel("Score (400 queries)")
    a.set_title("(a) Four metrics at cutoff 10", loc="left")

    for method in methods:
        rec, prec = pr[method]["recall"], pr[method]["precision"]
        b.plot(rec, prec, color=COLORS[method], lw=1.5)
        b.plot([rec[9]], [prec[9]], "o", color=COLORS[method], ms=6, mec="white", mew=0.8)
    b.set_xlabel("Recall")
    b.set_ylabel("Precision")
    b.set_title("(b) Precision-recall, k = 1 to 50", loc="left")
    b.text(0.535, 0.06, "large markers: k = 10", fontsize=8, color=GREY)
    b.set_ylim(0, 0.97)
    b.grid(color="#E5E7EB", lw=0.6)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               handlelength=1.2, columnspacing=1.6)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.81, bottom=0.15)
    return save(fig, "fig_results.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    paths = [fig_dataset(), fig_pipeline(), fig_dual_space(), fig_indexing_designs(), fig_results()]
    for p in paths:
        print("wrote", p)
    return paths


if __name__ == "__main__":
    main()

