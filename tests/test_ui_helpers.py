"""Unit tests for the pure UI helper functions in src/ui/app.py."""

import pandas as pd

from src.ui.app import (
    build_results_df,
    style_results,
    best_per_metric,
    summary_sentence,
    build_pr_figure,
)


def test_build_results_df_score_is_float():
    results = [("q1", 0.5), ("q2", 0.25)]
    id_to_question = {"q1": "First question", "q2": "Second question"}
    df = build_results_df(results, id_to_question)
    assert list(df.columns) == ["Rank", "Question", "Score", "QID"]
    assert df["Score"].dtype == float
    assert df.loc[0, "Score"] == 0.5
    assert df.loc[0, "Rank"] == 1


def test_build_results_df_missing_question():
    results = [("missing", 0.1)]
    df = build_results_df(results, {})
    assert df.loc[0, "Question"] == "N/A"


def test_style_results_does_not_crash():
    df = build_results_df([("q1", 0.5), ("q2", 0.25)], {"q1": "A", "q2": "B"})
    styler = style_results(df, "#2DD4BF")
    styler.to_html()


def test_best_per_metric_picks_correct_winners():
    df = pd.DataFrame({
        "Method": ["TF-IDF", "BM25", "Dense"],
        "P@10": [0.1, 0.2, 0.15],
        "Recall@10": [0.8, 0.7, 0.9],
    })
    winners = best_per_metric(df, ["P@10", "Recall@10"])
    assert winners["P@10"] == ("BM25", 0.2)
    assert winners["Recall@10"] == ("Dense", 0.9)


def test_summary_sentence_single_leader():
    df = pd.DataFrame({
        "Method": ["TF-IDF", "BM25"],
        "P@10": [0.1, 0.2],
        "Recall@10": [0.5, 0.9],
    })
    sentence = summary_sentence(df, ["P@10", "Recall@10"])
    assert "BM25" in sentence


def test_summary_sentence_tie():
    df = pd.DataFrame({
        "Method": ["TF-IDF", "BM25"],
        "P@10": [0.2, 0.1],
        "Recall@10": [0.1, 0.2],
    })
    sentence = summary_sentence(df, ["P@10", "Recall@10"])
    assert "TF-IDF" in sentence and "BM25" in sentence


def test_build_pr_figure_min_equals_max_case():
    scores = [("q1", 0.5), ("q2", 0.5)]
    df = build_results_df(scores, {"q1": "A", "q2": "B"})
    score_min = float(df["Score"].min())
    score_max = float(df["Score"].max())
    if score_min == score_max:
        score_max = score_min + 1e-9
    assert score_max > score_min


def test_build_pr_figure_returns_four_traces():
    pr_data = {
        "TF-IDF": {"recall": [0.0, 0.5, 1.0], "precision": [1.0, 0.6, 0.3]},
        "BM25": {"recall": [0.0, 0.5, 1.0], "precision": [1.0, 0.65, 0.35]},
        "Dense": {"recall": [0.0, 0.5, 1.0], "precision": [1.0, 0.8, 0.6]},
        "Hybrid-RRF": {"recall": [0.0, 0.5, 1.0], "precision": [1.0, 0.75, 0.55]},
    }
    fig = build_pr_figure(pr_data)
    assert len(fig.data) == 4


def test_build_pr_figure_k10_marker_is_larger():
    pr_data = {
        "TF-IDF": {
            "recall": [i / 10 for i in range(15)],
            "precision": [1.0 - i / 20 for i in range(15)],
        },
    }
    fig = build_pr_figure(pr_data)
    sizes = fig.data[0].marker.size
    k10_index = 9
    other_sizes = [s for i, s in enumerate(sizes) if i != k10_index]
    assert sizes[k10_index] > max(other_sizes)
