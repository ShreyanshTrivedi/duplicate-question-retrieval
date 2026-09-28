"""Streamlit search UI for Duplicate Question Retrieval."""

import sys
import os
import json
from pathlib import Path
import yaml
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure project root is on the path so src.* imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid_rrf import HybridRRFRetriever


MODEL_COLORS = {
    "TF-IDF": "#2DD4BF",
    "BM25": "#60A5FA",
    "Dense": "#C084FC",
    "Hybrid-RRF": "#FBBF24",
}

METHOD_DESCRIPTIONS = {
    "TF-IDF": "Combines light word-and-bigram matching with heavy stemmed matching for robust lexical retrieval.",
    "BM25": "Ranks questions with complementary light and stemmed term-frequency matching.",
    "Dense": "Finds semantic similarity using sentence embeddings rather than exact word overlap.",
    "Hybrid-RRF": "Fuses BM25 lexical ranking with Dense semantic ranking using reciprocal rank fusion.",
}


def build_results_df(results: list, id_to_question: dict) -> pd.DataFrame:
    """Builds a results dataframe from ranked (doc_id, score) pairs.

    Args:
        results: List of (doc_id, score) tuples from a retriever.
        id_to_question: Mapping from doc_id to question text.

    Returns:
        DataFrame with Rank, Question, Score (float), QID columns.
    """
    rows = []
    for rank_idx, (doc_id, score) in enumerate(results, start=1):
        rows.append({
            "Rank": rank_idx,
            "Question": id_to_question.get(doc_id, "N/A"),
            "Score": float(score),
            "QID": doc_id,
        })
    return pd.DataFrame(rows)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a #RRGGBB hex color to an rgba() string at the given alpha."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def style_results(df: pd.DataFrame, accent: str):
    """Tints the rank-1 row of a results dataframe with the accent color.

    Args:
        df: Results dataframe as built by build_results_df.
        accent: Hex color string for the highlight.

    Returns:
        A pandas Styler.
    """
    highlight = _hex_to_rgba(accent, 0.12)
    return df.style.apply(
        lambda row: [f"background-color: {highlight}" if row.name == 0 else "" for _ in row],
        axis=1,
    )


def best_per_metric(df: pd.DataFrame, metric_columns: list) -> dict:
    """Finds the winning method and value for each metric column.

    Args:
        df: Comparison dataframe with a Method column and metric columns.
        metric_columns: Metric column names to evaluate.

    Returns:
        Dict mapping metric name to (method, value).
    """
    winners = {}
    for column in metric_columns:
        best_index = df[column].idxmax()
        winners[column] = (df.loc[best_index, "Method"], df.loc[best_index, column])
    return winners


def summary_sentence(df: pd.DataFrame, metric_columns: list) -> str:
    """Builds a plain-language sentence naming the method with the most metric wins.

    Args:
        df: Comparison dataframe with a Method column and metric columns.
        metric_columns: Metric column names to evaluate.

    Returns:
        A summary sentence.
    """
    wins = df[metric_columns].eq(df[metric_columns].max())
    win_counts = wins.sum(axis=1)
    leaders = df.loc[win_counts == win_counts.max(), "Method"].tolist()
    if len(leaders) == 1:
        return f"{leaders[0]} leads the most metric categories in this evaluation."
    return f"The leading methods are tied across the most metric categories: {', '.join(leaders)}."


def build_pr_figure(pr_data: dict) -> go.Figure:
    """Builds a Precision-Recall plotly figure from curve data.

    Args:
        pr_data: Dict of method -> {"recall": [...], "precision": [...]}.

    Returns:
        A plotly Figure with one trace per method.
    """
    k10_index = 9
    fig = go.Figure()
    for method, values in pr_data.items():
        color = MODEL_COLORS.get(method, "#94A3B8")
        n_points = len(values["recall"])
        marker_sizes = [5] * n_points
        marker_line_widths = [0] * n_points
        if n_points > k10_index:
            marker_sizes[k10_index] = 11
            marker_line_widths[k10_index] = 1.5
        fig.add_trace(go.Scatter(
            x=values["recall"],
            y=values["precision"],
            mode="lines+markers",
            name=method,
            line={"color": color, "width": 2.5},
            marker={
                "size": marker_sizes,
                "line": {"width": marker_line_widths, "color": "#FFFFFF"},
            },
        ))
    axis_style = {
        "showline": True,
        "linecolor": "rgba(230,237,247,0.45)",
        "mirror": True,
        "ticks": "outside",
        "tickcolor": "rgba(230,237,247,0.45)",
        "tickfont": {"color": "#C9D3E6", "size": 12},
        "gridcolor": "rgba(230,237,247,0.10)",
        "zeroline": False,
        "automargin": True,
        "title": {"font": {"color": "#E6EDF7", "size": 13}},
    }
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D1626",
        font={"color": "#E5E7EB"},
        legend={"orientation": "h", "y": 1.18, "x": 0, "xanchor": "left", "font": {"color": "#E6EDF7"}},
        margin={"l": 20, "r": 20, "t": 90, "b": 70},
        height=460,
        hovermode="x unified",
        hoverlabel={"bgcolor": "#111A2E", "bordercolor": "rgba(148,163,184,0.16)", "font": {"color": "#E6EDF7"}},
        xaxis=axis_style,
        yaxis=axis_style,
    )
    return fig


def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the project configuration.

    Args:
        config_path: Path to the yaml config file.

    Returns:
        The configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading corpus and fitting retrievers …")
def load_retrievers() -> tuple[dict, pd.DataFrame]:
    """Loads the corpus and fits all retrievers once at startup.

    Returns:
        Tuple of (retrievers dict, corpus dataframe).
    """
    config = load_config()
    corpus_df = pd.read_parquet(config["data"]["corpus_path"])

    tfidf = TfidfRetriever(config)
    tfidf.fit(corpus_df)

    bm25 = BM25Retriever(config)
    bm25.fit(corpus_df)

    dense = DenseRetriever(config)
    dense.fit(corpus_df)

    hybrid = HybridRRFRetriever([bm25, dense], config)

    retrievers = {
        "TF-IDF": tfidf,
        "BM25": bm25,
        "Dense": dense,
        "Hybrid-RRF": hybrid,
    }
    return retrievers, corpus_df


def main() -> None:
    """Entry point for the Streamlit search UI."""
    st.set_page_config(page_title="Duplicate Question Search", page_icon="🔍", layout="wide")

    css_path = Path(__file__).resolve().parent / "theme.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

    eval_query_count = load_config().get("data", {}).get("eval_query_count")
    st.markdown(
        f"""
        <h1 class="hero-title">Duplicate Question <span class="accent">Retrieval</span></h1>
        <div class="hero-tagline">Has this question already been asked?</div>
        <div class="hero-sub">Search the Quora Question Pairs corpus with lexical, semantic and hybrid retrieval.</div>
        <div class="chip-row">
            <span class="chip">4 retrieval methods</span>
            <span class="chip">{eval_query_count} held-out queries</span>
            <span class="chip">Quora Question Pairs</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    search_tab, comparison_tab, curve_tab = st.tabs(
        ["Search", "Performance Comparison", "Precision-Recall Curve"]
    )

    with search_tab:
        retrievers, corpus_df = load_retrievers()

        # Build a fast lookup: qid -> question text
        id_to_question = dict(zip(corpus_df["qid"].astype(str), corpus_df["question"]))
        config = load_config()
        top_k = config.get("retrieval", {}).get("top_k", 10)

        with st.container(border=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input("Enter your question:", placeholder="e.g. How do I learn Python fast?")
            with col2:
                method = st.selectbox("Retrieval method:", list(retrievers.keys()))
                st.caption(METHOD_DESCRIPTIONS[method])

            if query:
                retriever = retrievers[method]
                results = retriever.rank(query, top_k)

                if results:
                    st.subheader(f"Top-{top_k} results ({method})")
                    results_df = build_results_df(results, id_to_question)
                    accent = MODEL_COLORS[method]
                    styled_results = style_results(results_df, accent)
                    scores = results_df["Score"]
                    score_min = float(scores.min())
                    score_max = float(scores.max())
                    if score_min == score_max:
                        score_max = score_min + 1e-9
                    st.dataframe(
                        styled_results,
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Score",
                                format="%.4f",
                                min_value=score_min,
                                max_value=score_max,
                            )
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning("No results found for this query.")

    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    with comparison_tab:
        comparison_path = reports_dir / "comparison_table.csv"
        if comparison_path.exists():
            with st.container(border=True):
                st.subheader("Held-out retrieval performance")
                comparison_df = pd.read_csv(comparison_path)
                metric_columns = ["P@10", "Recall@10", "MAP@10", "nDCG@10"]
                metric_cards = st.columns(4)
                winners = best_per_metric(comparison_df, metric_columns)
                for column, card in zip(metric_columns, metric_cards):
                    best_method, best_value = winners[column]
                    card.metric(column, f"{best_value:.4f}", best_method)
                styled_comparison = comparison_df.style.highlight_max(
                    axis=0,
                    subset=metric_columns,
                    color=f"{MODEL_COLORS['TF-IDF']}55",
                )
                st.dataframe(styled_comparison, use_container_width=True, hide_index=True)
                st.caption(summary_sentence(comparison_df, metric_columns))
        else:
            st.warning("Performance comparison data is unavailable.")

    with curve_tab:
        curve_data_path = reports_dir / "pr_curve_data.json"
        curve_path = reports_dir / "pr_curve.png"
        if curve_data_path.exists():
            with st.container(border=True):
                st.subheader("Precision-Recall curves")
                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                with curve_data_path.open(encoding="utf-8") as file:
                    curve_data = json.load(file)
                fig = build_pr_figure(curve_data)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Larger markers show k = 10.")
        elif curve_path.exists():
            with st.container(border=True):
                st.image(str(curve_path), use_container_width=True)
                st.info("The interactive chart appears after the next evaluation run.")
        else:
            st.warning("Precision-Recall graph is unavailable.")

if __name__ == "__main__":
    main()
