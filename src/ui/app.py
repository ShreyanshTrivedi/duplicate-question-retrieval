"""Streamlit search UI for Duplicate Question Retrieval."""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import streamlit as st

# Ensure project root is on the path so src.* imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid_rrf import HybridRRFRetriever


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
    st.set_page_config(page_title="Duplicate Question Search", layout="wide")
    st.title("🔍 Duplicate Question Retrieval")
    st.markdown("Find semantically similar questions from the Quora Question Pairs corpus.")

    search_tab, comparison_tab, curve_tab = st.tabs(
        ["Search", "Performance Comparison", "Precision-Recall Curve"]
    )

    with search_tab:
        retrievers, corpus_df = load_retrievers()

        # Build a fast lookup: qid -> question text
        id_to_question = dict(zip(corpus_df["qid"].astype(str), corpus_df["question"]))
        config = load_config()
        top_k = config.get("retrieval", {}).get("top_k", 10)

        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter your question:", placeholder="e.g. How do I learn Python fast?")
        with col2:
            method = st.selectbox("Retrieval method:", list(retrievers.keys()))

        if query:
            retriever = retrievers[method]
            results = retriever.rank(query, top_k)

            if results:
                rows = []
                for rank_idx, (doc_id, score) in enumerate(results, start=1):
                    rows.append({
                        "Rank": rank_idx,
                        "Question": id_to_question.get(doc_id, "N/A"),
                        "Score": f"{score:.4f}",
                        "QID": doc_id,
                    })
                st.subheader(f"Top-{top_k} results ({method})")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No results found for this query.")

    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    with comparison_tab:
        comparison_path = reports_dir / "comparison_table.csv"
        if comparison_path.exists():
            st.subheader("Held-out retrieval performance")
            st.dataframe(pd.read_csv(comparison_path), use_container_width=True, hide_index=True)
        else:
            st.warning("Performance comparison data is unavailable.")

    with curve_tab:
        curve_path = reports_dir / "pr_curve.png"
        if curve_path.exists():
            st.subheader("Precision-Recall curves")
            st.image(str(curve_path), use_container_width=True)
        else:
            st.warning("Precision-Recall graph is unavailable.")

if __name__ == "__main__":
    main()
