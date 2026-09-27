"""Run the held-out lexical evaluation split by the documented word threshold."""

from __future__ import annotations

import os

import pandas as pd
import yaml

from src.evaluation.metrics import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.tfidf import TfidfRetriever


def _evaluate(retriever, queries: pd.DataFrame, qrels: dict[str, set[str]], k: int) -> dict[str, float]:
    totals = {"precision_at_10": 0.0, "recall_at_10": 0.0, "map_at_10": 0.0, "ndcg_at_10": 0.0}
    for index, row in enumerate(queries.itertuples(index=False), start=1):
        if index % 50 == 0:
            print(f"  progress: {index}/{len(queries)}", flush=True)
        ranked = [doc_id for doc_id, _ in retriever.rank(row.question, top_k=k)]
        relevant = qrels[str(row.qid)]
        totals["precision_at_10"] += precision_at_k(ranked, relevant, k)
        totals["recall_at_10"] += recall_at_k(ranked, relevant, k)
        totals["map_at_10"] += average_precision(ranked, relevant)
        totals["ndcg_at_10"] += ndcg_at_k(ranked, relevant, k)
    return {metric: value / len(queries) for metric, value in totals.items()}


def main() -> None:
    with open("config.yaml", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    corpus = pd.read_parquet(config["data"]["corpus_path"])
    queries = pd.read_parquet(config["data"]["eval_queries_path"])
    qrels_df = pd.read_parquet(config["data"]["qrels_path"])
    qrels = {str(qid): set(group["doc_id"].astype(str)) for qid, group in qrels_df.groupby("query_id")}
    queries = queries.copy()
    queries["qid"] = queries["qid"].astype(str)
    queries["question"] = queries["question"].astype(str)
    queries = queries[queries["qid"].isin(qrels)]
    threshold = config.get("preprocessing", {}).get("short_text_word_threshold", 5)
    groups = {
        "short": queries[queries["question"].str.split().str.len() <= threshold],
        "long": queries[queries["question"].str.split().str.len() > threshold],
    }
    rows = []
    for method, retriever in {"TF-IDF": TfidfRetriever(config), "BM25": BM25Retriever(config)}.items():
        print(f"Fitting {method}...", flush=True)
        retriever.fit(corpus)
        for group, group_queries in groups.items():
            print(f"Evaluating {method} {group} ({len(group_queries)} queries)...", flush=True)
            rows.append({
                "method": method,
                "group": group,
                "query_count": len(group_queries),
                **_evaluate(retriever, group_queries, qrels, 10),
            })
    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(rows).to_csv("reports/segmented_metrics.csv", index=False)
    print("Wrote reports/segmented_metrics.csv", flush=True)


if __name__ == "__main__":
    main()