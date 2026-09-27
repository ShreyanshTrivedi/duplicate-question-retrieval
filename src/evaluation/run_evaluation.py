import os
import yaml
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid_rrf import HybridRRFRetriever
from src.evaluation.metrics import precision_at_k, recall_at_k, average_precision, ndcg_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_evaluation():
    config = load_config()
    corpus_path = config["data"]["corpus_path"]
    qrels_path = config["data"]["qrels_path"]
    eval_queries_path = config["data"]["eval_queries_path"]
    
    logging.info("Loading datasets...")
    corpus_df = pd.read_parquet(corpus_path)
    qrels_df = pd.read_parquet(qrels_path)
    eval_queries_df = pd.read_parquet(eval_queries_path)
    
    qrels_dict = {}
    for qid, group in qrels_df.groupby("query_id"):
        qrels_dict[str(qid)] = set(group["doc_id"].astype(str).tolist())
        
    logging.info("Initializing and fitting retrievers...")
    retrievers = {
        "TF-IDF": TfidfRetriever(config),
        "BM25": BM25Retriever(config),
        "Dense": DenseRetriever(config)
    }
    
    for name, r in retrievers.items():
        logging.info(f"Fitting {name}...")
        r.fit(corpus_df)
        
    logging.info("Fitting Hybrid RRF...")
    hybrid = HybridRRFRetriever([retrievers["BM25"], retrievers["Dense"]], config)
    retrievers["Hybrid-RRF"] = hybrid

    id_to_question = dict(
        zip(corpus_df["qid"].astype(str), corpus_df["question"].astype(str))
    )
    _write_spot_check(retrievers, eval_queries_df, qrels_dict, id_to_question)
    
    k_eval = config.get("retrieval", {}).get("top_k", 10)
    
    results = []
    
    # For PR curve
    k_range = list(range(1, 51)) # up to K=50 for PR curve
    pr_data = {name: {"P": np.zeros(len(k_range)), "R": np.zeros(len(k_range))} for name in retrievers.keys()}
    
    queries = eval_queries_df.to_dict("records")
    
    logging.info(f"Evaluating {len(queries)} queries...")
    for name, r in retrievers.items():
        logging.info(f"Evaluating {name}...")
        p_at_k_sum = 0
        r_at_k_sum = 0
        map_sum = 0
        ndcg_sum = 0
        
        valid_queries = 0
        for q_row in queries:
            q_id = str(q_row["qid"])
            q_text = q_row["question"]
            
            if q_id not in qrels_dict:
                continue
                
            relevant_docs = qrels_dict[q_id]
            valid_queries += 1
            if valid_queries % 50 == 0:
                logging.info("%s progress: %d/%d queries", name, valid_queries, len(queries))
            
            max_k = max(k_eval, max(k_range))
            ranked_res = r.rank(q_text, top_k=max_k)
            ranked_list = [x[0] for x in ranked_res]
            
            p_at_k_sum += precision_at_k(ranked_list, relevant_docs, k_eval)
            r_at_k_sum += recall_at_k(ranked_list, relevant_docs, k_eval)
            map_sum += average_precision(ranked_list[:k_eval], relevant_docs) 
            ndcg_sum += ndcg_at_k(ranked_list, relevant_docs, k_eval)
            
            for idx, k_val in enumerate(k_range):
                pr_data[name]["P"][idx] += precision_at_k(ranked_list, relevant_docs, k_val)
                pr_data[name]["R"][idx] += recall_at_k(ranked_list, relevant_docs, k_val)
                
        if valid_queries > 0:
            results.append({
                "Method": name,
                f"P@{k_eval}": p_at_k_sum / valid_queries,
                f"Recall@{k_eval}": r_at_k_sum / valid_queries,
                f"MAP@{k_eval}": map_sum / valid_queries,
                f"nDCG@{k_eval}": ndcg_sum / valid_queries
            })
            
            for idx in range(len(k_range)):
                pr_data[name]["P"][idx] /= valid_queries
                pr_data[name]["R"][idx] /= valid_queries
                
    os.makedirs("reports", exist_ok=True)
    
    df_res = pd.DataFrame(results)
    df_res.to_csv("reports/comparison_table.csv", index=False)
    
    print("\n--- Evaluation Results ---")
    print(df_res.to_string(index=False))
    
    plt.figure(figsize=(8, 6))
    for name in retrievers.keys():
        plt.plot(pr_data[name]["R"], pr_data[name]["P"], label=name, marker='.', markersize=4)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("reports/pr_curve.png", dpi=300, bbox_inches='tight')
    logging.info("Saved reports to reports/ directory.")


SPOT_CHECK_QIDS = [1182, 1353, 4119, 5569, 9495]


def _write_spot_check(retrievers, eval_queries_df, qrels_dict, id_to_question) -> None:
    """Ranks five held-out eval queries with every method and writes a text dump."""
    os.makedirs("reports", exist_ok=True)
    lines = ["# Spot-check rankings (eval queries)", ""]
    eval_by_qid = {
        str(row["qid"]): row["question"]
        for row in eval_queries_df.to_dict("records")
    }
    for qid in SPOT_CHECK_QIDS:
        qid_s = str(qid)
        query = eval_by_qid.get(qid_s)
        if query is None:
            lines.append(f"## qid={qid}: NOT IN eval_queries")
            continue
        rel = qrels_dict.get(qid_s, set())
        rel_texts = [f"{d}: {id_to_question.get(d, 'MISSING')}" for d in sorted(rel)]
        lines.append(f"## qid={qid}")
        lines.append(f"Query: {query}")
        lines.append("Qrels:")
        for rt in rel_texts:
            lines.append(f"  - {rt}")
        for name, retriever in retrievers.items():
            ranked = retriever.rank(query, top_k=10)
            doc_ids = [d for d, _ in ranked]
            assert len(doc_ids) == len(set(doc_ids)), f"duplicate ids from {name}"
            assert qid_s not in doc_ids, f"{name} returned the query doc itself"
            hits = [d for d in doc_ids if d in rel]
            lines.append(f"### {name} (qrel hits in top-10: {len(hits)}/{len(rel)})")
            for i, (doc_id, score) in enumerate(ranked, start=1):
                mark = "HIT" if doc_id in rel else "   "
                text = id_to_question.get(doc_id, "MISSING")
                lines.append(f"  {i:2d}. [{mark}] {score:.4f}  {doc_id}  {text}")
        lines.append("")
    path = "reports/spot_check_rankings.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logging.info("Wrote %s", path)


if __name__ == "__main__":
    run_evaluation()
