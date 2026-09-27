"""Temporary e2e spot-check; not part of the product."""
from __future__ import annotations

import time

import pandas as pd
import yaml

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid_rrf import HybridRRFRetriever
from src.retrieval.tfidf import TfidfRetriever

SPOT_QIDS = [475310, 486542, 19263, 45891, 38315]


def main() -> None:
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    corpus_df = pd.read_parquet(config["data"]["corpus_path"])
    qrels_df = pd.read_parquet(config["data"]["qrels_path"])
    eval_df = pd.read_parquet(config["data"]["eval_queries_path"])

    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)
    id_to_q = dict(zip(corpus_df["qid"].astype(str), corpus_df["question"].astype(str)))
    eval_by = dict(zip(eval_df["qid"].astype(str), eval_df["question"].astype(str)))
    qrels = {
        qid: set(g["doc_id"].tolist())
        for qid, g in qrels_df.groupby("query_id")
    }

    t0 = time.time()
    tfidf = TfidfRetriever(config)
    tfidf.fit(corpus_df)
    print(f"TF-IDF fit in {time.time()-t0:.1f}s, n={len(tfidf.doc_ids)}", flush=True)

    t0 = time.time()
    bm25 = BM25Retriever(config)
    bm25.fit(corpus_df)
    print(f"BM25 fit in {time.time()-t0:.1f}s, n={len(bm25.doc_ids)}", flush=True)

    t0 = time.time()
    dense = DenseRetriever(config)
    dense.fit(corpus_df)
    print(f"Dense fit in {time.time()-t0:.1f}s, n={len(dense.doc_ids)}", flush=True)

    hybrid = HybridRRFRetriever([bm25, dense], config)
    retrievers = {
        "TF-IDF": tfidf,
        "BM25": bm25,
        "Dense": dense,
        "Hybrid-RRF": hybrid,
    }

    for qid in SPOT_QIDS:
        qid_s = str(qid)
        query = eval_by[qid_s]
        rel = qrels[qid_s]
        print("\n" + "=" * 80, flush=True)
        print(f"qid={qid}  n_rel={len(rel)}", flush=True)
        print(f"QUERY: {query}", flush=True)
        print("QRELS:", flush=True)
        for d in sorted(rel):
            print(f"  {d}: {id_to_q.get(d, 'MISSING')}", flush=True)

        for name, r in retrievers.items():
            t0 = time.time()
            ranked = r.rank(query, top_k=10)
            elapsed = time.time() - t0
            ids = [d for d, _ in ranked]
            scores = [s for _, s in ranked]
            assert len(ranked) == 10, f"{name} returned {len(ranked)}"
            assert len(ids) == len(set(ids)), f"{name} duplicate ids"
            assert qid_s not in ids, f"{name} returned query id"
            assert scores == sorted(scores, reverse=True), f"{name} not sorted"
            hits = [d for d in ids if d in rel]
            print(
                f"\n{name}  hits={len(hits)}/{len(rel)}  rank_s={elapsed:.3f}",
                flush=True,
            )
            for i, (doc_id, score) in enumerate(ranked, start=1):
                mark = "HIT" if doc_id in rel else "   "
                print(
                    f"  {i:2d}. [{mark}] {score:.4f}  {doc_id}  {id_to_q.get(doc_id, 'MISSING')}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
