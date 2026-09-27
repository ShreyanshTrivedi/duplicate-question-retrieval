import os
import logging

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval.base import RetrieverBase


class DenseRetriever(RetrieverBase):
    def __init__(self, config: dict):
        self.config = config
        model_name = config.get("retrieval", {}).get(
            "dense_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_ids = []
        self.doc_texts = []

    def _embeddings_paths(self) -> tuple[str, str]:
        """Returns (embeddings_path, ids_path) from config with defaults."""
        embeddings_path = self.config.get("data", {}).get(
            "embeddings_path", "data/processed/embeddings.npy"
        )
        ids_path = embeddings_path.replace(".npy", "_ids.npy")
        return embeddings_path, ids_path

    def _must_keep_qids(self, corpus_df: pd.DataFrame) -> set[str]:
        """QIDs that must remain in a subsample so eval qrels stay answerable."""
        must_keep: set[str] = set()
        data_cfg = self.config.get("data", {})
        eval_path = data_cfg.get("eval_queries_path")
        qrels_path = data_cfg.get("qrels_path")
        if not eval_path or not os.path.exists(eval_path):
            return must_keep

        eval_df = pd.read_parquet(eval_path)
        eval_ids = set(eval_df["qid"].astype(str))
        must_keep.update(eval_ids)
        if qrels_path and os.path.exists(qrels_path):
            qrels_df = pd.read_parquet(qrels_path)
            query_ids = qrels_df["query_id"].astype(str)
            doc_ids = qrels_df["doc_id"].astype(str)
            must_keep.update(doc_ids[query_ids.isin(eval_ids)].tolist())
        return must_keep

    def _select_corpus(self, corpus_df: pd.DataFrame) -> pd.DataFrame:
        """Optionally subsample the corpus while keeping eval-relevant docs."""
        subsample = self.config.get("retrieval", {}).get("corpus_subsample")
        if not subsample or len(corpus_df) <= subsample:
            return corpus_df

        must_keep = self._must_keep_qids(corpus_df)
        corpus_ids = corpus_df["qid"].astype(str)
        keep_mask = corpus_ids.isin(must_keep)
        keep_df = corpus_df.loc[keep_mask]
        remaining = corpus_df.loc[~keep_mask]
        n_fill = max(0, int(subsample) - len(keep_df))
        if n_fill > 0 and len(remaining) > 0:
            fill = remaining.sample(n=min(n_fill, len(remaining)), random_state=42)
            selected = pd.concat([keep_df, fill], ignore_index=True)
        else:
            selected = keep_df
        logging.info(
            "Subsampled corpus to %s documents (%s forced-in for eval coverage).",
            len(selected),
            keep_mask.sum(),
        )
        return selected

    def _load_cached_embeddings(self, n_docs: int) -> np.ndarray | None:
        embeddings_path, ids_path = self._embeddings_paths()
        if not (os.path.exists(embeddings_path) and os.path.exists(ids_path)):
            return None
        logging.info("Loading cached embeddings...")
        embeddings = np.load(embeddings_path)
        cached_ids = np.load(ids_path, allow_pickle=True).tolist()
        if cached_ids == self.doc_ids and len(embeddings) == n_docs:
            return embeddings
        logging.info("Cached embeddings do not match current corpus, recomputing...")
        return None

    def fit(self, corpus_df: pd.DataFrame) -> None:
        corpus_df = self._select_corpus(corpus_df)
        self.doc_ids = corpus_df["qid"].astype(str).tolist()
        self.doc_texts = corpus_df["question"].astype(str).tolist()
        texts = self.doc_texts

        embeddings = self._load_cached_embeddings(len(texts))
        if embeddings is None:
            logging.info("Computing embeddings...")
            embeddings = self.model.encode(
                texts, show_progress_bar=True, normalize_embeddings=True
            )
            embeddings_path, ids_path = self._embeddings_paths()
            os.makedirs(os.path.dirname(embeddings_path) or ".", exist_ok=True)
            np.save(embeddings_path, embeddings)
            np.save(ids_path, np.array(self.doc_ids, dtype=object))

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(np.asarray(embeddings, dtype="float32"))

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        query_emb = self.model.encode([query], normalize_embeddings=True)
        n_docs = self.index.ntotal
        fetch = min(n_docs, max(top_k + 5, top_k))
        scores, indices = self.index.search(
            np.asarray(query_emb, dtype="float32"), fetch
        )

        qnorm = query.strip()
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if self.doc_texts[idx].strip() == qnorm:
                continue
            results.append((self.doc_ids[idx], float(score)))
            if len(results) >= top_k:
                break
        return results
