import numpy as np
from rank_bm25 import BM25Okapi
from src.retrieval.base import RetrieverBase
from src.preprocessing.clean import preprocess
from src.retrieval.hybrid_rrf import reciprocal_rank_fusion

class BM25Retriever(RetrieverBase):
    def __init__(self, config: dict):
        self.config = config
        self.doc_ids = []
        self.doc_texts = []
        self.light_bm25 = None
        self.heavy_bm25 = None

    def _preprocess_mode(self, text: str, short_text_mode: bool) -> list[str]:
        text_config = dict(self.config)
        text_config["preprocessing"] = dict(self.config.get("preprocessing", {}))
        text_config["preprocessing"]["short_text_mode"] = short_text_mode
        return preprocess(text, text_config)
        
    def fit(self, corpus_df):
        self.doc_ids = corpus_df['qid'].astype(str).tolist()
        self.doc_texts = corpus_df['question'].astype(str).tolist()
        light_corpus = [self._preprocess_mode(text, True) for text in self.doc_texts]
        heavy_corpus = [self._preprocess_mode(text, False) for text in self.doc_texts]
        self.light_bm25 = BM25Okapi(light_corpus)
        self.heavy_bm25 = BM25Okapi(heavy_corpus)

    def _rank_space(self, model, tokens: list[str], query: str, pool_size: int) -> list[tuple[str, float]]:
        scores = model.get_scores(tokens)
        order = np.argsort(scores)[::-1]
        return [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in order[:pool_size]
            if scores[idx] > 0 and self.doc_texts[idx].strip() != query.strip()
        ]
        
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        pool_size = max(100, top_k)
        result_lists = [
            self._rank_space(self.light_bm25, self._preprocess_mode(query, True), query, pool_size),
            self._rank_space(self.heavy_bm25, self._preprocess_mode(query, False), query, pool_size),
        ]
        rrf_k = self.config.get("retrieval", {}).get("rrf_k", 60)
        return reciprocal_rank_fusion(result_lists, top_k, rrf_k)
