import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.base import RetrieverBase
from src.preprocessing.clean import is_short_text, preprocess
from src.retrieval.hybrid_rrf import reciprocal_rank_fusion

class TfidfRetriever(RetrieverBase):
    def __init__(self, config: dict):
        self.config = config
        self.light_vectorizer = TfidfVectorizer(
            analyzer=self._light_features,
            token_pattern=None,
        )
        self.heavy_vectorizer = TfidfVectorizer(
            analyzer=self._heavy_features,
            token_pattern=None,
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5)
        )
        self.doc_ids = []
        self.doc_texts = []
        self.light_matrix = None
        self.heavy_matrix = None
        self.char_tfidf_matrix = None

    def preprocess_text(self, text: str) -> list[str]:
        """Applies light or heavy preprocessing according to this text's length."""
        mode = is_short_text(text, self.config)
        return self._preprocess_mode(text, mode)

    def _preprocess_mode(self, text: str, short_text_mode: bool) -> list[str]:
        text_config = dict(self.config)
        text_config["preprocessing"] = dict(self.config.get("preprocessing", {}))
        text_config["preprocessing"]["short_text_mode"] = short_text_mode
        return preprocess(text, text_config)

    def _light_features(self, text: str) -> list[str]:
        """Produces light tokens and bigrams for every document and query."""
        tokens = self._preprocess_mode(text, True)
        bigrams = [f"{left}__{right}" for left, right in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _heavy_features(self, text: str) -> list[str]:
        """Produces heavy tokens for every document and query."""
        return self._preprocess_mode(text, False)

    def _char_text(self, text: str) -> str:
        """Returns the raw text used by the corpus-wide char space."""
        return text

    def fit(self, corpus_df):
        self.doc_ids = corpus_df['qid'].astype(str).tolist()
        self.doc_texts = corpus_df['question'].astype(str).tolist()
        self.light_matrix = self.light_vectorizer.fit_transform(self.doc_texts)
        self.heavy_matrix = self.heavy_vectorizer.fit_transform(self.doc_texts)
        self.char_tfidf_matrix = self.char_vectorizer.fit_transform(self.doc_texts)

    def _rank_scores(self, vectorizer, matrix, query: str, pool_size: int) -> list[tuple[str, float]]:
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, matrix).flatten()
        order = np.argsort(similarities)[::-1]
        return [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in order[:pool_size]
            if similarities[idx] > 0 and self.doc_texts[idx].strip() != query.strip()
        ]
        
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        pool_size = max(100, top_k)
        result_lists = [
            self._rank_scores(self.light_vectorizer, self.light_matrix, query, pool_size),
            self._rank_scores(self.heavy_vectorizer, self.heavy_matrix, query, pool_size),
        ]
        if is_short_text(query, self.config):
            result_lists.append(self._rank_scores(self.char_vectorizer, self.char_tfidf_matrix, query, pool_size))
        rrf_k = self.config.get("retrieval", {}).get("rrf_k", 60)
        return reciprocal_rank_fusion(result_lists, top_k, rrf_k)
