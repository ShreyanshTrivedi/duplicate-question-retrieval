from collections import defaultdict
from src.retrieval.base import RetrieverBase


def reciprocal_rank_fusion(result_lists: list[list[tuple[str, float]]], top_k: int, rrf_k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked result lists using reciprocal rank fusion."""
    rrf_scores = defaultdict(float)
    for results in result_lists:
        for rank_idx, (doc_id, _) in enumerate(results, start=1):
            rrf_scores[doc_id] += 1.0 / (rrf_k + rank_idx)
    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_results[:top_k]


class HybridRRFRetriever(RetrieverBase):
    """Combines multiple retrievers using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, retrievers: list[RetrieverBase], config: dict):
        """
        Args:
            retrievers (list[RetrieverBase]): List of instantiated retrievers.
            config (dict): Project configuration.
        """
        self.retrievers = retrievers
        self.rrf_k = config.get("retrieval", {}).get("rrf_k", 60)
        
    def fit(self, corpus_df):
        """Fits all underlying retrievers."""
        for r in self.retrievers:
            if hasattr(r, 'fit'):
                r.fit(corpus_df)
                
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Ranks documents using RRF across all underlying retrievers.
        
        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.
            
        Returns:
            list[tuple[str, float]]: List of (doc_id, rrf_score) pairs.
        """
        # Fetch more candidates from each retriever to ensure high overlap in RRF scoring
        pool_size = max(100, top_k)
        
        result_lists = [retriever.rank(query, pool_size) for retriever in self.retrievers]
        return reciprocal_rank_fusion(result_lists, top_k, self.rrf_k)
