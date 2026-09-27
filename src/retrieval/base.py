from abc import ABC, abstractmethod

class RetrieverBase(ABC):
    @abstractmethod
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Ranks documents for a given query.
        
        Args:
            query (str): The search query.
            top_k (int): Number of top results to return.
            
        Returns:
            list[tuple[str, float]]: List of (doc_id, score) pairs.
        """
        pass
