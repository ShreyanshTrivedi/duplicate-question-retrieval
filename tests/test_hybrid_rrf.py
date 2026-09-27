from src.retrieval.hybrid_rrf import HybridRRFRetriever
from src.retrieval.base import RetrieverBase

class MockRetriever(RetrieverBase):
    def __init__(self, ranked_list):
        self.ranked_list = ranked_list
        
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.ranked_list[:top_k]

def test_rrf_math():
    config = {"retrieval": {"rrf_k": 60}}
    
    # R1 ranks: D1 (1), D2 (2), D3 (3)
    r1 = MockRetriever([("D1", 0.9), ("D2", 0.8), ("D3", 0.7)])
    
    # R2 ranks: D3 (1), D1 (2), D4 (3)
    r2 = MockRetriever([("D3", 10.0), ("D1", 5.0), ("D4", 2.0)])
    
    hybrid = HybridRRFRetriever([r1, r2], config)
    
    results = hybrid.rank("dummy query", top_k=3)
    
    # Expected scores:
    # D1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # D2: 1/(60+2) = 1/62 = 0.016129
    # D3: 1/(60+3) + 1/(60+1) = 1/63 + 1/61 = 0.015873 + 0.016393 = 0.032266
    # D4: 1/(60+3) = 1/63 = 0.015873
    
    # Sort order: D1, D3, D2, D4
    # top 3: D1, D3, D2
    
    doc_ids = [res[0] for res in results]
    assert doc_ids == ["D1", "D3", "D2"]
    
    # verify exact math on D1
    expected_d1_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert abs(results[0][1] - expected_d1_score) < 1e-6
