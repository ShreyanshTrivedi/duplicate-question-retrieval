import pandas as pd
from src.retrieval.inverted_index import InvertedIndexRetriever
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

def test_retrievers(tmp_path):
    config = {
        "preprocessing": {
            "lowercase": True,
            "remove_punct": True,
            "remove_stopwords": True,
            "stem": True
        },
        "retrieval": {
            "dense_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        "data": {
            "embeddings_path": str(tmp_path / "embeddings.npy"),
        },
    }
    
    corpus_df = pd.DataFrame({
        "qid": [1, 2, 3, 4],
        "question": [
            "How do I learn Python fast?",
            "What is the best way to learn Python quickly?",
            "How do I bake a chocolate cake?",
            "Where can I buy a good laptop?"
        ]
    })
    
    query = "learn Python fast"
    
    retrievers = [
        InvertedIndexRetriever(config),
        TfidfRetriever(config),
        BM25Retriever(config),
        DenseRetriever(config)
    ]
    
    for r in retrievers:
        r.fit(corpus_df)
        results = r.rank(query, top_k=2)
        
        assert len(results) > 0
        doc_ids = [res[0] for res in results]
        assert "1" in doc_ids or "2" in doc_ids
        
        # Note: Dense retrievers always return top_k nearest neighbors,
        # so we don't test for empty results here.

    exact = "How do I learn Python fast?"
    for r in retrievers:
        results = r.rank(exact, top_k=3)
        doc_ids = [res[0] for res in results]
        assert "1" not in doc_ids
        assert len(doc_ids) == len(set(doc_ids))
        scores = [res[1] for res in results]
        assert scores == sorted(scores, reverse=True)
