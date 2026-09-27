import pytest
import pandas as pd
import yaml
from unittest.mock import Mock
from src.retrieval.tfidf import TfidfRetriever
from src.preprocessing.clean import is_short_text

def test_short_text_bug():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # We can create a tiny fake corpus or load the real one
    # The real one might be too slow for a unit test, but the prompt says 
    # "re-runs the 6 example queries ... through the FIXED pipeline"
    # Loading corpus is okay if we must, or we can just make a small one containing them.
    # Actually, the pipeline should be tested on a small dataframe to avoid memory overhead in tests.
    
    queries = [
        "What is the google?",
        "What irritates you more?",
        "How do I post here?",
        "Who is the Yeti?",
        "What is mechatronics?",
        "When are you most creative?"
    ]
    
    # Let's create a minimal DataFrame that has these queries and some others
    doc_ids = list(range(len(queries) * 2))
    docs = queries + ["A completely different long document about science"] * len(queries)
    
    df = pd.DataFrame({"qid": doc_ids, "question": docs})
    
    tfidf = TfidfRetriever(config)
    tfidf.fit(df)
    
    for q in queries:
        tokens = tfidf.preprocess_text(q)
        # 1. Assert > 1 surviving feature dimension
        assert len(tokens) > 1, f"Query '{q}' collapsed to {len(tokens)} tokens"
        
        # 2. Assert no exact 1.0 scores across non-identical documents
        results = tfidf.rank(q, top_k=5)
        scores = [score for doc_id, score in results]
        
        # Check if there are duplicate 1.0 scores
        exact_ones = [s for s in scores if abs(s - 1.0) < 1e-6]
        assert len(exact_ones) <= 1, f"Query '{q}' has multiple 1.0 scores: {scores}"


def test_scoped_preprocessing_keeps_long_text_heavy():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tfidf = TfidfRetriever(config)
    short_query = "What is the google?"
    long_query = "What are the best ways to organize running projects effectively?"

    short_tokens = tfidf.preprocess_text(short_query)
    long_tokens = tfidf.preprocess_text(long_query)

    assert is_short_text(short_query, config)
    assert not is_short_text(long_query, config)
    assert "what" in short_tokens and "is" in short_tokens and "the" in short_tokens
    assert "what" not in long_tokens and "are" not in long_tokens and "the" not in long_tokens
    assert "organ" in long_tokens
    assert "organize" not in long_tokens


def test_dual_spaces_retain_cross_length_duplicate_signal():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    query = "How is time travel possible?"
    duplicate = "Do you think time travel is possible?"
    corpus = pd.DataFrame({
        "qid": [1, 2, 3, 4],
        "question": [query, duplicate, "What is the capital of France?", "How do I bake a cake?"],
    })

    retriever = TfidfRetriever(config)
    retriever.fit(corpus)
    scores = dict(retriever.rank(query, top_k=4))

    assert scores["2"] > 0.01


def test_dual_vectorizers_fit_uniform_modes_once():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    short_text = "How is time travel possible?"
    long_text = "Do you think time travel is possible?"
    corpus = pd.DataFrame({
        "qid": [1, 2],
        "question": [short_text, long_text],
    })

    retriever = TfidfRetriever(config)
    retriever.light_vectorizer.fit_transform = Mock(wraps=retriever.light_vectorizer.fit_transform)
    retriever.heavy_vectorizer.fit_transform = Mock(wraps=retriever.heavy_vectorizer.fit_transform)
    retriever.fit(corpus)

    assert retriever.light_vectorizer.fit_transform.call_count == 1
    assert retriever.heavy_vectorizer.fit_transform.call_count == 1
    assert "is" in retriever.light_vectorizer.build_analyzer()(long_text)
    assert "organize" not in retriever.heavy_vectorizer.build_analyzer()(
        "What are the best ways to organize projects effectively?"
    )
    assert "organ" in retriever.heavy_vectorizer.build_analyzer()(
        "What are the best ways to organize projects effectively?"
    )
