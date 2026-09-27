import pandas as pd
from src.data.build_corpus import process_data

def test_process_data():
    """Test process_data creates proper corpus, qrels, and eval sets."""
    df = pd.DataFrame({
        "qid1": [1, 2, 3],
        "qid2": [4, 5, 1],
        "question1": ["q1", "q2", "q3"],
        "question2": ["q4", "q5", "q1"],
        "is_duplicate": [1, 0, 1]
    })
    
    corpus_df, qrels_df, eval_queries_df = process_data(df, eval_query_count=2, seed=42)
    
    # Corpus should contain unique questions: 1, 2, 3, 4, 5 -> 5 unique qids
    assert len(corpus_df) == 5
    assert set(corpus_df["qid"]) == {1, 2, 3, 4, 5}
    
    # Qrels should come from duplicates: (1,4), (4,1), (3,1), (1,3)
    assert len(qrels_df) == 4
    qrels_set = set(zip(qrels_df["query_id"], qrels_df["doc_id"]))
    assert (1, 4) in qrels_set
    assert (4, 1) in qrels_set
    assert (3, 1) in qrels_set
    assert (1, 3) in qrels_set
    assert (2, 5) not in qrels_set
    
    # Eval queries should be exactly 2 queries out of the ones with qrels
    assert len(eval_queries_df) == 2
    assert all(qid in {1, 3, 4} for qid in eval_queries_df["qid"])
