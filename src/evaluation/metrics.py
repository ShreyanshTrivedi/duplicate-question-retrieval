import numpy as np

def precision_at_k(ranked_list: list[str], relevant_docs: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return hits / k

def recall_at_k(ranked_list: list[str], relevant_docs: set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    top_k = ranked_list[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_docs)
    return hits / len(relevant_docs)

def average_precision(ranked_list: list[str], relevant_docs: set[str]) -> float:
    if not relevant_docs:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(ranked_list):
        if doc_id in relevant_docs:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant_docs)

def ndcg_at_k(ranked_list: list[str], relevant_docs: set[str], k: int) -> float:
    if not relevant_docs:
        return 0.0
    
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        if doc_id in relevant_docs:
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(k, len(relevant_docs))):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
