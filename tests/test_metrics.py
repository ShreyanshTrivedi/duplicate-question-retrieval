from src.evaluation.metrics import precision_at_k, recall_at_k, average_precision, ndcg_at_k


def test_precision_at_k():
    """Test precision@k with known ranked list."""
    ranked = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "E"}
    assert precision_at_k(ranked, relevant, 1) == 1.0     # A is relevant
    assert precision_at_k(ranked, relevant, 2) == 0.5     # 1 of 2
    assert precision_at_k(ranked, relevant, 5) == 0.6     # 3 of 5
    assert precision_at_k(ranked, relevant, 0) == 0.0


def test_recall_at_k():
    """Test recall@k with known ranked list."""
    ranked = ["A", "B", "C", "D", "E"]
    relevant = {"A", "C", "E"}
    assert recall_at_k(ranked, relevant, 1) == 1 / 3
    assert recall_at_k(ranked, relevant, 3) == 2 / 3      # A, C found
    assert recall_at_k(ranked, relevant, 5) == 1.0
    assert recall_at_k(ranked, set(), 5) == 0.0            # no relevant docs


def test_average_precision():
    """Test AP with known ranked list."""
    ranked = ["A", "B", "C"]
    relevant = {"A", "C"}
    # hit A at pos 1 -> precision 1/1, hit C at pos 3 -> precision 2/3
    # AP = (1.0 + 2/3) / 2 = 5/6
    ap = average_precision(ranked, relevant)
    assert abs(ap - 5 / 6) < 1e-9
    assert average_precision(ranked, set()) == 0.0


def test_ndcg_at_k():
    """Test nDCG@k with known ranked list."""
    ranked = ["A", "B", "C"]
    relevant = {"A", "C"}
    # DCG: 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG: 1/log2(2) + 1/log2(3) ≈ 1.0 + 0.6309 = 1.6309
    ndcg = ndcg_at_k(ranked, relevant, 3)
    assert 0 < ndcg < 1.0
    # Perfect ranking
    ranked_perfect = ["A", "C", "B"]
    assert abs(ndcg_at_k(ranked_perfect, relevant, 3) - 1.0) < 1e-9
    assert ndcg_at_k(ranked, set(), 3) == 0.0
