"""Comprehensive verification of the short-text bug fix."""
import sys, os, yaml, copy
sys.path.insert(0, os.path.abspath("."))

from src.preprocessing.clean import preprocess

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Build the light config the same way tfidf.py does
light_config = copy.deepcopy(config)
if "preprocessing" not in light_config:
    light_config["preprocessing"] = {}
light_config["preprocessing"]["short_text_mode"] = True

print("=" * 60)
print("STEP 1: Verify light preprocessing mode")
print("=" * 60)
test_queries = [
    "What is the google?",
    "How do I post here?",
    "When are you most creative?"
]
for q in test_queries:
    heavy = preprocess(q, config)
    light = preprocess(q, light_config)
    print(f"  Query: {q!r}")
    print(f"    Heavy (old): {heavy}")
    print(f"    Light (new): {light}")
    print()

print("=" * 60)
print("STEP 3: Run 6 problem queries through TfidfRetriever")
print("=" * 60)

import pandas as pd
from src.retrieval.tfidf import TfidfRetriever

corpus_df = pd.read_parquet("data/processed/corpus.parquet")
id_to_q = dict(zip(corpus_df["qid"].astype(str), corpus_df["question"]))

print("Fitting TF-IDF on full corpus... (this takes ~60s)")
tfidf = TfidfRetriever(config)
tfidf.fit(corpus_df)
print("Done fitting.\n")

problem_queries = [
    "What is the google?",
    "What irritates you more?",
    "How do I post here?",
    "Who is the Yeti?",
    "What is mechatronics?",
    "When are you most creative?"
]

any_bug = False
for q in problem_queries:
    results = tfidf.rank(q, top_k=10)
    scores = [s for _, s in results]
    exact_ones = sum(1 for s in scores if abs(s - 1.0) < 1e-6)
    if exact_ones > 1:
        any_bug = True
    print(f"Query: {q!r}  (exact 1.0 ties: {exact_ones})")
    for i, (did, sc) in enumerate(results, 1):
        txt = id_to_q.get(did, "N/A")[:80]
        print(f"  {i:2d}. {sc:.6f}  {txt}")
    print()

print(">>> CORE BUG STATUS:", "STILL BROKEN" if any_bug else "FIXED")
print()

print("=" * 60)
print("STEP 4: Run 6 queries through BM25Retriever")
print("=" * 60)

from src.retrieval.bm25 import BM25Retriever
print("Fitting BM25... (this takes ~20s)")
bm25 = BM25Retriever(config)
bm25.fit(corpus_df)
print("Done fitting.\n")

for q in problem_queries:
    results = bm25.rank(q, top_k=5)
    print(f"Query: {q!r}")
    for i, (did, sc) in enumerate(results, 1):
        txt = id_to_q.get(did, "N/A")[:80]
        print(f"  {i:2d}. {sc:.4f}  {txt}")
    print()

print("VERIFICATION COMPLETE")
