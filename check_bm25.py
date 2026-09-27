import pandas as pd
import yaml
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

corpus_df = pd.read_parquet("data/processed/corpus.parquet")
id_to_question = dict(zip(corpus_df["qid"].astype(str), corpus_df["question"]))

bm25 = BM25Retriever(config)
bm25.fit(corpus_df)

queries = [
    "What is the google?",
    "What irritates you more?",
    "How do I post here?",
    "Who is the Yeti?",
    "What is mechatronics?",
    "When are you most creative?"
]

print("=== BM25 Results ===")
for q in queries:
    results = bm25.rank(q, top_k=10)
    print(f"\nQuery: {q}")
    for idx, (doc_id, score) in enumerate(results[:5]):
        text = id_to_question.get(doc_id, "N/A")
        print(f"  {idx+1}. Score: {score:.4f} | {text}")

# Briefly check Dense retriever tokenization dependency to confirm it uses raw text
print("\n=== Dense Retriever Dependency Check ===")
import inspect
from src.retrieval.dense import DenseRetriever
print(inspect.getsource(DenseRetriever.rank))
