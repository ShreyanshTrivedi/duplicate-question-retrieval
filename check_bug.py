import pandas as pd
import yaml
from src.preprocessing.clean import preprocess

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_parquet("data/processed/eval_queries.parquet")
short_queries = df[df['question'].str.split().str.len() <= 5]

print(f"Total short queries (<=5 words): {len(short_queries)}")
count_0_or_1 = 0

for _, row in short_queries.iterrows():
    q = row['question']
    tokens = preprocess(q, config)
    if len(tokens) <= 1:
        count_0_or_1 += 1
        if count_0_or_1 <= 10:
            print(f"Original: {q} -> Tokens: {tokens}")

print(f"Number of short queries collapsing to 0-1 tokens: {count_0_or_1} out of {len(short_queries)}")
