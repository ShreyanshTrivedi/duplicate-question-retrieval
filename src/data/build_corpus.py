"""
Builds the retrieval corpus and qrels from the raw QQP dataset.
"""
import os
import yaml
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the project configuration.
    
    Args:
        config_path (str): Path to the yaml config file.
        
    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def process_data(df: pd.DataFrame, eval_query_count: int, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Processes the raw QQP dataframe into corpus, qrels, and eval queries.
    
    Args:
        df (pd.DataFrame): Raw QQP dataframe.
        eval_query_count (int): Number of evaluation queries to hold out.
        seed (int): Random seed for sampling.
        
    Returns:
        tuple: (corpus_df, qrels_df, eval_queries_df)
    """
    # Drop rows with null values in necessary columns
    df = df.dropna(subset=["question1", "question2", "qid1", "qid2"])
    
    # 1. Build deduplicated corpus
    q1 = df[['qid1', 'question1']].rename(columns={'qid1': 'qid', 'question1': 'question'})
    q2 = df[['qid2', 'question2']].rename(columns={'qid2': 'qid', 'question2': 'question'})
    corpus_df = pd.concat([q1, q2]).drop_duplicates(subset=['qid'])
    
    # 2. Build qrels
    positives = df[df['is_duplicate'] == 1]
    qrels_1 = positives[['qid1', 'qid2']].rename(columns={'qid1': 'query_id', 'qid2': 'doc_id'})
    qrels_2 = positives[['qid2', 'qid1']].rename(columns={'qid2': 'query_id', 'qid1': 'doc_id'})
    qrels_df = pd.concat([qrels_1, qrels_2]).drop_duplicates()
    
    # 3. Hold out eval queries
    queries_with_qrels = qrels_df['query_id'].unique()
    np.random.seed(seed)
    
    sample_size = min(eval_query_count, len(queries_with_qrels))
    eval_query_ids = np.random.choice(queries_with_qrels, size=sample_size, replace=False)
    eval_queries_df = corpus_df[corpus_df['qid'].isin(eval_query_ids)]
    
    return corpus_df, qrels_df, eval_queries_df

def main() -> None:
    """Loads raw data, processes it, and saves to parquet files."""
    config = load_config()
    raw_path = config["data"]["raw_path"]
    corpus_path = config["data"]["corpus_path"]
    qrels_path = config["data"]["qrels_path"]
    eval_queries_path = config["data"]["eval_queries_path"]
    eval_query_count = config["data"]["eval_query_count"]
    seed = config.get("seed", 42)
    
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    
    logging.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    
    corpus_df, qrels_df, eval_queries_df = process_data(df, eval_query_count, seed)
    
    # Save to parquet
    logging.info("Saving processed datasets to parquet...")
    corpus_df.to_parquet(corpus_path, index=False)
    qrels_df.to_parquet(qrels_path, index=False)
    eval_queries_df.to_parquet(eval_queries_path, index=False)
    
    logging.info(f"Corpus size: {len(corpus_df)}")
    logging.info(f"Qrels count: {len(qrels_df)}")
    logging.info(f"Eval query count: {len(eval_queries_df)}")

if __name__ == "__main__":
    main()
