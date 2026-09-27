"""
Downloads the Quora Question Pairs dataset to the raw path specified in config.yaml.
"""
import os
import yaml
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QQP_URL = "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the project configuration.
    
    Args:
        config_path (str): Path to the yaml config file.
        
    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def download_data() -> None:
    """Downloads the QQP dataset and saves it to the configured raw path as CSV."""
    config = load_config()
    raw_path = config["data"]["raw_path"]
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    if os.path.exists(raw_path):
        logging.info(f"File already exists at {raw_path}. Skipping download.")
        return
        
    logging.info(f"Downloading QQP dataset from {QQP_URL}...")
    try:
        df = pd.read_csv(
            QQP_URL, 
            sep="\t",
            storage_options={'User-Agent': 'Mozilla/5.0'}
        )
        df.to_csv(raw_path, index=False)
        logging.info(f"Download completed and saved to {raw_path}.")
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise

if __name__ == "__main__":
    download_data()
