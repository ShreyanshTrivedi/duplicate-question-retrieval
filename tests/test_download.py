import os
from src.data.download import load_config

def test_load_config():
    """Test that the configuration loads properly."""
    config = load_config("config.yaml")
    assert "data" in config
    assert "raw_path" in config["data"]
