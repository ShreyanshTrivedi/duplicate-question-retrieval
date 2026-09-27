import yaml

def test_config_loads():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "data" in config
    assert "raw_path" in config["data"]
    assert config["data"]["raw_path"] == "data/raw/questions.csv"
