import pytest
from src.preprocessing.clean import preprocess

@pytest.fixture
def base_config():
    return {
        "preprocessing": {
            "lowercase": True,
            "remove_punct": True,
            "remove_stopwords": True,
            "stem": True
        }
    }

def test_empty_string(base_config):
    assert preprocess("", base_config) == []
    assert preprocess("   ", base_config) == []
    assert preprocess(None, base_config) == []

def test_numbers(base_config):
    tokens = preprocess("We have 123 apples and 456 oranges.", base_config)
    assert "123" in tokens
    assert "456" in tokens
    assert "appl" in tokens
    assert "orang" in tokens

def test_contractions():
    config_no_punct = {
        "preprocessing": {
            "lowercase": True,
            "remove_punct": False,
            "remove_stopwords": False,
            "stem": False
        }
    }
    # NLTK word_tokenize splits "don't" into "do" and "n't"
    assert preprocess("don't", config_no_punct) == ["do", "n't"]
    
    config_punct = {
        "preprocessing": {
            "lowercase": True,
            "remove_punct": True,
            "remove_stopwords": False,
            "stem": False
        }
    }
    # With punctuation replaced by space, "don't" becomes "don t" -> ["don", "t"]
    assert preprocess("don't", config_punct) == ["don", "t"]

def test_all_features(base_config):
    text = "The quick brown foxes are jumping over the lazy dogs!"
    tokens = preprocess(text, base_config)
    # the -> stopword
    # quick -> quick
    # brown -> brown
    # foxes -> fox
    # are -> stopword
    # jumping -> jump
    # over -> stopword (in NLTK english stopwords)
    # the -> stopword
    # lazy -> lazi (Porter stem of lazy)
    # dogs -> dog
    # ! -> punct removed
    assert tokens == ["quick", "brown", "fox", "jump", "lazi", "dog"]
