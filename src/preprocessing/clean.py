import string
import copy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def is_short_text(text: str, config: dict) -> bool:
    """Returns whether text is short using the project's whitespace-word rule."""
    if not isinstance(text, str):
        return False
    threshold = config.get("preprocessing", {}).get("short_text_word_threshold", 5)
    return len(text.split()) <= threshold


def preprocess_for_retrieval(text: str, config: dict) -> list[str]:
    """Preprocess one text with the lexical mode selected from its own length.

    Short text retains stopwords and surface forms to avoid one-token collapse.
    Long text uses the original stopword-removal and stemming pipeline.
    """
    text_config = copy.deepcopy(config)
    text_config.setdefault("preprocessing", {})["short_text_mode"] = is_short_text(text, config)
    return preprocess(text, text_config)

def preprocess(text: str, config: dict) -> list[str]:
    """Preprocesses a string into a list of tokens based on the provided configuration.
    
    Args:
        text (str): The input text to preprocess.
        config (dict): The configuration dictionary (should contain 'preprocessing' key).
        
    Returns:
        list[str]: The list of preprocessed tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []
        
    prep_config = config.get("preprocessing", {})
    
    if prep_config.get("lowercase", True):
        text = text.lower()
        
    if prep_config.get("remove_punct", True):
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(translator)
        
    tokens = word_tokenize(text)
    
    if prep_config.get("remove_stopwords", True) and not prep_config.get("short_text_mode", False):
        tokens = [t for t in tokens if t not in stop_words]
        
    if prep_config.get("stem", True) and not prep_config.get("short_text_mode", False):
        tokens = [stemmer.stem(t) for t in tokens]
        
    return tokens
