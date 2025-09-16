import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Clean input text by removing URLs, non-letters, stopwords."""
    text = re.sub(r"http\S+", "", str(text))  
    text = re.sub(r"[^a-zA-Z]", " ", text)  
    text = text.lower()
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text
