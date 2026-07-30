import os
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Tuple

# Download nltk resources on import
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    """Cleans raw text: lowercases, removes punctuation, tokenizes, removes stopwords, and lemmatizes."""
    if not isinstance(text, str):
        return ''
    # Lowercase
    text = text.lower()
    # Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Stopword removal and lemmatization
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(cleaned)

class SentimentAnalyzer:
    def __init__(self, 
                 model_path: str = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/app/models/sentiment_model.pkl',
                 vectorizer_path: str = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/app/models/vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.load_models()

    def load_models(self):
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Sentiment model and vectorizer loaded successfully.")
            except Exception as e:
                print(f"Error loading sentiment models: {e}")
        else:
            print("Sentiment models not found. Prediction will not work.")

    def predict(self, text: str) -> Tuple[str, float]:
        """Predicts sentiment label and returns confidence (probability)."""
        if self.model is None or self.vectorizer is None:
            return "unknown", 0.0
            
        cleaned = preprocess_text(text)
        vec = self.vectorizer.transform([cleaned])
        
        # Predict label
        pred_label = self.model.predict(vec)[0]
        
        # Predict probability
        probs = self.model.predict_proba(vec)[0]
        class_idx = list(self.model.classes_).index(pred_label)
        confidence = float(probs[class_idx])
        
        return pred_label, confidence
