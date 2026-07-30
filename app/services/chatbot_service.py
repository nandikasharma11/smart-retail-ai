import os
import pickle
import random
from typing import Tuple, Dict, Any, List
from app.services.nlp_service import preprocess_text

class ChatbotService:
    def __init__(self, model_path: str = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/app/models/chatbot_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.vectorizer = None
        self.intents = []
        self.strict_rules = {}
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    chatbot_data = pickle.load(f)
                self.model = chatbot_data['model']
                self.vectorizer = chatbot_data['vectorizer']
                self.intents = chatbot_data['intents']
                self.strict_rules = chatbot_data['strict_rules']
                print("Chatbot model loaded successfully.")
            except Exception as e:
                print(f"Error loading chatbot model: {e}")
        else:
            print("Chatbot model not found. Chatbot will run with fallback responses only.")

    def get_response(self, user_query: str, threshold: float = 0.35) -> Tuple[str, str, float]:
        """Returns the bot response, predicted intent tag, and confidence score."""
        if not user_query or not user_query.strip():
            return "Please say something!", "empty", 1.0

        query_lower = user_query.lower().strip()
        
        # 1. Rule-based check
        for keyword, tag in self.strict_rules.items():
            if keyword in query_lower:
                for intent in self.intents:
                    if intent['tag'] == tag:
                        return random.choice(intent['responses']), tag, 1.0
                        
        # 2. ML Classifier Fallback
        if self.model is None or self.vectorizer is None:
            return ("I am sorry, I am currently undergoing maintenance. Please email support@smartretail.com.", 
                    "error", 0.0)
            
        cleaned = preprocess_text(user_query)
        vec = self.vectorizer.transform([cleaned])
        
        pred_tag = self.model.predict(vec)[0]
        probs = self.model.predict_proba(vec)[0]
        class_idx = list(self.model.classes_).index(pred_tag)
        confidence = float(probs[class_idx])
        
        if confidence >= threshold:
            for intent in self.intents:
                if intent['tag'] == pred_tag:
                    return random.choice(intent['responses']), pred_tag, confidence
                    
        return ("I am sorry, I didn't quite get that. Could you please rephrase or contact support at support@smartretail.com?", 
                "fallback", confidence)
