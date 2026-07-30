from app.services.face_recognition_module import FaceRecognizer
from app.services.cv_service import ProductClassifier
from app.services.nlp_service import SentimentAnalyzer
from app.services.chatbot_service import ChatbotService

class UnifiedPipeline:
    def __init__(self):
        print("Initializing Unified ML Pipeline and loading models...")
        self.face_recognizer = FaceRecognizer()
        self.product_classifier = ProductClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.chatbot = ChatbotService()
        self.sentiment_stats = {"positive": 0, "negative": 0, "neutral": 0}
        self.chatbot_usage_count = 0
        print("Unified ML Pipeline initialized successfully. All models loaded.")

# Singleton instance for global import
pipeline = UnifiedPipeline()
