from pydantic import BaseModel, Field
from typing import Dict, Any, List

class SentimentRequest(BaseModel):
    text: str = Field(..., description="The raw review text to analyze")

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="The predicted sentiment: positive, negative, or neutral")
    confidence: float = Field(..., description="Prediction confidence score")

class ChatbotRequest(BaseModel):
    message: str = Field(..., description="The message sent by the user")

class ChatbotResponse(BaseModel):
    response: str = Field(..., description="The generated response from the chatbot")
    tag: str = Field(..., description="The matched intent tag")
    confidence: float = Field(..., description="Classification confidence score")

class FaceRecognitionResponse(BaseModel):
    name: str = Field(..., description="Recognized customer name or Unknown / No Face Detected")
    confidence: float = Field(..., description="Confidence/similarity score")
    timestamp: str = Field(None, description="Visit logging timestamp if recognized")

class ProductClassificationResponse(BaseModel):
    category: str = Field(..., description="Predicted product category")
    confidence: float = Field(..., description="Prediction confidence score")

class DashboardStatsResponse(BaseModel):
    total_visits: int = Field(..., description="Total number of logged customer visits")
    recent_visits: List[Dict[str, str]] = Field(..., description="List of recent visits with timestamps")
    sentiment_distribution: Dict[str, int] = Field(..., description="Frequency of positive, negative, and neutral feedback")
    chatbot_usage: int = Field(..., description="Total chatbot messages processed")
