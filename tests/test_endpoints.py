import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Smart Retail" in response.text


def test_analyze_sentiment():
    # Test positive sentiment
    response = client.post("/analyze-sentiment", json={"text": "I love shopping here, great service!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert "confidence" in data

    # Test negative sentiment
    response = client.post("/analyze-sentiment", json={"text": "Horrible shoes, they broke on day one."})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"

def test_chatbot():
    # Test rules
    response = client.post("/chatbot", json={"message": "hello bot"})
    assert response.status_code == 200
    data = response.json()
    assert data["tag"] == "greeting"
    assert "response" in data

    # Test ML classification fallback
    response = client.post("/chatbot", json={"message": "what are the hours of the store?"})
    assert response.status_code == 200
    data = response.json()
    assert data["tag"] == "store_hours"

def test_recognize_face():
    # Use lena.jpg as face upload
    image_path = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/data/lena.jpg'
    assert os.path.exists(image_path), "Test face image not found"
    
    with open(image_path, "rb") as f:
        response = client.post("/recognize-face", files={"file": ("lena.jpg", f, "image/jpeg")})
        
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Lena"
    assert "confidence" in data
    assert "timestamp" in data

def test_classify_product():
    # Use lena.jpg as a mock product image
    image_path = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/data/lena.jpg'
    assert os.path.exists(image_path), "Test product image not found"
    
    with open(image_path, "rb") as f:
        response = client.post("/classify-product", files={"file": ("lena.jpg", f, "image/jpeg")})
        
    assert response.status_code == 200
    data = response.json()
    assert "category" in data
    assert "confidence" in data

def test_dashboard_stats():
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_visits" in data
    assert "recent_visits" in data
    assert "sentiment_distribution" in data
    assert "chatbot_usage" in data
