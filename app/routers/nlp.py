from fastapi import APIRouter, HTTPException
from app.services.pipeline import pipeline
from app.schemas import SentimentRequest, SentimentResponse

router = APIRouter(tags=["nlp"])

@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    """Analyzes user text and returns sentiment (positive, negative, or neutral) with confidence score."""
    try:
        sentiment, conf = pipeline.sentiment_analyzer.predict(payload.text)
        # Update pipeline in-memory statistics
        if sentiment in pipeline.sentiment_stats:
            pipeline.sentiment_stats[sentiment] += 1
            
        return SentimentResponse(sentiment=sentiment, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
