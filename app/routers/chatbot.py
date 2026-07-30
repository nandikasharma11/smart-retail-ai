from fastapi import APIRouter, HTTPException
from app.services.pipeline import pipeline
from app.schemas import ChatbotRequest, ChatbotResponse

router = APIRouter(tags=["chatbot"])

@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot(payload: ChatbotRequest):
    """Interacts with the hybrid retail chatbot (FAQ rule matching + ML fallback classifier)."""
    try:
        response, tag, conf = pipeline.chatbot.get_response(payload.message)
        pipeline.chatbot_usage_count += 1
        return ChatbotResponse(response=response, tag=tag, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
