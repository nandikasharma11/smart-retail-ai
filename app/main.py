import os
import csv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.routers import vision, nlp, chatbot
from app.schemas import DashboardStatsResponse
from app.services.pipeline import pipeline

app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform",
    description="An AI-powered retail analytics and FAQ chatbot platform.",
    version="1.0.0"
)

# Include routers
app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the main interactive dashboard UI page."""
    html_path = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/app/static/index.html'
    if os.path.exists(html_path):
        try:
            with open(html_path, 'r') as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        except Exception as e:
            return HTMLResponse(content=f"<h3>Error loading UI: {str(e)}</h3>", status_code=500)
    return HTMLResponse(content="<h3>Smart Retail Dashboard HTML not found.</h3>", status_code=404)

@app.get("/dashboard/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats():
    """Aggregates and returns real-time visits and customer sentiment statistics."""
    log_path = '/Users/nandikasharma/smart-retail-ai/smart-retail-ai/data/customer_visits.csv'
    recent_visits = []
    total_visits = 0
    
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                total_visits = len(rows)
                # Get the last 5 visits
                for row in reversed(rows[-5:]):
                    recent_visits.append({
                        "name": row.get('name', 'Unknown'),
                        "timestamp": row.get('timestamp', '')
                    })
        except Exception as e:
            print(f"Error reading customer visits log: {e}")
            
    return DashboardStatsResponse(
        total_visits=total_visits,
        recent_visits=recent_visits,
        sentiment_distribution=pipeline.sentiment_stats,
        chatbot_usage=pipeline.chatbot_usage_count
    )

@app.get("/api/status")
async def get_api_status():
    return {"status": "ok"}
