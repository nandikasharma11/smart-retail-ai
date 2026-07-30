import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.pipeline import pipeline
from app.schemas import FaceRecognitionResponse, ProductClassificationResponse

router = APIRouter(tags=["vision"])

@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(file: UploadFile = File(...)):
    """Uploads a portrait image, detects the face, compares it to the registered database, and logs the visit."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file encoding.")
        
        name, conf = pipeline.face_recognizer.recognize_face(img)
        timestamp = None
        if name != "Unknown" and name != "No Face Detected":
            timestamp = pipeline.face_recognizer.log_visit(name)
            
        return FaceRecognitionResponse(name=name, confidence=conf, timestamp=timestamp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/classify-product", response_model=ProductClassificationResponse)
async def classify_product(file: UploadFile = File(...)):
    """Uploads a product image and classifies it into one of the 10 Fashion-MNIST categories."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file encoding.")
            
        category, conf = pipeline.product_classifier.predict(img)
        return ProductClassificationResponse(category=category, confidence=conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
