from fastapi import FastAPI, Depends
from src.api.schemas.prediction_request import PredictionRequest
from src.api.schemas.prediction_response import PredictionResponse
from src.api.dependencies import get_spam_service
from src.core.services.spam_detector_service import SpamDetectionService
from src.core.domain.entities.sms_message import SMSMessage

app = FastAPI(
    title="SMS Spam Detection API",
    description="An API for detecting whether an SMS message is spam or not.",
    version="1.0.0"
)

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    service: SpamDetectionService = Depends(get_spam_service)
):
    domain_message = SMSMessage(text=request.text)
    
    prediction = service.detect(domain_message)
    
    return prediction