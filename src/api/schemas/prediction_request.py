from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """
    Schema for the prediction request.
    """
    text: str = Field(..., min_length=1, description="Claim your prize now!")
    
    