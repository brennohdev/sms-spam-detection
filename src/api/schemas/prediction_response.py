from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """
    What we send back to the user.
    """
    label: str
    is_spam: bool
    confidence: float
    model_version: str