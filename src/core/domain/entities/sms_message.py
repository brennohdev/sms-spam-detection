from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from datetime import datetime

class SMSMessage(BaseModel):
    """
    Represents the input data from the user
    """
    id: UUID = Field(default_factory=uuid4)
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
