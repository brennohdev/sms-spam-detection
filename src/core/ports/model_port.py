from abc import ABC, abstractmethod
from typing import List
from src.core.domain.entities.spam_prediction import SpamPrediction

class ModelPort(ABC):
    @abstractmethod
    def predict(self, processed_text: List[str]) -> SpamPrediction:
        """Predict whether the input text is spam or not."""
        pass