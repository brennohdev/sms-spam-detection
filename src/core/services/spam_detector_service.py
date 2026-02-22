import time
from src.core.domain.entities.sms_message import SMSMessage
from src.core.domain.entities.spam_prediction import SpamPrediction
from src.core.ports.nlp_processor import NLPProcessorPort
from src.core.ports.model_port import ModelPort

class SpamDetectionService:
    """
    Service class responsible for orchestrating the spam detection process.
    """
    def __init__(self, nlp_port: NLPProcessorPort, model_port: ModelPort):
        self.nlp = nlp_port
        self.model = model_port
        
    def detect(self, message: SMSMessage) -> SpamPrediction:
        start_time = time.time()
        
        tokens = self.nlp.process_text(message.text)
        prediction = self.model.predict(tokens)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        prediction.processing_time_ms = latency_ms
        
        return prediction
