import pytest
from unittest.mock import MagicMock
from src.core.services.spam_detector_service import SpamDetectionService
from src.core.domain.entities.sms_message import SMSMessage
from src.core.domain.entities.spam_prediction import SpamPrediction

def test_spam_detector_orchestration():
    # 1. ARRANGE: Prepare the environment
    # We create "fake" versions of our Ports
    mock_nlp = MagicMock()
    mock_model = MagicMock()
    
    # We define what the "fake" NLP should return when called
    mock_nlp.process_text.return_value = ["fake", "tokens"]
    
    # We define what the "fake" Model should return
    expected_prediction = SpamPrediction(
        label="ham", 
        confidence=0.99, 
        is_spam=False, 
        model_version="test-1.0"
    )
    mock_model.predict.return_value = expected_prediction

    # Initialize the service with our mocks (Dependency Injection)
    service = SpamDetectionService(nlp_port=mock_nlp, model_port=mock_model)
    
    # Create a dummy message
    message = SMSMessage(text="Hello, how are you?")

    # 2. ACT: Execute the logic
    result = service.detect(message)

    # 3. ASSERT: Verify the outcome
    # Check if the service actually called the ports with the right data
    mock_nlp.process_text.assert_called_once_with("Hello, how are you?")
    mock_model.predict.assert_called_once_with(["fake", "tokens"])
    
    # Verify the result is what we expected
    assert result.label == "ham"
    assert result.is_spam is False
    assert result.processing_time_ms is not None  # Ensure latency was calculated