import pytest
from unittest.mock import MagicMock
from src.core.services.spam_detector_service import SpamDetectionService
from src.adapters.nlp.nltk_processor_adapter import NLTKProcessorAdapter
from src.core.domain.entities.sms_message import SMSMessage
from src.core.domain.entities.spam_prediction import SpamPrediction

def test_service_with_real_nlp_adapter():
    # 1. ARRANGE
    # Use the REAL NLTK adapter
    real_nlp_adapter = NLTKProcessorAdapter(extra_stop_words=["claim"])
    
    # We still mock the model for now
    mock_model = MagicMock()
    mock_model.predict.return_value = SpamPrediction(
        label="spam", confidence=0.85, is_spam=True
    )

    # Inject the REAL adapter into the service
    service = SpamDetectionService(nlp_port=real_nlp_adapter, model_port=mock_model)
    
    # A sentence that requires lemmatization and stopword removal
    message = SMSMessage(text="I am currently studying the best ways to claim prizes!")

    # 2. ACT
    result = service.detect(message)

    # 3. ASSERT
    # Let's verify if NLTK actually did its job through the service
    # We check what was sent to the model's 'predict' method
    called_tokens = mock_model.predict.call_args[0][0]
    
    # 'studying' should become 'study' (Lemmatization)
    # 'prizes' should become 'prize' (Lemmatization)
    # 'claim' should be removed (Extra stop words)
    # 'am', 'the', 'to' should be removed (Default stopwords)
    
    assert "study" in called_tokens
    assert "prize" in called_tokens
    assert "claim" not in called_tokens
    assert result.processing_time_ms > 0