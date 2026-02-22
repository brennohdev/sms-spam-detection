from src.core.services.spam_detector_service import SpamDetectionService
from src.adapters.nlp.nltk_processor_adapter import NLTKProcessorAdapter
from src.adapters.models.sklearn_model_adapter import SklearnModelAdapter

_nlp_processor = NLTKProcessorAdapter()
_model_adapter = SklearnModelAdapter(model_path="models/spam_detector_v1.joblib")
_service = SpamDetectionService(nlp_port=_nlp_processor, model_port=_model_adapter)

def get_spam_service() -> SpamDetectionService:
    """
    Provides the fully initialized service to the API routes.
    """
    return _service

