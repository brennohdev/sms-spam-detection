from src.adapters.nlp.nltk_processor_adapter import NLTKProcessorAdapter

# This is a global function for Joblib/Pickle to find
def custom_tokenizer(text: str):
    # We use the adapter here because it's part of the Infrastructure layer
    return NLTKProcessorAdapter().process_text(text)