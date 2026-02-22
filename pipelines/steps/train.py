from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.adapters.nlp.tokenizer import custom_tokenizer

class ModelTrainer:
    """The 'Chef' of the pipeline. Takes ingredients and makes the model."""
    
    def execute(self, X_train, y_train, params: dict) -> Pipeline:
        print(f" Initializing training with params: {params}")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                tokenizer=custom_tokenizer, 
                token_pattern=None, 
                ngram_range=params.get('ngram_range', (1, 2))
            )),
            ('clf', MultinomialNB(alpha=params.get('alpha', 0.01)))
        ])
        
        pipeline.fit(X_train, y_train)
        return pipeline