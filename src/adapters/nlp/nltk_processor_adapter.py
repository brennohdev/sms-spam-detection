import nltk
import re

from typing import List
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from src.core.ports.nlp_processor import NLPProcessorPort

class NLTKProcessorAdapter(NLPProcessorPort):
    def __init__(self, extra_stop_words: List[str] = None):
        self._ensure_resources()
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        if extra_stop_words:
            self.stop_words.update(extra_stop_words)
            
    
    def _ensure_resources(self):
        """Hidden detail: handles NLTK's specific data requirements."""
        resources = ['stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'punkt_tab']
        for res in resources:
            nltk.download(res, quiet=True)
            
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """The 'Bridge' logic we discussed earlier."""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN
    
    
    def process_text(self, text: str) -> List[str]:
        """The implementation of the contract."""
        # 1. Normalization
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # 2. Tokenization
        tokens = text.split()
        
        # 3. Stopword removal
        filtered = [w for w in tokens if w not in self.stop_words]
        
        # 4. POS-Aware Lemmatization
        pos_tags = nltk.pos_tag(filtered)
        return [
            self.lemmatizer.lemmatize(word, pos=self._get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]