from abc import ABC, abstractmethod
from typing import List

class NLPProcessorPort(ABC):
    @abstractmethod
    def process_text(self, text: str) -> List[str]:
        """Process the input text and return a list of processed tokens."""
        pass