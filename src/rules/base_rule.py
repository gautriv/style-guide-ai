"""
Base Rule Class - Abstract interface for all writing rules.
All rules must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import re

class BaseRule(ABC):
    """Abstract base class for all writing rules."""
    
    def __init__(self):
        self.rule_type = self._get_rule_type()
        self.severity_levels = ['low', 'medium', 'high']
    
    @abstractmethod
    def _get_rule_type(self) -> str:
        """Return the rule type identifier (e.g., 'passive_voice', 'sentence_length')."""
        pass
    
    @abstractmethod
    def analyze(self, text: str, sentences: List[str], nlp=None) -> List[Dict[str, Any]]:
        """
        Analyze text and return list of errors found.
        
        Args:
            text: Full text to analyze
            sentences: List of sentences
            nlp: SpaCy nlp object (optional)
            
        Returns:
            List of error dictionaries with structure:
            {
                'type': str,
                'message': str,
                'suggestions': List[str],
                'sentence': str,
                'sentence_index': int,
                'severity': str
            }
        """
        pass
    
    def _analyze_sentence_structure(self, sentence: str, nlp=None) -> Optional[object]:
        """Helper method to get SpaCy doc for a sentence."""
        if nlp:
            return nlp(sentence)
        return None
    
    def _estimate_syllables(self, word: str) -> int:
        """Fallback syllable estimation when syllables library is not available."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count
    
    def _token_to_dict(self, token) -> Dict[str, Any]:
        """Convert SpaCy token to JSON-serializable dictionary."""
        if token is None:
            return None
        
        try:
            return {
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'idx': token.idx,
                'i': token.i,
                'morphology': {k: v for k, v in token.morph.to_dict().items()} if hasattr(token.morph, 'to_dict') else str(token.morph)
            }
        except:
            # Fallback for any serialization issues
            return {
                'text': str(token),
                'lemma': getattr(token, 'lemma_', ''),
                'pos': getattr(token, 'pos_', ''),
                'tag': getattr(token, 'tag_', ''),
                'dep': getattr(token, 'dep_', ''),
                'idx': getattr(token, 'idx', 0),
                'i': getattr(token, 'i', 0),
                'morphology': {}
            }
    
    def _tokens_to_list(self, tokens) -> List[Dict[str, Any]]:
        """Convert list of SpaCy tokens to JSON-serializable list."""
        if not tokens:
            return []
        
        return [self._token_to_dict(token) for token in tokens if token is not None]
    
    def _make_serializable(self, data: Any) -> Any:
        """Recursively convert data structure to be JSON serializable."""
        if data is None:
            return None
        
        # Handle SpaCy tokens
        if hasattr(data, 'text') and hasattr(data, 'lemma_'):
            return self._token_to_dict(data)
        
        # Handle SpaCy MorphAnalysis objects
        if hasattr(data, '__iter__') and hasattr(data, 'get') and not isinstance(data, (str, dict, list, tuple)):
            try:
                return dict(data)
            except:
                return str(data)
        
        # Handle dictionaries
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                try:
                    result[key] = self._make_serializable(value)
                except Exception as e:
                    # If serialization fails, convert to string
                    result[key] = str(value)
            return result
        
        # Handle lists
        if isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        
        # Handle tuples
        if isinstance(data, tuple):
            return tuple(self._make_serializable(item) for item in data)
        
        # Handle sets (convert to list)
        if isinstance(data, set):
            return [self._make_serializable(item) for item in data]
        
        # Return primitive types as-is
        if isinstance(data, (str, int, float, bool)):
            return data
        
        # For unknown types, try to convert to string
        try:
            return str(data)
        except:
            return None
    
    def _create_error(self, sentence: str, sentence_index: int, message: str, 
                     suggestions: List[str], severity: str = 'medium', 
                     **extra_data) -> Dict[str, Any]:
        """Helper method to create standardized error dictionary."""
        error = {
            'type': self.rule_type,
            'message': message,
            'suggestions': suggestions,
            'sentence': sentence,
            'sentence_index': sentence_index,
            'severity': severity
        }
        
        # Convert extra_data to be JSON serializable and handle each item safely
        for key, value in extra_data.items():
            try:
                error[key] = self._make_serializable(value)
            except Exception as e:
                # If serialization fails, store error info instead
                error[key] = f"<serialization_error: {str(e)}>"
        
        return error 