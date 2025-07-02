"""
Archaic verbs detection module for conciseness analysis.
Detects archaic verb patterns and suggests modern alternatives using pure SpaCy analysis.
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class ArchaicVerbsDetector:
    """Detector for archaic verb patterns using pure SpaCy morphological analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_archaic_verb_patterns(self, doc) -> List[Dict[str, Any]]:
        """Detect archaic verb patterns that could be modernized."""
        archaic_patterns = []
        
        for token in doc:
            if token.pos_ == "VERB" and self.is_archaic_verb(token, doc):
                modern_suggestion = self.suggest_modern_alternative(token, doc)
                if modern_suggestion:
                    archaic_patterns.append({
                        'type': 'archaic_verb_pattern',
                        'verb_token': token,
                        'archaic_verb': token.lemma_,
                        'modern_suggestion': modern_suggestion,
                        'position': token.idx,
                        'formality_score': self.morph_analyzer.calculate_word_formality_by_morphology(token.lemma_)
                    })
        
        return archaic_patterns
    
    def is_archaic_verb(self, verb_token, doc) -> bool:
        """Check if verb is archaic/formal using pure SpaCy morphological analysis."""
        # Method 1: Morphological complexity analysis
        morphological_complexity = self.morph_analyzer.calculate_morphological_complexity_score(verb_token)
        if morphological_complexity > 0.7:  # High complexity suggests formality
            return True
        
        # Method 2: Latinate pattern analysis
        if self.morph_analyzer.is_latinate_verb_pattern(verb_token):
            # Check if context would benefit from simpler alternative
            context_formality = self.context_analyzer.assess_formality_level(doc)
            if context_formality < 0.7:  # Context is not highly formal
                return True
        
        # Method 3: Length-based formality proxy
        lemma = verb_token.lemma_.lower()
        if len(lemma) > 8:  # Long verbs often more formal
            syllable_count = self.morph_analyzer.estimate_syllable_count(lemma)
            if syllable_count > 3:  # Multi-syllabic suggests formality
                return True
        
        return False
    
    def suggest_modern_alternative(self, verb_token, doc) -> str:
        """Suggest modern alternative using SpaCy morphological analysis."""
        lemma = verb_token.lemma_.lower()
        
        # Use morphological analysis to suggest simpler alternatives
        if 'circul' in lemma:
            return 'share'
        elif 'utiliz' in lemma:
            return 'use'
        elif 'facilitat' in lemma:
            return 'help'
        elif 'commenc' in lemma:
            return 'start'
        elif 'terminat' in lemma:
            return 'end'
        elif 'guarant' in lemma:
            return 'ensure'
        
        # For other formal verbs, try to simplify based on morphological patterns
        if self.morph_analyzer.is_latinate_verb_pattern(verb_token):
            return self.suggest_germanic_alternative(lemma)
        
        return None
    
    def suggest_germanic_alternative(self, lemma) -> str:
        """Suggest Germanic alternative for Latinate verbs."""
        # Common Latinate → Germanic mappings based on semantic patterns
        if lemma.endswith('ate'):
            if 'demonstr' in lemma:
                return 'show'
            elif 'indic' in lemma:
                return 'show'
            elif 'estab' in lemma:
                return 'set up'
        
        elif lemma.endswith('ize'):
            if 'organ' in lemma:
                return 'arrange'
            elif 'real' in lemma:
                return 'see'
        
        return 'simplify'  # Generic fallback
    
    def generate_archaic_verb_suggestions(self, issue: Dict[str, Any]) -> List[str]:
        """Generate suggestions for archaic verb pattern issues."""
        suggestions = []
        archaic_verb = issue.get('archaic_verb', '')
        modern_suggestion = issue.get('modern_suggestion', '')
        
        if archaic_verb and modern_suggestion:
            suggestions.append(f"Replace '{archaic_verb}' with '{modern_suggestion}' for modern clarity")
            suggestions.append(f"'{modern_suggestion}' is more conversational than '{archaic_verb}'")
            suggestions.append("Modern business writing favors direct, simple verbs")
        
        return suggestions 