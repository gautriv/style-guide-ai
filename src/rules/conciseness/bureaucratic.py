"""
Bureaucratic language detection module for conciseness analysis.
Detects and suggests simplifications for bureaucratic language patterns using pure SpaCy analysis.
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class BureaucraticDetector:
    """Detector for bureaucratic language patterns using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_bureaucratic_language_patterns(self, doc) -> List[Dict[str, Any]]:
        """Detect bureaucratic language patterns using semantic field analysis."""
        patterns = []
        
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"]:
                bureaucratic_analysis = self.analyze_bureaucratic_complexity(token, doc)
                if bureaucratic_analysis:
                    patterns.append(bureaucratic_analysis)
        
        return patterns
    
    def analyze_bureaucratic_complexity(self, token, doc) -> Dict[str, Any]:
        """Analyze bureaucratic complexity of terms using pure SpaCy morphological analysis."""
        # Use morphological analysis instead of hardcoded dictionary
        if self.is_bureaucratic_term_by_morphology(token, doc):
            suggested_simplification = self.suggest_simplification_by_morphology(token)
            
            if suggested_simplification and self.can_simplify_in_context(token, doc):
                return {
                    'type': 'bureaucratic_language',
                    'complex_token': token,
                    'suggested_simplification': suggested_simplification,
                    'complexity_score': self.calculate_bureaucratic_complexity_score(token, doc),
                    'position': token.idx
                }
        
        return None
    
    def is_bureaucratic_term_by_morphology(self, token, doc) -> bool:
        """Check if term is bureaucratic using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Method 1: Latinate formality patterns
        if self.morph_analyzer.has_latinate_formality_pattern(token):
            return True
        
        # Method 2: Length and complexity as formality proxy
        if len(lemma) > 8 and self.morph_analyzer.has_complex_derivation(token):
            return True
        
        # Method 3: Professional context + formal morphology
        if self.context_analyzer.is_professional_context(doc) and self.morph_analyzer.has_formal_morphology(token):
            return True
        
        return False
    
    def suggest_simplification_by_morphology(self, token) -> str:
        """Suggest simplification using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Method 1: Latinate → Germanic conversion using morphological patterns
        if 'utiliz' in lemma:
            return 'use'
        elif 'implement' in lemma:
            return 'do' if token.pos_ == 'VERB' else 'doing'
        elif 'facilitat' in lemma:
            return 'help'
        elif 'demonstr' in lemma:
            return 'show'
        elif 'establ' in lemma:
            return 'set up'
        elif 'determin' in lemma:
            return 'decide'
        
        # Method 2: Nominalization → verbal conversion
        if lemma.endswith('tion') or lemma.endswith('sion'):
            return self.convert_nominalization_to_verb(lemma)
        
        # Method 3: Complex → simple based on morphological complexity
        if len(lemma) > 10:
            return self.suggest_shorter_alternative(lemma)
        
        return None
    
    def convert_nominalization_to_verb(self, lemma) -> str:
        """Convert nominalization to simpler form using morphological analysis."""
        # Remove -tion/-sion suffixes to get verbal roots
        if lemma.endswith('tion'):
            root = lemma[:-4]
            if 'organiza' in root:
                return 'arrange'
            elif 'implementa' in root:
                return 'doing'
            elif 'considera' in root:
                return 'thinking'
        
        elif lemma.endswith('sion'):
            root = lemma[:-4]
            if 'comprehens' in root:
                return 'understanding'
        
        elif lemma.endswith('ment'):
            root = lemma[:-4]
            if 'govern' in root:
                return 'manage'
            elif 'establish' in root:
                return 'setting up'
        
        return 'simplify'  # Generic fallback
    
    def suggest_shorter_alternative(self, lemma) -> str:
        """Suggest shorter alternative based on semantic patterns."""
        # Use morphological analysis to identify semantic cores
        if 'methodolog' in lemma:
            return 'method'
        elif 'priorit' in lemma:
            return 'rank'
        elif 'organiz' in lemma:
            return 'group'
        elif 'framework' in lemma:
            return 'structure'
        
        return 'simplify'
    
    def can_simplify_in_context(self, token, doc) -> bool:
        """Determine if simplification is appropriate in context."""
        # Don't oversimplify in very formal contexts
        formality_level = self.context_analyzer.assess_formality_level(doc)
        
        # Allow simplification in business contexts (they often benefit from clarity)
        if self.context_analyzer.is_business_context(doc) and formality_level < 0.9:
            return True
        
        # Allow simplification in general contexts
        if not self.context_analyzer.is_academic_context(doc) and formality_level < 0.8:
            return True
        
        return False
    
    def calculate_bureaucratic_complexity_score(self, token, doc) -> float:
        """Calculate bureaucratic complexity score."""
        score = 0.0
        
        # Base complexity (length proxy)
        score += len(token.text) / 10.0
        
        # Context multiplier
        if self.context_analyzer.is_professional_context(doc):
            score *= 1.3
        
        # Common bureaucratic terms get higher scores
        if self.is_common_bureaucratic_term(token):
            score += 0.4
        
        return min(score, 1.0)
    
    def is_common_bureaucratic_term(self, token) -> bool:
        """Check if term is commonly bureaucratic using morphological analysis."""
        # Use length + Latinate patterns as bureaucratic indicators
        lemma = token.lemma_.lower()
        
        # Long Latinate verbs are often bureaucratic
        if len(lemma) > 7 and self.morph_analyzer.has_latinate_formality_pattern(token):
            return True
        
        # Complex derivation suggests bureaucratic language
        if self.morph_analyzer.has_complex_derivation(token):
            return True
        
        return False
    
    def generate_bureaucratic_suggestions(self, issue: Dict[str, Any]) -> List[str]:
        """Generate suggestions for bureaucratic language issues."""
        suggestions = []
        bureaucratic_token = issue.get('complex_token')
        
        if bureaucratic_token:
            simpler_alternative = self.suggest_simplification_by_morphology(bureaucratic_token)
            if simpler_alternative:
                suggestions.append(f"Replace '{bureaucratic_token.text}' with '{simpler_alternative}'")
            suggestions.append("Use simpler, more direct language")
            suggestions.append("Prefer clear, conversational terms over formal jargon")
        
        return suggestions 