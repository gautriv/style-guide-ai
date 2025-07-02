"""
Semantic appropriateness detection module for conciseness analysis.
Detects words that don't fit the semantic context using pure SpaCy morphological analysis.
Example: 'simplifications' in quality assurance context should be 'validations'
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class SemanticAppropriatenessDetector:
    """Detector for semantically inappropriate word choices using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_semantic_appropriateness_issues(self, doc) -> List[Dict[str, Any]]:
        """Detect semantically inappropriate words using SpaCy morphological and contextual analysis."""
        issues = []
        
        # Analyze each content word for semantic fit
        for token in doc:
            if self._is_content_word(token):
                semantic_analysis = self._analyze_semantic_fit(token, doc)
                if semantic_analysis['is_inappropriate']:
                    issues.append({
                        'type': 'semantic_appropriateness',
                        'inappropriate_token': token,
                        'context_mismatch': semantic_analysis['context_mismatch'],
                        'suggested_alternatives': semantic_analysis['suggested_alternatives'],
                        'confidence_score': semantic_analysis['confidence_score'],
                        'position': token.idx
                    })
        
        return issues
    
    def _is_content_word(self, token) -> bool:
        """Check if token is content word using SpaCy POS analysis."""
        return token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and not token.is_stop
    
    def _analyze_semantic_fit(self, token, doc) -> Dict[str, Any]:
        """Analyze if word fits semantic context using SpaCy morphological analysis."""
        analysis = {
            'is_inappropriate': False,
            'context_mismatch': '',
            'suggested_alternatives': [],
            'confidence_score': 0.0
        }
        
        # Identify sentence context
        sentence_context = self._identify_sentence_semantic_context(token.sent, doc)
        
        # Check for specific context mismatches
        if sentence_context['domain'] == 'quality_assurance':
            if self._is_process_word_in_qa_context(token):
                mismatch = self._analyze_qa_context_mismatch(token, sentence_context)
                if mismatch['is_mismatch']:
                    analysis.update(mismatch)
        
        elif sentence_context['domain'] == 'technical_analysis':
            if self._is_inappropriate_for_technical_context(token, sentence_context):
                mismatch = self._analyze_technical_context_mismatch(token, sentence_context)
                if mismatch['is_mismatch']:
                    analysis.update(mismatch)
        
        return analysis
    
    def _identify_sentence_semantic_context(self, sentence, doc) -> Dict[str, Any]:
        """Identify semantic context of sentence using SpaCy analysis."""
        context = {
            'domain': 'general',
            'semantic_field': 'unknown',
            'key_concepts': [],
            'verb_patterns': []
        }
        
        # Method 1: Quality assurance context detection
        qa_indicators = ['error', 'catch', 'check', 'validate', 'quality', 'assurance', 'test']
        qa_count = sum(1 for token in sentence if any(indicator in token.lemma_.lower() for indicator in qa_indicators))
        
        if qa_count >= 2:
            context['domain'] = 'quality_assurance'
            context['semantic_field'] = 'verification_validation'
        
        # Method 2: Technical analysis context detection  
        tech_indicators = ['analyze', 'data', 'trend', 'result', 'performance', 'report']
        tech_count = sum(1 for token in sentence if any(indicator in token.lemma_.lower() for indicator in tech_indicators))
        
        if tech_count >= 2:
            context['domain'] = 'technical_analysis'
            context['semantic_field'] = 'data_analysis'
        
        # Method 3: Extract key concepts using NER and morphology
        for token in sentence:
            if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 4:
                context['key_concepts'].append(token.lemma_.lower())
        
        # Method 4: Extract verb patterns
        for token in sentence:
            if token.pos_ == "VERB":
                context['verb_patterns'].append(token.lemma_.lower())
        
        return context
    
    def _is_process_word_in_qa_context(self, token) -> bool:
        """Check if word relates to processes in QA context."""
        lemma = token.lemma_.lower()
        
        # Process-related words that might be misused in QA context
        process_patterns = ['simplif', 'process', 'procedur', 'method', 'approach']
        return any(pattern in lemma for pattern in process_patterns)
    
    def _analyze_qa_context_mismatch(self, token, context) -> Dict[str, Any]:
        """Analyze quality assurance context mismatch using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Specific case: "simplifications" in QA context
        if 'simplif' in lemma and context['domain'] == 'quality_assurance':
            return {
                'is_inappropriate': True,
                'is_mismatch': True,
                'context_mismatch': 'process_term_in_qa_context',
                'suggested_alternatives': ['validations', 'checks', 'quality assurance measures', 'verifications'],
                'confidence_score': 0.85,
                'explanation': f"'{token.text}' refers to making things simpler, but context suggests quality validation activities"
            }
        
        return {'is_mismatch': False}
    
    def _is_inappropriate_for_technical_context(self, token, context) -> bool:
        """Check if word is inappropriate for technical context."""
        lemma = token.lemma_.lower()
        
        # Check for overly casual terms in formal technical context
        if context.get('formality_level', 0.5) > 0.7:
            casual_patterns = ['stuff', 'thing', 'bunch', 'kinda', 'sorta']
            return any(pattern in lemma for pattern in casual_patterns)
        
        return False
    
    def _analyze_technical_context_mismatch(self, token, context) -> Dict[str, Any]:
        """Analyze technical context mismatch."""
        # Implementation for technical context mismatches
        return {'is_mismatch': False}
    
    def generate_semantic_appropriateness_suggestions(self, issue: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for semantic appropriateness issues."""
        suggestions = []
        
        inappropriate_token = issue.get('inappropriate_token')
        context_mismatch = issue.get('context_mismatch', '')
        suggested_alternatives = issue.get('suggested_alternatives', [])
        
        if inappropriate_token and suggested_alternatives:
            # Primary suggestion
            best_alternative = suggested_alternatives[0]
            suggestions.append(f"Replace '{inappropriate_token.text}' with '{best_alternative}' for semantic accuracy")
            
            # Context-specific explanation
            if context_mismatch == 'process_term_in_qa_context':
                suggestions.append(f"'{inappropriate_token.text}' implies making things simpler, but context requires quality validation terminology")
                suggestions.append("Use terminology that matches the quality assurance domain")
                
                # Show multiple alternatives
                if len(suggested_alternatives) > 1:
                    alternatives_text = ', '.join(f"'{alt}'" for alt in suggested_alternatives[:3])
                    suggestions.append(f"Consider alternatives: {alternatives_text}")
        
        # SpaCy-based contextual suggestion
        if doc and inappropriate_token:
            sentence_context = self._identify_sentence_semantic_context(inappropriate_token.sent, doc)
            if sentence_context['domain'] != 'general':
                suggestions.append(f"Ensure terminology matches {sentence_context['domain'].replace('_', ' ')} domain")
        
        return suggestions 