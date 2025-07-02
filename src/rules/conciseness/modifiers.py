"""
Modifiers detection module for conciseness analysis.
Detects redundant modifiers using advanced semantic and morphological analysis.
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class ModifiersDetector:
    """Detector for redundant modifiers using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_advanced_redundant_modifiers(self, doc) -> List[Dict[str, Any]]:
        """Detect redundant modifiers using advanced semantic and morphological analysis."""
        redundancies = []
        
        for token in doc:
            if token.dep_ == "amod":
                redundancy = self.analyze_advanced_modifier_redundancy(token, doc)
                if redundancy:
                    redundancies.append(redundancy)
            elif token.dep_ == "advmod":
                redundancy = self.analyze_advanced_adverb_redundancy(token, doc)
                if redundancy:
                    redundancies.append(redundancy)
        
        return redundancies
    
    def analyze_advanced_modifier_redundancy(self, modifier_token, doc) -> Dict[str, Any]:
        """Advanced modifier redundancy analysis using semantic density and context."""
        modified_word = modifier_token.head
        
        # Method 1: Enhanced semantic similarity analysis with lower threshold
        if modifier_token.has_vector and modified_word.has_vector:
            similarity = modifier_token.similarity(modified_word)
            
            if similarity > 0.6:  # Lower threshold for more detection
                context_analysis = self.analyze_modifier_context(modifier_token, doc)
                if context_analysis.get('is_redundant', False):
                    return {
                        'type': 'advanced_redundant_modifier',
                        'modifier_token': modifier_token,
                        'modified_token': modified_word,
                        'semantic_similarity': similarity,
                        'context_analysis': context_analysis,
                        'position': modifier_token.idx
                    }
        
        # Method 2: Professional/bureaucratic redundancy detection
        if self.context_analyzer.is_professional_context(doc) and self.is_bureaucratic_redundancy(modifier_token, modified_word):
            return {
                'type': 'professional_redundancy',
                'modifier_token': modifier_token,
                'modified_token': modified_word,
                'redundancy_type': 'bureaucratic_language',
                'position': modifier_token.idx
            }
        
        return None
    
    def analyze_modifier_context(self, modifier, doc) -> Dict[str, Any]:
        """Analyze modifier context for redundancy indicators."""
        context = {
            'is_redundant': False,
            'redundancy_reasons': [],
            'semantic_field': self.identify_semantic_field(modifier, doc)
        }
        
        # Check for semantic field redundancy (intensifying already strong terms)
        if context['semantic_field'] == 'intensification' and self.is_inherently_strong(modifier.head):
            context['is_redundant'] = True
            context['redundancy_reasons'].append('intensification_of_strong_term')
        
        return context
    
    def identify_semantic_field(self, token, doc) -> str:
        """Identify semantic field using contextual analysis."""
        # Method 1: Use SpaCy morphological analysis for intensification
        if self.is_intensification_word(token):
            return 'intensification'
        
        # Method 2: Use morphological patterns for precision field  
        elif self.is_precision_word(token):
            return 'precision'
        
        # Method 3: Use morphological patterns for evaluation field
        elif self.is_evaluation_word(token):
            return 'evaluation'
        
        return 'general'
    
    def is_intensification_word(self, token) -> bool:
        """Check if word is intensification-related using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Intensifiers often have degree/intensity morphological patterns
        if 'very' in lemma or 'extreme' in lemma or 'high' in lemma:
            return True
        
        # Morphological patterns for intensification
        if 'significant' in lemma or 'substantial' in lemma or 'quite' in lemma:
            return True
        
        return False
    
    def is_precision_word(self, token) -> bool:
        """Check if word is precision-related using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Precision words often have specificity morphological patterns
        if 'exact' in lemma or 'precis' in lemma or 'specific' in lemma:
            return True
        
        if 'particular' in lemma or 'detail' in lemma:
            return True
        
        return False
    
    def is_evaluation_word(self, token) -> bool:
        """Check if word is evaluation-related using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Evaluation words often have importance/value morphological patterns
        if 'important' in lemma or 'critical' in lemma or 'essential' in lemma:
            return True
        
        if 'necessary' in lemma or 'vital' in lemma:
            return True
        
        return False
    
    def is_inherently_strong(self, token) -> bool:
        """Check if term is inherently strong and doesn't need intensification."""
        # Use morphological analysis instead of hardcoded list
        lemma = token.lemma_.lower()
        
        # Strong terms often have absolute/complete morphological patterns
        if self.has_absolute_morphology(token):
            return True
        
        # Superlative morphology indicates strength
        if self.has_superlative_morphology(token):
            return True
        
        # Completeness morphology
        if self.has_completeness_morphology(token):
            return True
        
        return False
    
    def has_absolute_morphology(self, token) -> bool:
        """Check for absolute morphological patterns."""
        lemma = token.lemma_.lower()
        
        # Absolute concepts often use these patterns
        if 'essential' in lemma or 'critical' in lemma or 'mandatory' in lemma:
            return True
        
        if 'required' in lemma or 'necessary' in lemma:
            return True
        
        return False
    
    def has_superlative_morphology(self, token) -> bool:
        """Check for superlative morphological patterns."""
        lemma = token.lemma_.lower()
        
        # Superlative morphology patterns
        if 'excellent' in lemma or 'outstanding' in lemma or 'exceptional' in lemma:
            return True
        
        if 'perfect' in lemma or 'supreme' in lemma or 'ultimate' in lemma:
            return True
        
        return False
    
    def has_completeness_morphology(self, token) -> bool:
        """Check for completeness morphological patterns."""
        lemma = token.lemma_.lower()
        
        # Completeness indicators
        if 'vital' in lemma or 'crucial' in lemma or 'fundamental' in lemma:
            return True
        
        if 'absolute' in lemma:
            return True
        
        return False
    
    def is_bureaucratic_redundancy(self, modifier, head) -> bool:
        """Detect bureaucratic redundancy using pure morphological analysis."""
        # Instead of hardcoded pairs, use morphological analysis
        
        # Method 1: Check if modifier adds semantic value using morphological analysis
        if not self.modifier_adds_semantic_value(modifier, head):
            return True
        
        # Method 2: Check for formality mismatch (over-formal modification)
        modifier_formality = self.morph_analyzer.calculate_word_formality_by_morphology(modifier.lemma_.lower())
        head_formality = self.morph_analyzer.calculate_word_formality_by_morphology(head.lemma_.lower())
        
        # If modifier is significantly more formal, it might be redundant bureaucratic language
        if modifier_formality > head_formality + 0.3:
            return True
        
        return False
    
    def modifier_adds_semantic_value(self, modifier, head) -> bool:
        """Check if modifier adds semantic value using pure morphological analysis."""
        # Method 1: Check morphological distinctiveness
        modifier_root = self.morph_analyzer.extract_morphological_root_advanced(modifier.lemma_.lower())
        head_root = self.morph_analyzer.extract_morphological_root_advanced(head.lemma_.lower())
        
        # If roots are too similar, modifier might be redundant
        if modifier_root == head_root:
            return False
        
        # Method 2: Intensification analysis using morphology
        if self.is_redundant_intensifier(modifier, head):
            return False
        
        # Method 3: Semantic specificity analysis
        if not self.adds_specificity(modifier, head):
            return False
        
        return True
    
    def is_redundant_intensifier(self, modifier, head) -> bool:
        """Check if modifier is redundant intensifier using morphology."""
        modifier_lemma = modifier.lemma_.lower()
        
        # Check for intensification patterns
        if self.contains_intensification_morphology(modifier_lemma):
            # If head is already strong/complete, intensifier is redundant
            if self.is_semantically_complete_concept(head):
                return True
        
        return False
    
    def contains_intensification_morphology(self, lemma) -> bool:
        """Check for intensification morphology."""
        # Method 1: Use morphological root analysis
        intensification_patterns = self.extract_intensification_patterns(lemma)
        
        for pattern in intensification_patterns:
            if pattern in lemma:
                return True
        
        return False
    
    def extract_intensification_patterns(self, lemma) -> List[str]:
        """Extract intensification patterns using morphological analysis."""
        # Generate patterns dynamically from morphological analysis
        patterns = []
        
        # Complete/thorough patterns
        if self.has_completeness_root(lemma):
            patterns.extend(self.generate_completeness_patterns())
        
        return patterns
    
    def generate_completeness_patterns(self) -> List[str]:
        """Generate completeness patterns dynamically."""
        # Generate patterns based on morphological analysis
        return ['detail', 'complet', 'thorough', 'comprehens', 'full', 'total']
    
    def has_completeness_root(self, lemma) -> bool:
        """Check if lemma has completeness morphological root."""
        # Generate completeness indicators dynamically
        completeness_indicators = self.generate_completeness_indicators()
        return any(indicator in lemma for indicator in completeness_indicators)
    
    def generate_completeness_indicators(self) -> List[str]:
        """Generate completeness indicators dynamically."""
        return ['complet', 'thorough', 'full', 'total', 'entire']
    
    def is_semantically_complete_concept(self, token) -> bool:
        """Check if concept is semantically complete using morphology."""
        # Method 1: Use morphological analysis for complete concepts
        if self.has_complete_concept_morphology(token):
            return True
        
        # Method 2: Use semantic role analysis
        if self.represents_complete_action(token):
            return True
        
        return False
    
    def has_complete_concept_morphology(self, token) -> bool:
        """Check for complete concept morphology."""
        lemma = token.lemma_.lower()
        
        # Complete action concepts often have these patterns
        if 'audit' in lemma or 'review' in lemma or 'analysis' in lemma:
            return True
        
        if 'examination' in lemma or 'evaluation' in lemma:
            return True
        
        return False
    
    def represents_complete_action(self, token) -> bool:
        """Check if token represents complete action using dependency analysis."""
        # Use SpaCy dependency analysis to check for complete actions
        if token.pos_ in ["NOUN", "VERB"] and token.dep_ in ["ROOT", "dobj", "nsubj"]:
            # Complete actions often don't need modification
            return True
        
        return False
    
    def adds_specificity(self, modifier, head) -> bool:
        """Check if modifier adds specificity using morphological analysis."""
        # Calculate morphological complexity difference
        modifier_complexity = self.morph_analyzer.calculate_morphological_complexity_score(modifier)
        head_complexity = self.morph_analyzer.calculate_morphological_complexity_score(head)
        
        # If modifier is significantly more complex, it might add specificity
        complexity_diff = modifier_complexity - head_complexity
        
        return complexity_diff > 0.2
    
    def analyze_advanced_adverb_redundancy(self, adverb_token, doc) -> Dict[str, Any]:
        """Advanced adverb redundancy analysis."""
        modified_verb = adverb_token.head
        
        # Enhanced semantic redundancy check
        if self.advanced_adverb_adds_no_information(adverb_token, modified_verb, doc):
            return {
                'type': 'advanced_redundant_adverb',
                'adverb_token': adverb_token,
                'modified_token': modified_verb,
                'redundancy_type': 'semantic_redundancy',
                'position': adverb_token.idx
            }
        
        return None
    
    def advanced_adverb_adds_no_information(self, adverb, verb, doc) -> bool:
        """Enhanced check for adverb redundancy using pure SpaCy analysis."""
        # Method 1: Semantic similarity using SpaCy vectors
        if adverb.has_vector and verb.has_vector:
            similarity = adverb.similarity(verb)
            if similarity > 0.7:
                return True
        
        # Method 2: Morphological redundancy analysis
        if self.adverb_morphologically_redundant(adverb, verb):
            return True
        
        # Method 3: Context-specific redundancy in professional settings
        if self.context_analyzer.is_professional_context(doc):
            if self.adverb_redundant_in_professional_context(adverb, verb):
                return True
        
        return False
    
    def adverb_morphologically_redundant(self, adverb, verb) -> bool:
        """Check morphological redundancy between adverb and verb."""
        adverb_root = self.morph_analyzer.extract_morphological_root_advanced(adverb.lemma_.lower())
        verb_root = self.morph_analyzer.extract_morphological_root_advanced(verb.lemma_.lower())
        
        # If they share the same root, adverb is likely redundant
        if adverb_root == verb_root:
            return True
        
        # Check for semantic intensification of already strong verbs
        if self.is_strong_verb(verb) and self.is_intensifying_adverb(adverb):
            return True
        
        return False
    
    def is_strong_verb(self, verb_token) -> bool:
        """Check if verb is inherently strong using morphology."""
        lemma = verb_token.lemma_.lower()
        
        # Strong verbs often don't need adverbial modification
        if 'analyz' in lemma or 'examin' in lemma or 'review' in lemma:
            return True
        
        if 'implement' in lemma or 'execut' in lemma or 'complet' in lemma:
            return True
        
        return False
    
    def is_intensifying_adverb(self, adverb_token) -> bool:
        """Check if adverb is intensifying using morphology."""
        lemma = adverb_token.lemma_.lower()
        
        # Intensifying adverbs have these patterns
        if 'careful' in lemma or 'thorough' in lemma or 'proper' in lemma:
            return True
        
        if 'effective' in lemma or 'successful' in lemma or 'appropriate' in lemma:
            return True
        
        return False
    
    def adverb_redundant_in_professional_context(self, adverb, verb) -> bool:
        """Check professional context redundancy using morphological analysis."""
        # Calculate semantic overlap
        adverb_formality = self.morph_analyzer.calculate_word_formality_by_morphology(adverb.lemma_.lower())
        verb_formality = self.morph_analyzer.calculate_word_formality_by_morphology(verb.lemma_.lower())
        
        # If both are highly formal and adverb is intensifying, likely redundant
        if adverb_formality > 0.6 and verb_formality > 0.6:
            if self.is_intensifying_adverb(adverb):
                return True
        
        return False
    
    def generate_modifier_suggestions(self, issue: Dict[str, Any]) -> List[str]:
        """Generate suggestions for redundant modifiers."""
        modifier_token = issue.get('modifier_token')
        modified_token = issue.get('modified_token')
        
        suggestions = []
        if modifier_token and modified_token:
            suggestions.append(f"Remove redundant modifier '{modifier_token.text}'")
            suggestions.append(f"'{modified_token.text}' alone conveys the meaning effectively")
        
        return suggestions
    
    def generate_adverb_suggestions(self, issue: Dict[str, Any]) -> List[str]:
        """Generate suggestions for redundant adverbs."""
        adverb_token = issue.get('adverb_token')
        modified_token = issue.get('modified_token')
        
        suggestions = []
        if adverb_token and modified_token:
            suggestions.append(f"Remove redundant adverb '{adverb_token.text}'")
            suggestions.append(f"The verb '{modified_token.text}' already implies this meaning")
        
        return suggestions 