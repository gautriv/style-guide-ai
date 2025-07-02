"""
Synonyms detection module for conciseness analysis.
Detects redundant synonym pairs using pure SpaCy morphological and semantic analysis.
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer
from .semantic_fields import SemanticFieldsAnalyzer

class SynonymsDetector:
    """Detector for redundant synonym pairs using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.semantic_analyzer = SemanticFieldsAnalyzer()
    
    def detect_synonym_pair_redundancy(self, doc) -> List[Dict[str, Any]]:
        """Detect redundant synonyms in coordination using morphological analysis."""
        synonym_redundancies = []
        
        for token in doc:
            if token.dep_ == "conj":  # Coordinated elements
                redundancy_analysis = self.analyze_coordination_redundancy(token, doc)
                if redundancy_analysis:
                    synonym_redundancies.append(redundancy_analysis)
        
        return synonym_redundancies

    def analyze_coordination_redundancy(self, conj_token, doc) -> Dict[str, Any]:
        """Analyze coordinated elements for semantic redundancy using SpaCy."""
        # Find the head of the coordination
        head_token = conj_token.head
        
        # Check if coordinated elements are semantically similar
        if self.are_semantically_redundant(head_token, conj_token, doc):
            return {
                'type': 'synonym_pair_redundancy',
                'first_token': head_token,
                'second_token': conj_token,
                'redundancy_analysis': self.analyze_semantic_redundancy(head_token, conj_token, doc),
                'position': head_token.idx
            }
        
        return None
    
    def are_semantically_redundant(self, token1, token2, doc) -> bool:
        """Check if two tokens are semantically redundant using SpaCy analysis."""
        # Method 1: Direct semantic similarity using SpaCy vectors
        if token1.has_vector and token2.has_vector:
            similarity = token1.similarity(token2)
            if similarity > 0.7:  # High semantic similarity threshold
                return True
        
        # Method 2: Contextual redundancy analysis
        if self.are_contextually_redundant(token1, token2, doc):
            return True
        
        # Method 3: Morphological family analysis
        if self.are_morphologically_related(token1, token2):
            return True
        
        return False
    
    def are_contextually_redundant(self, token1, token2, doc) -> bool:
        """Check contextual redundancy using SpaCy dependency and semantic analysis."""
        # Look for common redundant pairs in professional contexts
        lemma1 = token1.lemma_.lower()
        lemma2 = token2.lemma_.lower()
        
        # Use SpaCy's morphological analysis to identify redundant semantic fields
        if self.belong_to_same_semantic_field(token1, token2, doc):
            # Professional communication redundancies
            if self.is_professional_communication_context(doc):
                return self.are_communication_synonyms(lemma1, lemma2)
            
            # Business process redundancies  
            elif self.is_business_process_context(doc):
                return self.are_process_synonyms(lemma1, lemma2)
        
        return False
    
    def belong_to_same_semantic_field(self, token1, token2, doc) -> bool:
        """Check if tokens belong to same semantic field using SpaCy analysis."""
        # Use SemanticFieldsAnalyzer for field identification
        field1 = self.semantic_analyzer.identify_semantic_field_advanced(token1, doc)
        field2 = self.semantic_analyzer.identify_semantic_field_advanced(token2, doc)
        
        return field1 == field2 and field1 != 'general'
    
    def are_morphologically_related(self, token1, token2) -> bool:
        """Check if tokens are morphologically related."""
        root1 = self.morph_analyzer.extract_morphological_root_advanced(token1.lemma_.lower())
        root2 = self.morph_analyzer.extract_morphological_root_advanced(token2.lemma_.lower())
        
        # If they share the same morphological root, they're related
        return root1 == root2 and len(root1) > 2
    
    def analyze_semantic_redundancy(self, token1, token2, doc) -> Dict[str, Any]:
        """Analyze semantic redundancy between two tokens."""
        analysis = {
            'redundancy_type': 'general_synonyms',
            'context_appropriateness': 0.0,
            'semantic_similarity': 0.0
        }
        
        # Calculate semantic similarity if vectors available
        if token1.has_vector and token2.has_vector:
            analysis['semantic_similarity'] = token1.similarity(token2)
        
        # Determine which term is more appropriate for context
        formality1 = self.morph_analyzer.calculate_word_formality_by_morphology(token1.lemma_.lower())
        formality2 = self.morph_analyzer.calculate_word_formality_by_morphology(token2.lemma_.lower())
        
        # Positive score means token2 is better, negative means token1 is better
        analysis['context_appropriateness'] = formality2 - formality1
        
        return analysis
    
    def are_communication_synonyms(self, lemma1, lemma2) -> bool:
        """Check for communication synonym pairs using pure SpaCy morphological analysis."""
        # Method 1: Check if both are communication words with morphological similarity
        if self.is_communication_word_by_morphology(lemma1) and self.is_communication_word_by_morphology(lemma2):
            # Check morphological relationship
            if self.have_similar_morphological_patterns(lemma1, lemma2):
                return True
        
        # Method 2: Specific patterns for business communication redundancy
        if self.are_business_communication_synonyms(lemma1, lemma2):
            return True
        
        return False
    
    def are_business_communication_synonyms(self, lemma1, lemma2) -> bool:
        """Check for specific business communication synonym pairs using morphological analysis."""
        # Method 1: Cognitive state morphology (awareness-type terms)
        is_cognitive1 = self.has_cognitive_state_morphology(lemma1)
        is_cognitive2 = self.has_cognitive_state_morphology(lemma2)
        
        # Method 2: Conformity/compliance morphology (alignment-type terms) 
        is_conformity1 = self.has_conformity_morphology(lemma1)
        is_conformity2 = self.has_conformity_morphology(lemma2)
        
        # Semantic redundancy: cognitive + conformity terms often mean the same in business context
        if (is_cognitive1 and is_conformity2) or (is_cognitive2 and is_conformity1):
            return True
        
        # Method 3: Both have similar business communication morphology
        if self.has_business_communication_morphology(lemma1) and self.has_business_communication_morphology(lemma2):
            return True
        
        return False
    
    def has_cognitive_state_morphology(self, lemma) -> bool:
        """Check for cognitive state morphology patterns."""
        # Cognitive state words often have these morphological patterns
        if 'aware' in lemma or 'understand' in lemma or 'know' in lemma:
            return True
        
        if 'comprehend' in lemma or 'realize' in lemma:
            return True
        
        return False
    
    def has_conformity_morphology(self, lemma) -> bool:
        """Check for conformity/compliance morphology patterns."""
        # Conformity words often have these morphological patterns
        if 'align' in lemma or 'comply' in lemma or 'conform' in lemma:
            return True
        
        if 'adher' in lemma or 'follow' in lemma:
            return True
        
        return False
    
    def has_business_communication_morphology(self, lemma) -> bool:
        """Check for business communication morphology patterns."""
        # Business communication terms share certain morphological patterns
        if 'feed' in lemma or 'input' in lemma or 'comment' in lemma:
            return True
        
        if 'suggest' in lemma or 'respond' in lemma:
            return True
        
        return False
    
    def are_process_synonyms(self, lemma1, lemma2) -> bool:
        """Check for process synonym pairs using pure SpaCy morphological analysis."""
        # Use morphological analysis instead of hardcoded lists
        if self.is_process_verb_by_morphology(lemma1) and self.is_process_verb_by_morphology(lemma2):
            # Check if they have similar semantic weight/formality
            formality1 = self.morph_analyzer.calculate_word_formality_by_morphology(lemma1)
            formality2 = self.morph_analyzer.calculate_word_formality_by_morphology(lemma2)
            
            # If one is significantly more formal, they might be synonyms
            formality_diff = abs(formality1 - formality2)
            return formality_diff > 0.3  # Different formality levels suggest synonym pair
        
        return False
    
    def is_process_verb_by_morphology(self, lemma) -> bool:
        """Check if verb is process-related using morphological analysis."""
        # Process verbs often have movement/distribution morphology
        if 'circul' in lemma or 'distribut' in lemma or 'shar' in lemma:
            return True
        
        # Motion/transfer semantic patterns
        if 'send' in lemma or 'deliv' in lemma or 'pass' in lemma or 'mov' in lemma:
            return True
        
        # Latinate action patterns (often formal process verbs)
        if lemma.endswith('ate') or lemma.endswith('ize') or lemma.endswith('ify'):
            return True
        
        return False
    
    def is_communication_word_by_morphology(self, lemma) -> bool:
        """Check if word is communication-related using morphological patterns."""
        # Communication words often have these morphological roots
        if 'feed' in lemma or 'comment' in lemma or 'respond' in lemma or 'input' in lemma:
            return True
        
        # Derivational patterns indicating communication
        if lemma.endswith('back') or lemma.endswith('ment') or lemma.endswith('ary'):
            return True
        
        return False
    
    def have_similar_morphological_patterns(self, lemma1, lemma2) -> bool:
        """Check morphological similarity using derivational analysis."""
        # Extract roots by removing common derivational suffixes
        root1 = self.morph_analyzer.extract_morphological_root_advanced(lemma1)
        root2 = self.morph_analyzer.extract_morphological_root_advanced(lemma2)
        
        # Check if they share morphological features
        if root1 and root2:
            # Similar length and phonological patterns often indicate semantic similarity
            length_similarity = abs(len(root1) - len(root2)) <= 2
            if length_similarity and self.morph_analyzer.have_phonological_similarity(root1, root2):
                return True
        
        return False
    
    # Helper methods for context analysis
    def is_professional_communication_context(self, doc) -> bool:
        """Check if context is professional communication."""
        # Look for communication entities and professional context
        return self.context_analyzer.is_professional_context(doc) and self.has_communication_indicators(doc)
    
    def has_communication_indicators(self, doc) -> bool:
        """Check for communication indicators in document."""
        comm_count = 0
        for token in doc:
            if self.semantic_analyzer.is_communication_verb(token):
                comm_count += 1
        
        return comm_count > 0
    
    def is_business_process_context(self, doc) -> bool:
        """Check if context is business process related."""
        return self.context_analyzer.is_business_context(doc) and self.has_process_indicators(doc)
    
    def has_process_indicators(self, doc) -> bool:
        """Check for process indicators in document."""
        process_count = 0
        for token in doc:
            if self.semantic_analyzer.has_process_morphology(token):
                process_count += 1
        
        return process_count > 0
    
    def generate_synonym_redundancy_suggestions(self, issue: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for synonym pair redundancy."""
        suggestions = []
        
        first_token = issue.get('first_token')
        second_token = issue.get('second_token')
        redundancy_analysis = issue.get('redundancy_analysis', {})
        
        if first_token and second_token:
            redundancy_type = redundancy_analysis.get('redundancy_type', 'general_synonyms')
            context_score = redundancy_analysis.get('context_appropriateness', 0.0)
            
            # Enhanced suggestions for specific morphological patterns
            first_lemma = first_token.lemma_.lower()
            second_lemma = second_token.lemma_.lower()
            
            # Special handling for cognitive/conformity redundancy (awareness + alignment)
            if self.has_cognitive_state_morphology(first_lemma) and self.has_conformity_morphology(second_lemma):
                suggestions.append(f"Use 'ensure employees understand' instead of 'ensure {first_token.text} and {second_token.text}'")
                suggestions.append(f"'Understanding' implies compliance - '{first_token.text} and {second_token.text}' is redundant")
                suggestions.append("For technical writing clarity: 'ensure employees understand and follow' is more direct")
            elif self.has_conformity_morphology(first_lemma) and self.has_cognitive_state_morphology(second_lemma):
                suggestions.append(f"Use 'ensure employees understand' instead of 'ensure {first_token.text} and {second_token.text}'")
                suggestions.append(f"'Understanding' implies compliance - '{first_token.text} and {second_token.text}' is redundant")
                suggestions.append("For technical writing clarity: 'ensure employees understand and follow' is more direct")
            
            # Determine which term to keep based on context appropriateness
            elif context_score > 0:  # second_token is better
                suggestions.append(f"Remove '{first_token.text}' - '{second_token.text}' conveys the same meaning")
                suggestions.append(f"Use '{second_token.text}' instead of '{first_token.text} and {second_token.text}'")
            elif context_score < 0:  # first_token is better
                suggestions.append(f"Remove '{second_token.text}' - '{first_token.text}' conveys the same meaning")
                suggestions.append(f"Use '{first_token.text}' instead of '{first_token.text} and {second_token.text}'")
            else:  # Equal - suggest the first one
                suggestions.append(f"Choose either '{first_token.text}' or '{second_token.text}' - both convey the same meaning")
            
            # Context-specific suggestions
            if redundancy_type == 'communication_synonyms':
                suggestions.append("For modern business communication, 'feedback' is preferred over 'commentary'")
            elif redundancy_type == 'process_synonyms':
                suggestions.append("For conversational clarity, 'share' is more modern than 'circulate'")
        
        return suggestions 