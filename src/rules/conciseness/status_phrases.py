"""
Status phrases detection module for conciseness analysis.
Detects redundant status/condition phrases using pure SpaCy morphological analysis.
Examples: 'incomplete status' → 'incomplete', 'completed state' → 'completed'
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class StatusPhrasesDetector:
    """Detector for redundant status/condition phrases using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_redundant_status_phrases(self, doc) -> List[Dict[str, Any]]:
        """Detect redundant status/condition phrases using SpaCy dependency parsing.
        
        Examples: 'incomplete status' → 'incomplete', 'completed state' → 'completed'
        """
        redundant_phrases = []
        
        for token in doc:
            # Look for noun tokens that could be redundant status words
            if token.pos_ == "NOUN" and self.is_status_condition_noun(token):
                # Check if it has an adjectival modifier that makes it redundant
                for child in token.children:
                    if (child.dep_ == "amod" and  # Adjectival modifier
                        child.pos_ == "ADJ" and
                        self.adjective_makes_status_noun_redundant(child, token, doc)):
                        
                        redundant_phrases.append({
                            'type': 'redundant_status_phrase',
                            'adjective_token': child,
                            'status_noun_token': token,
                            'redundant_phrase': f"{child.text} {token.text}",
                            'suggested_replacement': child.text,
                            'position': child.idx,
                            'dependency_analysis': {
                                'adjective_dep': child.dep_,
                                'noun_dep': token.dep_,
                                'semantic_relationship': self.analyze_adjective_noun_semantics(child, token)
                            }
                        })
        
        return redundant_phrases

    def is_status_condition_noun(self, token) -> bool:
        """Check if noun represents status/condition using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Status/condition nouns using morphological patterns
        status_morphemes = ['status', 'state', 'condition', 'situation', 'position']
        
        for morpheme in status_morphemes:
            if morpheme in lemma:
                return True
        
        # Check for derived status words
        if lemma.endswith('ness') or lemma.endswith('ity') or lemma.endswith('ism'):
            return True
        
        return False

    def adjective_makes_status_noun_redundant(self, adj_token, noun_token, doc) -> bool:
        """Check if adjective makes status noun redundant using semantic analysis."""
        
        # Method 1: Semantic completeness analysis
        if self.adjective_semantically_contains_status(adj_token, noun_token):
            return True
        
        # Method 2: Professional context redundancy
        if self.context_analyzer.is_professional_context(doc) and self.is_redundant_in_professional_writing(adj_token, noun_token):
            return True
        
        # Method 3: Morphological redundancy
        if self.have_morphological_redundancy(adj_token, noun_token):
            return True
        
        return False

    def adjective_semantically_contains_status(self, adj_token, noun_token) -> bool:
        """Check if adjective already contains the status concept using pure SpaCy morphological analysis."""
        
        # Method 1: Use SpaCy's morphological analysis for status-expressing patterns
        if self.has_status_morphological_pattern(adj_token):
            # Check if the noun is a generic status word using morphological analysis
            if self.is_generic_status_noun_morphologically(noun_token):
                return True
        
        # Method 2: Use SpaCy's semantic similarity for status overlap
        if adj_token.has_vector and noun_token.has_vector:
            semantic_similarity = adj_token.similarity(noun_token)
            if semantic_similarity > 0.6:  # High semantic overlap
                # Additional check: does adjective express completion/state
                if self.expresses_completion_state_morphologically(adj_token):
                    return True
        
        # Method 3: Use morphological derivation analysis
        if self.adjective_derives_from_status_verb(adj_token, noun_token):
            return True
        
        return False

    def has_status_morphological_pattern(self, adj_token) -> bool:
        """Check if adjective has status-expressing morphological patterns using pure SpaCy analysis."""
        lemma = adj_token.lemma_.lower()
        
        # Method 1: Morphological completion patterns using SpaCy features
        if self.has_completion_morphology(adj_token):
            return True
        
        # Method 2: State/condition morphology using derivational analysis
        if self.has_state_condition_morphology(adj_token):
            return True
        
        # Method 3: Binary opposition morphology (active/inactive, valid/invalid)
        if self.has_binary_opposition_morphology(adj_token):
            return True
        
        return False

    def has_completion_morphology(self, adj_token) -> bool:
        """Check for completion morphological patterns using SpaCy analysis."""
        lemma = adj_token.lemma_.lower()
        
        # Completion is often expressed through past participle morphology
        if adj_token.morph.get("VerbForm") == ["Part"] and adj_token.morph.get("Tense") == ["Past"]:
            return True
        
        # Morphological patterns indicating completion using linguistic anchors
        completion_morphemes = ['complet', 'finish', 'end', 'final', 'done']
        return any(morpheme in lemma for morpheme in completion_morphemes)

    def has_state_condition_morphology(self, adj_token) -> bool:
        """Check for state/condition morphological patterns using SpaCy analysis."""
        lemma = adj_token.lemma_.lower()
        
        # State adjectives often derive from verbs with specific morphological patterns
        if self.derives_from_state_verb(adj_token):
            return True
        
        # Condition morphology using linguistic patterns
        state_morphemes = ['ready', 'available', 'operational', 'functional']
        return any(morpheme in lemma for morpheme in state_morphemes)

    def has_binary_opposition_morphology(self, adj_token) -> bool:
        """Check for binary opposition morphological patterns using SpaCy analysis."""
        lemma = adj_token.lemma_.lower()
        
        # Binary oppositions often use negative prefixes
        negative_prefixes = ['un', 'in', 'dis', 'non']
        if any(lemma.startswith(prefix) for prefix in negative_prefixes):
            # Check if the base form (without prefix) exists and is status-related
            for prefix in negative_prefixes:
                if lemma.startswith(prefix):
                    base_form = lemma[len(prefix):]
                    if len(base_form) > 3 and self.is_status_base_form(base_form):
                        return True
        
        return False

    def derives_from_state_verb(self, adj_token) -> bool:
        """Check if adjective derives from state verb using morphological analysis."""
        # Check if adjective has participial morphology from state verbs
        if adj_token.morph.get("VerbForm") == ["Part"]:
            return True
        
        # Check for -able/-ible derivations from verbs
        lemma = adj_token.lemma_.lower()
        if lemma.endswith('able') or lemma.endswith('ible'):
            return True
        
        return False

    def is_status_base_form(self, base_form: str) -> bool:
        """Check if base form represents status using morphological patterns."""
        # Status base forms often have action/state morphological patterns
        status_patterns = ['activ', 'valid', 'complet', 'finish', 'available']
        return any(pattern in base_form for pattern in status_patterns)

    def is_generic_status_noun_morphologically(self, noun_token) -> bool:
        """Check if noun is generic status word using morphological analysis."""
        lemma = noun_token.lemma_.lower()
        
        # Generic status nouns have specific morphological patterns
        status_noun_morphemes = ['status', 'state', 'condition', 'situation']
        if any(morpheme in lemma for morpheme in status_noun_morphemes):
            return True
        
        # Abstract state nominalizations
        if lemma.endswith('ness') or lemma.endswith('ity'):
            return True
        
        return False

    def expresses_completion_state_morphologically(self, adj_token) -> bool:
        """Check if adjective expresses completion/state using morphological analysis."""
        # Use SpaCy's morphological features to detect completion/state
        if adj_token.morph.get("VerbForm") == ["Part"] and adj_token.morph.get("Tense") == ["Past"]:
            return True
        
        # Use morphological complexity as indicator
        complexity = self.morph_analyzer.calculate_morphological_complexity_score(adj_token)
        if complexity > 1.5:  # Complex adjectives often express specific states
            return True
        
        return False

    def adjective_derives_from_status_verb(self, adj_token, noun_token) -> bool:
        """Check if adjective derives from same verb as noun using morphological analysis."""
        adj_root = self.morph_analyzer.extract_morphological_root_advanced(adj_token.lemma_.lower())
        noun_root = self.morph_analyzer.extract_morphological_root_advanced(noun_token.lemma_.lower())
        
        # If they share morphological root, likely derivation relationship
        if adj_root == noun_root and len(adj_root) > 3:
            return True
        
        # Check for semantic derivation using SpaCy vectors
        if adj_token.has_vector and noun_token.has_vector:
            similarity = adj_token.similarity(noun_token)
            if similarity > 0.7:  # Very high similarity suggests derivation
                return True
        
        return False

    def is_redundant_in_professional_writing(self, adj_token, noun_token) -> bool:
        """Check if combination is redundant in professional writing context."""
        adj_lemma = adj_token.lemma_.lower()
        noun_lemma = noun_token.lemma_.lower()
        
        # In professional writing, these combinations are typically redundant
        redundant_combinations = [
            ('incomplete', 'status'), ('complete', 'status'),
            ('final', 'state'), ('current', 'state'),
            ('operational', 'condition'), ('functional', 'state')
        ]
        
        for adj_pattern, noun_pattern in redundant_combinations:
            if adj_pattern in adj_lemma and noun_pattern in noun_lemma:
                return True
        
        return False

    def have_morphological_redundancy(self, adj_token, noun_token) -> bool:
        """Check for morphological redundancy between adjective and noun."""
        adj_root = self.morph_analyzer.extract_morphological_root_advanced(adj_token.lemma_.lower())
        noun_root = self.morph_analyzer.extract_morphological_root_advanced(noun_token.lemma_.lower())
        
        # If they share the same morphological root, it's redundant
        if adj_root == noun_root and len(adj_root) > 3:
            return True
        
        return False

    def analyze_adjective_noun_semantics(self, adj_token, noun_token) -> Dict[str, Any]:
        """Analyze semantic relationship between adjective and noun."""
        return {
            'semantic_overlap': self.calculate_semantic_overlap(adj_token, noun_token),
            'morphological_similarity': self.calculate_morphological_similarity(adj_token, noun_token),
            'redundancy_confidence': self.calculate_redundancy_confidence(adj_token, noun_token)
        }

    def calculate_semantic_overlap(self, adj_token, noun_token) -> float:
        """Calculate semantic overlap using SpaCy vectors."""
        if adj_token.has_vector and noun_token.has_vector:
            return adj_token.similarity(noun_token)
        return 0.0

    def calculate_morphological_similarity(self, adj_token, noun_token) -> float:
        """Calculate morphological similarity between tokens."""
        adj_root = self.morph_analyzer.extract_morphological_root_advanced(adj_token.lemma_.lower())
        noun_root = self.morph_analyzer.extract_morphological_root_advanced(noun_token.lemma_.lower())
        
        if adj_root == noun_root:
            return 1.0
        elif self.morph_analyzer.have_phonological_similarity(adj_root, noun_root):
            return 0.7
        else:
            return 0.0

    def calculate_redundancy_confidence(self, adj_token, noun_token) -> float:
        """Calculate confidence that this is a redundant phrase."""
        confidence = 0.0
        
        # Boost confidence for known redundant patterns
        if self.adjective_semantically_contains_status(adj_token, noun_token):
            confidence += 0.4
        
        if self.have_morphological_redundancy(adj_token, noun_token):
            confidence += 0.3
        
        # Semantic similarity boost
        semantic_overlap = self.calculate_semantic_overlap(adj_token, noun_token)
        confidence += semantic_overlap * 0.3
        
        return min(confidence, 1.0)

    def generate_redundant_status_suggestions(self, issue: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for redundant status phrase issues."""
        suggestions = []
        
        redundant_phrase = issue.get('redundant_phrase', '')
        suggested_replacement = issue.get('suggested_replacement', '')
        
        if redundant_phrase and suggested_replacement:
            suggestions.append(f"Replace '{redundant_phrase}' with '{suggested_replacement}'")
            suggestions.append(f"The word '{suggested_replacement}' already conveys the status information")
        
        # Add linguistic explanation
        dependency_analysis = issue.get('dependency_analysis', {})
        if dependency_analysis:
            semantic_rel = dependency_analysis.get('semantic_relationship', {})
            redundancy_confidence = semantic_rel.get('redundancy_confidence', 0)
            
            if redundancy_confidence > 0.7:
                suggestions.append("High confidence: the adjective contains the status meaning")
            elif redundancy_confidence > 0.5:
                suggestions.append("The adjective likely makes the status noun unnecessary")
        
        # Professional writing guidance
        if self.context_analyzer.is_professional_context(doc):
            suggestions.append("In professional writing, avoid redundant status qualifiers")
            suggestions.append("Use direct, precise language")
        
        return suggestions 