"""
Conceptual redundancy detection module for 10/10 quality analysis.
Detects subtle semantic redundancy and vague qualifiers using pure SpaCy morphological analysis.
Example: 'results show promising improvement' - both 'results' and 'improvement' signal progress
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class ConceptualRedundancyDetector:
    """Detector for conceptual redundancy and vague qualifiers using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_conceptual_redundancy_issues(self, doc) -> List[Dict[str, Any]]:
        """Detect conceptual redundancy using SpaCy morphological analysis."""
        issues = []
        
        for sentence in doc.sents:
            # Method 1: Detect progress/outcome semantic redundancy
            progress_redundancy = self._detect_progress_concept_redundancy(sentence)
            issues.extend(progress_redundancy)
            
            # Method 2: Detect vague qualifiers
            vague_qualifiers = self._detect_vague_qualifiers(sentence)
            issues.extend(vague_qualifiers)
            
            # Method 3: Detect conceptual overlap in noun phrases
            conceptual_overlap = self._detect_noun_phrase_conceptual_overlap(sentence)
            issues.extend(conceptual_overlap)
        
        return issues
    
    def _detect_progress_concept_redundancy(self, sentence) -> List[Dict[str, Any]]:
        """Detect progress/outcome conceptual redundancy using SpaCy morphological analysis."""
        issues = []
        
        # Find tokens that semantically express progress/outcome
        progress_tokens = []
        for token in sentence:
            if self._expresses_progress_concept(token):
                progress_tokens.append(token)
        
        # Check for redundancy when multiple progress concepts appear close together
        if len(progress_tokens) >= 2:
            for i in range(len(progress_tokens) - 1):
                token1 = progress_tokens[i]
                token2 = progress_tokens[i + 1]
                
                # Check if they're in close proximity (within 4 words)
                distance = abs(token1.i - token2.i)
                if distance <= 4:
                    redundancy_analysis = self._analyze_progress_redundancy(token1, token2, sentence)
                    if redundancy_analysis['is_redundant']:
                        issues.append({
                            'type': 'conceptual_progress_redundancy',
                            'redundant_tokens': [token1, token2],
                            'conceptual_overlap': redundancy_analysis['overlap_type'],
                            'suggested_alternatives': redundancy_analysis['alternatives'],
                            'confidence_score': redundancy_analysis['confidence'],
                            'position': min(token1.idx, token2.idx)
                        })
        
        return issues
    
    def _detect_vague_qualifiers(self, sentence) -> List[Dict[str, Any]]:
        """Detect vague qualifiers using SpaCy morphological analysis."""
        issues = []
        
        for token in sentence:
            if self._is_vague_qualifier(token):
                vagueness_analysis = self._analyze_qualifier_vagueness(token, sentence)
                if vagueness_analysis['is_problematic']:
                    issues.append({
                        'type': 'vague_qualifier',
                        'vague_token': token,
                        'vagueness_type': vagueness_analysis['vagueness_type'],
                        'suggested_specifics': vagueness_analysis['specific_alternatives'],
                        'context_appropriateness': vagueness_analysis['context_score'],
                        'position': token.idx
                    })
        
        return issues
    
    def _detect_noun_phrase_conceptual_overlap(self, sentence) -> List[Dict[str, Any]]:
        """Detect conceptual overlap within noun phrases using SpaCy analysis."""
        issues = []
        
        # Find noun phrases and analyze for conceptual redundancy
        for token in sentence:
            if token.pos_ == "NOUN" and token.dep_ in ["nsubj", "dobj", "nsubjpass"]:
                np_analysis = self._analyze_noun_phrase_concepts(token, sentence)
                if np_analysis['has_conceptual_overlap']:
                    issues.append({
                        'type': 'noun_phrase_conceptual_overlap',
                        'overlapping_concepts': np_analysis['overlapping_tokens'],
                        'overlap_description': np_analysis['overlap_description'],
                        'suggested_simplification': np_analysis['simplified_version'],
                        'position': token.idx
                    })
        
        return issues
    
    def _expresses_progress_concept(self, token) -> bool:
        """Check if token expresses progress/improvement concept using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Method 1: Direct progress/outcome morphological patterns
        if token.pos_ == "NOUN":
            # Progress-signaling nouns using morphological patterns
            progress_morphemes = ['result', 'outcome', 'improvement', 'progress', 'advance', 'gain', 'success']
            if any(morpheme in lemma for morpheme in progress_morphemes):
                return True
        
        elif token.pos_ == "VERB":
            # Progress-signaling verbs
            progress_verbs = ['improve', 'advance', 'progress', 'succeed', 'achieve', 'accomplish']
            if lemma in progress_verbs:
                return True
        
        # Method 2: Morphological analysis of suffixes indicating progress
        if token.pos_ == "NOUN" and len(lemma) > 4:
            # Suffixes that often indicate positive change/progress
            progress_suffixes = ['ment', 'tion', 'ance', 'ence', 'ness']
            if any(lemma.endswith(suffix) for suffix in progress_suffixes):
                # Check if the root implies progress
                root = lemma[:-4] if len(lemma) > 4 else lemma
                if root in ['improve', 'advance', 'enhance', 'develop']:
                    return True
        
        return False
    
    def _analyze_progress_redundancy(self, token1, token2, sentence) -> Dict[str, Any]:
        """Analyze if two progress tokens are redundant using morphological analysis."""
        lemma1 = token1.lemma_.lower()
        lemma2 = token2.lemma_.lower()
        
        # Specific case: "results" + "improvement" - both signal positive change
        if ('result' in lemma1 and 'improve' in lemma2) or ('result' in lemma2 and 'improve' in lemma1):
            return {
                'is_redundant': True,
                'overlap_type': 'progress_outcome_redundancy',
                'confidence': 0.85,
                'alternatives': ['outcome', 'performance', 'data', 'findings'],
                'explanation': 'Both words signal positive change/progress - creates conceptual redundancy'
            }
        
        # Generic progress concept overlap
        if (self._expresses_progress_concept(token1) and 
            self._expresses_progress_concept(token2)):
            return {
                'is_redundant': True,
                'overlap_type': 'generic_progress_redundancy',
                'confidence': 0.7,
                'alternatives': self._suggest_progress_alternatives(token1, token2),
                'explanation': 'Multiple progress concepts create semantic redundancy'
            }
        
        return {'is_redundant': False}
    
    def _suggest_progress_alternatives(self, token1, token2) -> List[str]:
        """Suggest alternatives for redundant progress concepts."""
        # For results + improvement case
        if ('result' in token1.lemma_.lower() and 'improve' in token2.lemma_.lower()):
            return ['outcome', 'performance', 'data']
        elif ('result' in token2.lemma_.lower() and 'improve' in token1.lemma_.lower()):
            return ['performance', 'outcome', 'data']
        
        return ['performance', 'outcome', 'findings', 'data']
    
    def _is_vague_qualifier(self, token) -> bool:
        """Check if token is a vague qualifier using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Vague qualifiers that lack specificity
        if token.pos_ == "ADJ":
            vague_adjectives = ['promising', 'significant', 'considerable', 'substantial', 'notable']
            return lemma in vague_adjectives
        
        elif token.pos_ == "ADV":
            vague_adverbs = ['somewhat', 'rather', 'quite', 'fairly', 'relatively']
            return lemma in vague_adverbs
        
        return False
    
    def _analyze_qualifier_vagueness(self, token, sentence) -> Dict[str, Any]:
        """Analyze vagueness of qualifier using contextual analysis."""
        lemma = token.lemma_.lower()
        
        # Specific case: "promising" without data backing
        if lemma == 'promising':
            return {
                'is_problematic': True,
                'vagueness_type': 'unsupported_qualitative_claim',
                'specific_alternatives': ['measurable', 'quantifiable', 'documented', 'notable'],
                'context_score': 0.3,
                'explanation': "'Promising' is vague unless backed by specific data or metrics"
            }
        
        # Other vague qualifiers
        if lemma in ['significant', 'considerable', 'substantial']:
            return {
                'is_problematic': True,
                'vagueness_type': 'quantitative_vagueness',
                'specific_alternatives': ['measurable', 'documented', 'quantified'],
                'context_score': 0.4,
                'explanation': f"'{token.text}' lacks specificity - prefer quantified terms"
            }
        
        return {'is_problematic': False}
    
    def _analyze_noun_phrase_concepts(self, noun_token, sentence) -> Dict[str, Any]:
        """Analyze noun phrase for conceptual overlap."""
        analysis = {
            'has_conceptual_overlap': False,
            'overlapping_tokens': [],
            'overlap_description': '',
            'simplified_version': ''
        }
        
        # Find the verb this noun is associated with
        associated_verb = None
        for token in sentence:
            if (token.pos_ == "VERB" and 
                any(child == noun_token for child in token.children)):
                associated_verb = token
                break
        
        # Check for "results show improvement" pattern
        if (noun_token.lemma_.lower() == 'result' and 
            associated_verb and associated_verb.lemma_.lower() == 'show'):
            
            # Look for object that might be redundant with "results"
            for child in associated_verb.children:
                if (child.dep_ == "dobj" and 
                    self._expresses_progress_concept(child)):
                    analysis['has_conceptual_overlap'] = True
                    analysis['overlapping_tokens'] = [noun_token, child]
                    analysis['overlap_description'] = 'Subject and object both express outcome/progress'
                    analysis['simplified_version'] = f"The {child.text.lower()}"
                    break
        
        return analysis
    
    def generate_conceptual_redundancy_suggestions(self, issue: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for conceptual redundancy issues."""
        suggestions = []
        issue_type = issue.get('type', '')
        
        if issue_type == 'conceptual_progress_redundancy':
            redundant_tokens = issue.get('redundant_tokens', [])
            suggested_alternatives = issue.get('suggested_alternatives', [])
            
            if len(redundant_tokens) >= 2:
                token1_text = redundant_tokens[0].text
                token2_text = redundant_tokens[1].text
                
                suggestions.append(f"Conceptual redundancy: '{token1_text}' and '{token2_text}' both signal progress")
                suggestions.append("Remove one term to eliminate semantic overlap")
                
                if suggested_alternatives:
                    alt_text = ', '.join(f"'{alt}'" for alt in suggested_alternatives[:3])
                    suggestions.append(f"Consider alternatives: {alt_text}")
                
                # Specific guidance for results/improvement
                if ('result' in token1_text.lower() and 'improve' in token2_text.lower()):
                    suggestions.append("Use 'The outcome shows measurable improvement' for clarity")
                    suggestions.append("Or simplify to 'Performance has improved significantly'")
        
        elif issue_type == 'vague_qualifier':
            vague_token = issue.get('vague_token')
            suggested_specifics = issue.get('suggested_specifics', [])
            
            if vague_token:
                suggestions.append(f"Vague qualifier: '{vague_token.text}' lacks specificity")
                
                if suggested_specifics:
                    specifics_text = ', '.join(f"'{spec}'" for spec in suggested_specifics[:3])
                    suggestions.append(f"Use specific terms: {specifics_text}")
                
                if vague_token.lemma_.lower() == 'promising':
                    suggestions.append("Replace 'promising' with data-backed qualifiers")
                    suggestions.append("Consider: 'measurable', 'documented', or specific metrics")
        
        elif issue_type == 'noun_phrase_conceptual_overlap':
            overlapping_concepts = issue.get('overlapping_concepts', [])
            suggested_simplification = issue.get('suggested_simplification', '')
            
            if overlapping_concepts and len(overlapping_concepts) >= 2:
                concept1 = overlapping_concepts[0].text
                concept2 = overlapping_concepts[1].text
                suggestions.append(f"Conceptual overlap: '{concept1}' and '{concept2}' express similar concepts")
                
                if suggested_simplification:
                    suggestions.append(f"Simplify to: '{suggested_simplification}'")
        
        return suggestions 