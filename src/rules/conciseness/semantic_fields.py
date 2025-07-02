"""
Semantic fields analyzer module for conciseness analysis.
Identifies semantic fields and contextual analysis using pure SpaCy morphological analysis.
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class SemanticFieldsAnalyzer:
    """Analyzer for semantic fields using pure SpaCy morphological analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def identify_semantic_field_advanced(self, token, doc) -> str:
        """Identify semantic field using advanced SpaCy analysis."""
        lemma = token.lemma_.lower()
        
        # Communication field detection using morphological and contextual cues
        if self.is_communication_term(token, doc):
            return 'communication'
        
        # Process/action field detection
        elif self.is_process_term(token, doc):
            return 'process'
        
        # Certainty/guarantee field detection
        elif self.is_certainty_term(token, doc):
            return 'certainty'
        
        # Awareness/understanding field detection
        elif self.is_awareness_term(token, doc):
            return 'awareness'
        
        return 'general'
    
    def is_communication_term(self, token, doc) -> bool:
        """Check if token is communication-related using pure SpaCy morphological analysis."""
        # Method 1: Morphological pattern analysis for communication terms
        if self.has_communication_morphology(token):
            return True
        
        # Method 2: Contextual analysis using SpaCy dependency parsing
        if self.appears_in_communication_context(token, doc):
            return True
        
        # Method 3: Semantic role analysis
        if self.has_communication_semantic_role(token, doc):
            return True
        
        return False
    
    def has_communication_morphology(self, token) -> bool:
        """Check communication morphology using SpaCy features."""
        lemma = token.lemma_.lower()
        
        # Communication words often have these morphological patterns
        if 'feed' in lemma or 'comment' in lemma or 'respond' in lemma:
            return True
        
        # Check for derivational patterns indicating communication
        if lemma.endswith('back') or lemma.endswith('ment') or lemma.endswith('ary'):
            return True
        
        return False
    
    def appears_in_communication_context(self, token, doc) -> bool:
        """Check if token appears in communication context using SpaCy."""
        # Look for communication verbs in the sentence
        for sent_token in token.sent:
            if sent_token.pos_ == "VERB":
                if self.is_communication_verb(sent_token):
                    return True
        
        # Look for entities that suggest communication context
        for ent in doc.ents:
            if self.is_person_or_org_communication_entity(ent, token):
                return True
        
        return False
    
    def has_communication_semantic_role(self, token, doc) -> bool:
        """Check semantic role for communication using SpaCy dependency analysis."""
        # Communication terms often appear as objects of communication verbs
        if self.is_communication_object_role(token):
            head = token.head
            if head.pos_ == "VERB" and self.is_communication_verb(head):
                return True
        
        return False
    
    def is_process_term(self, token, doc) -> bool:
        """Check if token is process-related using pure SpaCy morphological analysis."""
        # Method 1: Morphological analysis for process verbs
        if self.has_process_morphology(token):
            return True
        
        # Method 2: Syntactic context analysis
        if self.appears_in_process_context(token, doc):
            return True
        
        # Method 3: Business context + formal verb patterns
        if self.context_analyzer.is_business_context(doc) and self.has_formal_verb_morphology(token):
            return True
        
        return False
    
    def has_process_morphology(self, token) -> bool:
        """Check process morphology using SpaCy features."""
        lemma = token.lemma_.lower()
        
        # Process verbs often have movement/distribution semantics
        if 'circul' in lemma or 'distribut' in lemma or 'shar' in lemma:
            return True
        
        # Motion/transfer semantic patterns
        if 'send' in lemma or 'deliv' in lemma or 'pass' in lemma:
            return True
        
        return False
    
    def is_certainty_term(self, token, doc) -> bool:
        """Check if token relates to certainty using pure SpaCy morphological analysis."""
        # Method 1: Morphological analysis for certainty
        if self.has_certainty_morphology(token):
            return True
        
        # Method 2: Semantic context analysis
        if self.appears_in_certainty_context(token, doc):
            return True
        
        return False
    
    def has_certainty_morphology(self, token) -> bool:
        """Check certainty morphology using SpaCy features."""
        lemma = token.lemma_.lower()
        
        # Certainty words often have these semantic roots
        if 'guarant' in lemma or 'ensur' in lemma or 'assur' in lemma:
            return True
        
        # Verification/confirmation patterns
        if 'confirm' in lemma or 'verif' in lemma or 'certain' in lemma:
            return True
        
        return False
    
    def is_awareness_term(self, token, doc) -> bool:
        """Check if token relates to awareness using pure SpaCy morphological analysis."""
        # Method 1: Morphological patterns for awareness/understanding
        if self.has_awareness_morphology(token):
            return True
        
        # Method 2: Cognitive verb context
        if self.appears_in_cognitive_context(token, doc):
            return True
        
        return False
    
    def has_awareness_morphology(self, token) -> bool:
        """Check awareness morphology using SpaCy features."""
        lemma = token.lemma_.lower()
        
        # Awareness/cognition semantic roots
        if 'awar' in lemma or 'understand' in lemma or 'comprehend' in lemma:
            return True
        
        # Alignment/agreement patterns
        if 'align' in lemma or 'complian' in lemma or 'accord' in lemma:
            return True
        
        return False
    
    # Context analysis helper methods
    def is_communication_verb(self, verb_token) -> bool:
        """Check if verb is communication-related using morphology."""
        lemma = verb_token.lemma_.lower()
        
        # Communication verbs have these patterns
        if 'communic' in lemma or 'discuss' in lemma or 'inform' in lemma:
            return True
        
        if 'tell' in lemma or 'explain' in lemma or 'report' in lemma:
            return True
        
        if 'share' in lemma or 'send' in lemma or 'distribut' in lemma:
            return True
        
        return False
    
    def is_person_or_org_communication_entity(self, entity, token) -> bool:
        """Check if entity suggests communication context."""
        if entity.label_ in ["PERSON", "ORG"] and abs(entity.start - token.i) <= 5:
            return True
        return False
    
    def is_communication_object_role(self, token) -> bool:
        """Check if token has communication object dependency role."""
        return token.dep_ in ["dobj", "iobj", "nmod"]
    
    def appears_in_process_context(self, token, doc) -> bool:
        """Check if appears in process context using SpaCy analysis."""
        # Look for objects that suggest processes (documents, versions, etc.)
        for child in token.children:
            if self.is_process_object_role(child):
                if self.is_process_object(child):
                    return True
        
        return False
    
    def has_formal_verb_morphology(self, token) -> bool:
        """Check for formal verb morphology patterns."""
        return self.morph_analyzer.has_latinate_formality_pattern(token)
    
    def is_process_object_role(self, token) -> bool:
        """Check if token has process object dependency role."""
        return token.dep_ in ["dobj", "nmod"]
    
    def is_process_object(self, token) -> bool:
        """Check if token represents process object using morphology."""
        obj_lemma = token.lemma_.lower()
        
        # Process objects often have document/artifact patterns
        if 'version' in obj_lemma or 'document' in obj_lemma or 'report' in obj_lemma:
            return True
        
        if 'file' in obj_lemma or 'material' in obj_lemma or 'content' in obj_lemma:
            return True
        
        return False
    
    def appears_in_certainty_context(self, token, doc) -> bool:
        """Check certainty context using SpaCy dependency analysis."""
        # Certainty terms often modify abstract concepts
        for child in token.children:
            if self.is_certainty_object_role(child):
                if self.is_abstract_concept(child):
                    return True
        
        return False
    
    def is_certainty_object_role(self, token) -> bool:
        """Check if token has certainty object dependency role."""
        return token.dep_ in ["dobj", "nmod"]
    
    def is_abstract_concept(self, token) -> bool:
        """Check if token represents abstract concept."""
        lemma = token.lemma_.lower()
        
        # Abstract concepts often have these patterns
        if 'concept' in lemma or 'idea' in lemma or 'notion' in lemma:
            return True
        
        if 'principle' in lemma or 'theory' in lemma:
            return True
        
        return False
    
    def appears_in_cognitive_context(self, token, doc) -> bool:
        """Check cognitive context using SpaCy analysis."""
        # Look for cognitive verbs in the sentence
        for sent_token in token.sent:
            if sent_token.pos_ == "VERB":
                if self.is_cognitive_verb(sent_token):
                    return True
        
        return False
    
    def is_cognitive_verb(self, verb_token) -> bool:
        """Check if verb is cognitive using morphology."""
        lemma = verb_token.lemma_.lower()
        
        # Cognitive verbs have these patterns
        if 'know' in lemma or 'understand' in lemma or 'realiz' in lemma:
            return True
        
        if 'think' in lemma or 'believ' in lemma or 'consider' in lemma:
            return True
        
        return False 