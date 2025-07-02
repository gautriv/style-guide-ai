"""
Filler phrases detection module for conciseness analysis.
Detects temporal, purpose, and discourse filler phrases using pure SpaCy analysis.
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class FillerPhrasesDetector:
    """Detector for filler phrases using pure SpaCy morphological analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_filler_phrases(self, doc) -> List[Dict[str, Any]]:
        """Detect filler phrases using pure SpaCy morphological analysis."""
        filler_issues = []
        
        for token in doc:
            # Method 1: Detect temporal filler phrases using dependency analysis
            if self.is_temporal_filler_phrase(token, doc):
                filler_issue = self.analyze_temporal_filler(token, doc)
                if filler_issue:
                    filler_issues.append(filler_issue)
            
            # Method 2: Detect purpose filler phrases using syntactic patterns
            elif self.is_purpose_filler_phrase(token, doc):
                purpose_issue = self.analyze_purpose_filler(token, doc)
                if purpose_issue:
                    filler_issues.append(purpose_issue)
            
            # Method 3: Detect discourse marker fillers
            elif self.is_discourse_filler(token, doc):
                discourse_issue = self.analyze_discourse_filler(token, doc)
                if discourse_issue:
                    filler_issues.append(discourse_issue)
        
        return filler_issues
    
    def is_temporal_filler_phrase(self, token, doc) -> bool:
        """Detect temporal filler phrases using SpaCy morphological analysis."""
        # Use SpaCy's dependency parsing to identify temporal phrases
        if token.dep_ == "advmod" and token.pos_ == "ADV":
            # Check for temporal adverbs that can often be simplified
            temporal_patterns = self.analyze_temporal_pattern(token, doc)
            return temporal_patterns['is_filler']
        
        # Check for participial phrase fillers (e.g., "introduced recently")
        if token.pos_ == "VERB" and "Tense=Past" in str(token.morph):
            return self.is_participial_filler(token, doc)
        
        return False
    
    def analyze_temporal_pattern(self, adv_token, doc) -> Dict[str, Any]:
        """Analyze temporal pattern using SpaCy."""
        pattern_info = {
            'is_filler': False,
            'pattern_type': 'unknown',
            'simplification_potential': 0.0
        }
        
        # Use SpaCy's lemmatization for pattern detection
        lemma = adv_token.lemma_.lower()
        
        # Check for compound temporal constructions
        head_verb = adv_token.head
        if head_verb.pos_ == "VERB":
            # Look for past participle + temporal adverb patterns
            if "Tense=Past" in str(head_verb.morph) and "VerbForm=Part" in str(head_verb.morph):
                pattern_info['is_filler'] = True
                pattern_info['pattern_type'] = 'participial_temporal'
                pattern_info['simplification_potential'] = 0.8
        
        return pattern_info
    
    def is_participial_filler(self, verb_token, doc) -> bool:
        """Check if participial phrase is filler using SpaCy analysis."""
        # Look for past participles with temporal modifiers
        if "VerbForm=Part" in str(verb_token.morph):
            for child in verb_token.children:
                if child.dep_ == "advmod" and child.pos_ == "ADV":
                    # Use SpaCy's morphological analysis for temporal detection
                    if self.is_temporal_adverb(child):
                        return True
        
        return False
    
    def is_temporal_adverb(self, adverb_token) -> bool:
        """Check if adverb is temporal using morphological analysis."""
        lemma = adverb_token.lemma_.lower()
        
        # Temporal adverbs often have time-related morphological patterns
        if 'recent' in lemma or 'late' in lemma or 'new' in lemma:
            return True
        
        # Check for temporal morphological endings
        if lemma.endswith('ly') and ('time' in lemma or 'recent' in lemma):
            return True
        
        return False
    
    def is_purpose_filler_phrase(self, token, doc) -> bool:
        """Detect purpose filler phrases using SpaCy syntactic analysis."""
        # Look for infinitive purpose clauses that add little value
        if token.dep_ == "advcl" and token.pos_ == "VERB":
            # Check for purpose infinitives
            if "VerbForm=Inf" in str(token.morph):
                return self.is_redundant_purpose_clause(token, doc)
        
        return False
    
    def is_redundant_purpose_clause(self, token, doc) -> bool:
        """Check if purpose clause is redundant using SpaCy analysis."""
        # Analyze the purpose content
        purpose_analysis = self.analyze_purpose_content(token, doc)
        return purpose_analysis.get('is_redundant', False)
    
    def analyze_purpose_content(self, purpose_token, doc) -> Dict[str, Any]:
        """Analyze purpose clause content using SpaCy."""
        content_analysis = {
            'is_redundant': False,
            'redundancy_type': 'unknown',
            'semantic_value': 0.0
        }
        
        # Use SpaCy's lemmatization to identify common filler purposes
        if self.is_filler_purpose_verb(purpose_token):
            # Check if the purpose is vague or already implied
            semantic_specificity = self.calculate_purpose_specificity(purpose_token, doc)
            if semantic_specificity < 0.3:
                content_analysis['is_redundant'] = True
                content_analysis['redundancy_type'] = 'vague_purpose'
        
        return content_analysis
    
    def is_filler_purpose_verb(self, verb_token) -> bool:
        """Check if verb is filler purpose using morphological analysis."""
        lemma = verb_token.lemma_.lower()
        
        # Filler purposes often have vague action morphology
        if 'address' in lemma or 'handle' in lemma or 'deal' in lemma:
            return True
        
        if 'tackle' in lemma:
            return True
        
        return False
    
    def calculate_purpose_specificity(self, purpose_token, doc) -> float:
        """Calculate purpose specificity using SpaCy analysis."""
        specificity_score = 0.0
        
        # Check for specific objects or complements
        for child in purpose_token.children:
            if child.dep_ in ["dobj", "attr", "prep"]:
                specificity_score += 0.3
                
                # Bonus for named entities
                if child.ent_type_:
                    specificity_score += 0.2
        
        return min(specificity_score, 1.0)
    
    def is_discourse_filler(self, token, doc) -> bool:
        """Detect discourse marker fillers using SpaCy analysis."""
        # Look for discourse markers that add little value
        if token.dep_ == "advmod" and token.pos_ == "ADV":
            return self.is_discourse_marker_adverb(token)
        
        return False
    
    def is_discourse_marker_adverb(self, token) -> bool:
        """Check if adverb is discourse marker using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Discourse markers often have minimizing/hedging morphology
        if 'basic' in lemma or 'essential' in lemma or 'actual' in lemma:
            return True
        
        if 'real' in lemma:
            return True
        
        return False
    
    def analyze_temporal_filler(self, token, doc) -> Dict[str, Any]:
        """Analyze temporal filler phrase using SpaCy analysis."""
        analysis = {
            'type': 'temporal_filler',
            'token': token,
            'simplification_potential': 0.0,
            'redundancy_type': 'unknown',
            'position': token.idx
        }
        
        # Analyze the temporal pattern
        if token.dep_ == "advmod" and token.pos_ == "ADV":
            temporal_pattern = self.analyze_temporal_pattern(token, doc)
            if temporal_pattern['is_filler']:
                analysis['simplification_potential'] = temporal_pattern['simplification_potential']
                analysis['redundancy_type'] = temporal_pattern['pattern_type']
                return analysis
        
        # Check for participial temporal constructions
        elif token.pos_ == "VERB" and "Tense=Past" in str(token.morph):
            if self.is_participial_filler(token, doc):
                analysis['redundancy_type'] = 'participial_temporal'
                analysis['simplification_potential'] = 0.7
                return analysis
        
        return None
    
    def analyze_purpose_filler(self, token, doc) -> Dict[str, Any]:
        """Analyze purpose filler phrase using SpaCy analysis."""
        analysis = {
            'type': 'purpose_filler',
            'token': token,
            'redundancy_type': 'unknown',
            'position': token.idx
        }
        
        # Analyze purpose clause redundancy
        if token.dep_ == "advcl" and token.pos_ == "VERB":
            if "VerbForm=Inf" in str(token.morph):
                purpose_content = self.analyze_purpose_content(token, doc)
                if purpose_content['is_redundant']:
                    analysis['redundancy_type'] = purpose_content['redundancy_type']
                    analysis['semantic_value'] = purpose_content['semantic_value']
                    return analysis
        
        return None
    
    def analyze_discourse_filler(self, token, doc) -> Dict[str, Any]:
        """Analyze discourse marker filler using SpaCy analysis."""
        analysis = {
            'type': 'discourse_filler',
            'token': token,
            'redundancy_type': 'discourse_marker',
            'position': token.idx
        }
        
        # Check if this is indeed a discourse filler
        if token.dep_ == "advmod" and token.pos_ == "ADV":
            if self.is_discourse_marker_adverb(token):
                return analysis
        
        return None
    
    def generate_filler_suggestions(self, filler_issue: Dict[str, Any], doc) -> List[str]:
        """Generate filler phrase suggestions using SpaCy analysis."""
        suggestions = []
        issue_type = filler_issue.get('type')
        token = filler_issue.get('token')
        
        if issue_type == 'temporal_filler':
            # Use SpaCy to suggest simplifications
            if token.dep_ == "advmod" and token.head.pos_ == "VERB":
                head_verb = token.head
                if "VerbForm=Part" in str(head_verb.morph):
                    # "introduced recently" → "recent"
                    suggestions.append(f"Simplify '{head_verb.text} {token.text}' to 'recent {head_verb.head.text}'")
                    suggestions.append(f"Consider: 'recent' instead of '{head_verb.text} {token.text}'")
        
        elif issue_type == 'purpose_filler':
            # Generate purpose simplification suggestions
            suggestions.append(f"Consider removing '{token.text}' if the purpose is already clear")
            suggestions.append("This phrase may not add meaningful information")
        
        elif issue_type == 'discourse_filler':
            suggestions.append(f"Remove discourse filler '{token.text}' for conciseness")
            suggestions.append("Technical writing benefits from direct statements")
        
        return suggestions 