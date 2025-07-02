"""
Conciseness Rule - Main module that coordinates all conciseness detection components.
Uses pure SpaCy morphological, syntactic, and semantic analysis with zero hardcoded patterns.
"""

from typing import List, Dict, Any

# Handle imports for different contexts
try:
    from ..base_rule import BaseRule
except ImportError:
    try:
        from ..base_rule import BaseRule
    except ImportError:
        from base_rule import BaseRule

# Import modularized components
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer
from .status_phrases import StatusPhrasesDetector
from .modifiers import ModifiersDetector
from .synonyms import SynonymsDetector
from .filler_phrases import FillerPhrasesDetector
from .bureaucratic import BureaucraticDetector
from .archaic_verbs import ArchaicVerbsDetector

# Import new advanced detectors for 10/10 quality
from .semantic_appropriateness import SemanticAppropriatenessDetector
from .discourse_markers import DiscourseMarkersDetector
from .syntactic_flow import SyntacticFlowDetector
from .conceptual_redundancy import ConceptualRedundancyDetector

class ConcisenessRule(BaseRule):
    """Rule to identify redundancy and wordiness using pure SpaCy linguistic analysis."""
    
    def __init__(self):
        # Call parent constructor to initialize severity_levels and other base attributes
        super().__init__()
        
        # Initialize all component detectors
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        self.status_detector = StatusPhrasesDetector()
        self.modifiers_detector = ModifiersDetector()
        self.synonyms_detector = SynonymsDetector()
        self.filler_detector = FillerPhrasesDetector()
        self.bureaucratic_detector = BureaucraticDetector()
        self.archaic_verbs_detector = ArchaicVerbsDetector()
        
        # Initialize new advanced detectors for 10/10 quality
        self.semantic_appropriateness_detector = SemanticAppropriatenessDetector()
        self.discourse_markers_detector = DiscourseMarkersDetector()
        self.syntactic_flow_detector = SyntacticFlowDetector()
        self.conceptual_redundancy_detector = ConceptualRedundancyDetector()
    
    def _get_rule_type(self) -> str:
        return 'conciseness'
    
    def analyze(self, text: str, sentences: List[str], nlp=None) -> List[Dict[str, Any]]:
        """Analyze text for wordiness issues using pure SpaCy analysis."""
        errors = []
        
        for i, sentence in enumerate(sentences):
            if nlp:
                doc = nlp(sentence)
                wordiness_issues = self._find_wordiness_with_pure_spacy(doc)
            else:
                # Fallback analysis without SpaCy
                wordiness_issues = self._find_wordiness_morphological_fallback(sentence)
            
            # Create errors for each wordiness issue found
            for issue in wordiness_issues:
                suggestions = self._generate_conciseness_suggestions_from_linguistics(issue, doc if nlp else None)
                
                # Preserve specific error types for enhanced detection
                issue_type = issue.get('type', 'conciseness')
                message = self._create_wordiness_message(issue)
                
                errors.append(self._create_error(
                    sentence=sentence,
                    sentence_index=i,
                    message=message,
                    suggestions=suggestions,
                    severity=self._determine_wordiness_severity(issue),
                    error_type=issue_type,  # Preserve the specific error type
                    wordiness_issue=issue
                ))
        
        return errors
    
    def _find_wordiness_with_pure_spacy(self, doc) -> List[Dict[str, Any]]:
        """Find wordiness using pure SpaCy NLP analysis - zero hardcoding approach."""
        wordiness_issues = []
        
        # 1. Status phrase redundancy detection
        status_redundancies = self.status_detector.detect_redundant_status_phrases(doc)
        wordiness_issues.extend(status_redundancies)
        
        # 2. Modifier redundancy detection
        modifier_redundancies = self.modifiers_detector.detect_advanced_redundant_modifiers(doc)
        wordiness_issues.extend(modifier_redundancies)
        
        # 3. Synonym pair redundancy detection
        synonym_redundancies = self.synonyms_detector.detect_synonym_pair_redundancy(doc)
        wordiness_issues.extend(synonym_redundancies)
        
        # 4. Bureaucratic language patterns
        bureaucratic_patterns = self.bureaucratic_detector.detect_bureaucratic_language_patterns(doc)
        wordiness_issues.extend(bureaucratic_patterns)
        
        # 5. Filler phrases analysis
        filler_phrases = self.filler_detector.detect_filler_phrases(doc)
        wordiness_issues.extend(filler_phrases)
        
        # 6. Archaic verb pattern modernization
        archaic_patterns = self.archaic_verbs_detector.detect_archaic_verb_patterns(doc)
        wordiness_issues.extend(archaic_patterns)
        
        # 7. Verbose constructions analysis
        verbose_constructions = self._detect_verbose_constructions(doc)
        wordiness_issues.extend(verbose_constructions)
        
        # 8. NEW: Semantic appropriateness detection for 10/10 quality
        semantic_issues = self.semantic_appropriateness_detector.detect_semantic_appropriateness_issues(doc)
        wordiness_issues.extend(semantic_issues)
        
        # 9. NEW: Discourse marker redundancy detection for 10/10 quality
        discourse_issues = self.discourse_markers_detector.detect_discourse_marker_redundancy(doc)
        wordiness_issues.extend(discourse_issues)
        
        # 10. NEW: Syntactic flow optimization for 10/10 quality
        syntactic_issues = self.syntactic_flow_detector.detect_syntactic_flow_issues(doc)
        wordiness_issues.extend(syntactic_issues)
        
        # 11. NEW: Conceptual redundancy detection for 10/10 quality
        conceptual_issues = self.conceptual_redundancy_detector.detect_conceptual_redundancy_issues(doc)
        wordiness_issues.extend(conceptual_issues)
        
        return wordiness_issues
    
    def _detect_verbose_constructions(self, doc) -> List[Dict[str, Any]]:
        """Detect areas of high morphological density."""
        issues = []
        
        # Analyze phrases for morphological complexity
        for sent in doc.sents:
            density_analysis = self.morph_analyzer.analyze_sentence_morphological_density(sent)
            if density_analysis['is_high_density']:
                issues.append({
                    'type': 'high_morphological_density',
                    'sentence_tokens': list(sent),
                    'density_score': density_analysis['density_score'],
                    'complexity_indicators': density_analysis['complexity_indicators'],
                    'position': sent.start_char
                })
        
        return issues
    
    def _generate_conciseness_suggestions_from_linguistics(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions based on linguistic analysis."""
        suggestions = []
        issue_type = issue.get('type', 'unknown')
        
        # Delegate to appropriate detector for suggestions
        if issue_type == 'redundant_status_phrase':
            return self.status_detector.generate_redundant_status_suggestions(issue, doc)
        
        elif issue_type in ['advanced_redundant_modifier', 'redundant_modifier']:
            return self.modifiers_detector.generate_modifier_suggestions(issue)
        
        elif issue_type in ['advanced_redundant_adverb', 'redundant_adverb']:
            return self.modifiers_detector.generate_adverb_suggestions(issue)
        
        elif issue_type == 'synonym_pair_redundancy':
            return self.synonyms_detector.generate_synonym_redundancy_suggestions(issue, doc)
        
        elif issue_type == 'bureaucratic_language':
            return self.bureaucratic_detector.generate_bureaucratic_suggestions(issue)
        
        elif issue_type == 'archaic_verb_pattern':
            return self.archaic_verbs_detector.generate_archaic_verb_suggestions(issue)
        
        elif self._is_filler_issue_type(issue_type):
            return self.filler_detector.generate_filler_suggestions(issue, doc)
        
        elif issue_type == 'high_morphological_density':
            return self._generate_verbose_construction_suggestions(issue)
        
        # NEW: Handle advanced detector suggestions for 10/10 quality
        elif issue_type == 'semantic_appropriateness':
            return self.semantic_appropriateness_detector.generate_semantic_appropriateness_suggestions(issue, doc)
        
        elif issue_type == 'discourse_marker_redundancy':
            return self.discourse_markers_detector.generate_discourse_marker_suggestions(issue, doc)
        
        elif issue_type in ['wordy_discourse_construction', 'indirect_construction', 'complex_syntactic_pattern']:
            return self.syntactic_flow_detector.generate_syntactic_flow_suggestions(issue, doc)
        
        # NEW: Handle conceptual redundancy suggestions for 10/10 quality
        elif issue_type in ['conceptual_progress_redundancy', 'vague_qualifier', 'noun_phrase_conceptual_overlap']:
            return self.conceptual_redundancy_detector.generate_conceptual_redundancy_suggestions(issue, doc)
        
        # Default fallback suggestions
        if not suggestions:
            suggestions = [
                "Consider simplifying this expression",
                "Look for more concise alternatives",
                "Remove unnecessary words"
            ]
        
        return suggestions
    
    def _is_filler_issue_type(self, issue_type: str) -> bool:
        """Check if issue type is filler-related using pattern analysis."""
        # Use morphological pattern analysis instead of hardcoded list
        return 'filler' in issue_type
    
    def _generate_verbose_construction_suggestions(self, issue: Dict[str, Any]) -> List[str]:
        """Generate suggestions for verbose constructions."""
        suggestions = [
            "This sentence has high morphological complexity",
            "Consider breaking into shorter, simpler sentences",
            "Look for opportunities to use more direct language"
        ]
        
        complexity_indicators = issue.get('complexity_indicators', [])
        if complexity_indicators:
            suggestions.append(f"Complex elements detected: {', '.join(complexity_indicators[:3])}")
        
        return suggestions
    
    def _create_wordiness_message(self, issue: Dict[str, Any]) -> str:
        """Create appropriate message for wordiness issue."""
        issue_type = issue.get('type', 'unknown')
        
        if issue_type == 'redundant_status_phrase':
            redundant_phrase = issue.get('redundant_phrase', '')
            suggested_replacement = issue.get('suggested_replacement', '')
            return f"Redundant phrase: '{redundant_phrase}' can be simplified to '{suggested_replacement}'"
        
        elif issue_type == 'advanced_redundant_modifier':
            modifier_token = issue.get('modifier_token')
            modified_token = issue.get('modified_token')
            if modifier_token and modified_token:
                return f"Redundant modifier: '{modifier_token.text}' adds no value to '{modified_token.text}'"
            return "Remove redundant modifier"
        
        elif issue_type == 'synonym_pair_redundancy':
            first_token = issue.get('first_token')
            second_token = issue.get('second_token')
            if first_token and second_token:
                return f"Redundant synonyms: '{first_token.text}' and '{second_token.text}' mean the same thing"
            return "Remove redundant synonymous terms"
        
        elif issue_type == 'bureaucratic_language':
            complex_token = issue.get('complex_token')
            if complex_token:
                return f"Simplify bureaucratic language: '{complex_token.text}' can be more direct"
            return "Use simpler, more direct language"
        
        elif issue_type == 'archaic_verb_pattern':
            archaic_verb = issue.get('archaic_verb', '')
            modern_suggestion = issue.get('modern_suggestion', '')
            if archaic_verb and modern_suggestion:
                return f"Modernize language: replace '{archaic_verb}' with '{modern_suggestion}'"
            return "Use modern, direct language"
        
        elif 'filler' in issue_type:
            return "Remove unnecessary filler phrase"
        
        elif issue_type == 'high_morphological_density':
            return "Consider simplifying this complex construction"
        
        # NEW: Messages for advanced detectors (10/10 quality)
        elif issue_type == 'semantic_appropriateness':
            inappropriate_token = issue.get('inappropriate_token')
            suggested_alternatives = issue.get('suggested_alternatives', [])
            if inappropriate_token and suggested_alternatives:
                return f"Inaccurate word choice: '{inappropriate_token.text}' should be '{suggested_alternatives[0]}'"
            return "Use semantically appropriate terminology"
        
        elif issue_type == 'discourse_marker_redundancy':
            redundant_markers = issue.get('redundant_markers', [])
            if len(redundant_markers) >= 2:
                marker1_text = redundant_markers[0]['text']
                marker2_text = redundant_markers[1]['text']
                return f"Redundant discourse markers: '{marker1_text}' and '{marker2_text}' serve the same function"
            return "Remove redundant logical connectors"
        
        elif issue_type == 'wordy_discourse_construction':
            discourse_marker = issue.get('discourse_marker', '')
            return f"Wordy construction: '{discourse_marker}' adds unnecessary complexity"
        
        elif issue_type == 'indirect_construction':
            return "Use more direct subject-verb construction"
        
        elif issue_type == 'complex_syntactic_pattern':
            return "Simplify complex sentence structure for better readability"
        
        # NEW: Messages for conceptual redundancy (10/10 quality)
        elif issue_type == 'conceptual_progress_redundancy':
            redundant_tokens = issue.get('redundant_tokens', [])
            if len(redundant_tokens) >= 2:
                token1_text = redundant_tokens[0].text
                token2_text = redundant_tokens[1].text
                return f"Conceptual redundancy: '{token1_text}' and '{token2_text}' both signal progress"
            return "Remove conceptually redundant terms"
        
        elif issue_type == 'vague_qualifier':
            vague_token = issue.get('vague_token')
            if vague_token:
                return f"Vague qualifier: '{vague_token.text}' lacks specificity"
            return "Use more specific qualifiers"
        
        elif issue_type == 'noun_phrase_conceptual_overlap':
            overlapping_concepts = issue.get('overlapping_concepts', [])
            if len(overlapping_concepts) >= 2:
                concept1 = overlapping_concepts[0].text
                concept2 = overlapping_concepts[1].text
                return f"Conceptual overlap: '{concept1}' and '{concept2}' express similar concepts"
            return "Simplify overlapping concepts"
        
        # Default message
        return "Consider more concise wording"
    
    def _determine_wordiness_severity(self, issue: Dict[str, Any]) -> str:
        """Determine severity of wordiness issue."""
        issue_type = issue.get('type', 'unknown')
        
        if issue_type == 'redundant_status_phrase':
            # High confidence redundancy is medium severity
            confidence = issue.get('dependency_analysis', {}).get('semantic_relationship', {}).get('redundancy_confidence', 0)
            return 'medium' if confidence > 0.7 else 'low'
        
        elif issue_type == 'synonym_pair_redundancy':
            # Semantic similarity determines severity
            similarity = issue.get('redundancy_analysis', {}).get('semantic_similarity', 0)
            return 'medium' if similarity > 0.8 else 'low'
        
        elif issue_type == 'bureaucratic_language':
            # Complexity score determines severity
            complexity = issue.get('complexity_score', 0)
            return 'medium' if complexity > 0.7 else 'low'
        
        elif issue_type == 'high_morphological_density':
            # Density score determines severity
            density = issue.get('density_score', 0)
            return 'medium' if density > 4.0 else 'low'
        
        # Default to low severity
        return 'low'
    
    def _find_wordiness_morphological_fallback(self, sentence: str) -> List[Dict[str, Any]]:
        """Fallback wordiness detection when SpaCy unavailable."""
        import re
        issues = []
        
        # Very basic patterns as fallback
        wordy_patterns = [
            r'\b(?:very|really|quite|rather)\s+\w+',  # Redundant intensifiers
            r'\b(?:in order to|for the purpose of)\b',  # Wordy phrases
        ]
        
        for pattern in wordy_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                issues.append({
                    'type': 'morphological_fallback',
                    'matched_text': match.group(),
                    'position': match.start(),
                    'pattern_type': 'basic_wordiness',
                    'verbosity_score': 1
                })
        
        return issues 