"""
Stylistic Flow Rule - Detects flow issues using morphological analysis with linguistic anchors.
Targets: Dangling participles, formal transitions, complex relatives, causal connections.
"""

from typing import List, Dict, Any, Optional
import re

# Handle imports for different contexts
try:
    from .base_rule import BaseRule
except ImportError:
    from base_rule import BaseRule

class StylisticFlowRule(BaseRule):
    """Rule to detect stylistic flow issues using morphological analysis with linguistic anchors."""
    
    def _get_rule_type(self) -> str:
        return 'stylistic_flow'
    
    def analyze(self, text: str, sentences: List[str], nlp=None) -> List[Dict[str, Any]]:
        """Analyze text for stylistic flow issues using morphological analysis."""
        errors = []
        
        for i, sentence in enumerate(sentences):
            if nlp:
                doc = nlp(sentence)
                
                # 1. Detect dangling participles using morphological anchors
                dangling_issues = self._detect_dangling_participles(doc)
                
                # 2. Detect overly formal transitions using morphological patterns
                formal_transition_issues = self._detect_formal_transitions(doc)
                
                # 3. Detect complex relative clauses using syntactic anchors
                complex_relative_issues = self._detect_complex_relatives(doc)
                
                # 4. Detect missing causal connections using semantic anchors
                causal_issues = self._detect_weak_causal_connections(doc)
                
                all_issues = dangling_issues + formal_transition_issues + complex_relative_issues + causal_issues
            else:
                # Fallback: Use morphological pattern matching
                all_issues = self._detect_flow_issues_morphological_fallback(sentence)
            
            for issue in all_issues:
                suggestions = self._generate_flow_suggestions(issue, doc if nlp else None)
                errors.append(self._create_error(
                    sentence=sentence,
                    sentence_index=i,
                    message=issue.get('message', 'Stylistic flow issue detected'),
                    suggestions=suggestions,
                    severity=issue.get('severity', 'medium'),
                    issue_details=issue
                ))
        
        return errors
    
    def _detect_dangling_participles(self, doc) -> List[Dict[str, Any]]:
        """Detect dangling participles using morphological analysis with linguistic anchors."""
        issues = []
        
        for token in doc:
            # Look for sentence-initial participial phrases
            if self._is_sentence_initial_participle(token, doc):
                dangling_analysis = self._analyze_participle_anchor(token, doc)
                if dangling_analysis['is_dangling']:
                    issues.append({
                        'type': 'dangling_participle',
                        'token': token,
                        'message': f'Dangling participle: "{token.text}" may not clearly modify the intended subject',
                        'severity': 'medium',
                        'anchor_analysis': dangling_analysis,
                        'position': token.idx
                    })
        
        return issues
    
    def _is_sentence_initial_participle(self, token, doc) -> bool:
        """Check if token is a sentence-initial participle using morphological anchors."""
        # Must be a participle
        if not self._is_participle_by_morphology(token):
            return False
        
        # Must be at sentence beginning or after minimal punctuation
        sentence_start_threshold = 3  # Within first 3 tokens
        if token.i > sentence_start_threshold:
            return False
        
        # Check if it's in a participial phrase structure
        return self._has_participial_phrase_structure(token, doc)
    
    def _is_participle_by_morphology(self, token) -> bool:
        """Check if token is participle using SpaCy morphological features."""
        # Past participle
        if (token.morph.get("VerbForm") == ["Part"] and 
            token.morph.get("Tense") == ["Past"]):
            return True
        
        # Present participle (gerund forms used as adjectives)
        if (token.morph.get("VerbForm") == ["Part"] and 
            token.morph.get("Tense") == ["Pres"]):
            return True
        
        # Tag-based fallback
        if token.tag_ in ["VBN", "VBG"]:
            return True
        
        return False
    
    def _has_participial_phrase_structure(self, token, doc) -> bool:
        """Check for participial phrase structure using syntactic anchors."""
        # Look for comma after participial phrase
        comma_found = False
        for i in range(token.i, min(token.i + 10, len(doc))):
            if doc[i].text == ",":
                comma_found = True
                break
        
        return comma_found
    
    def _analyze_participle_anchor(self, participle_token, doc) -> Dict[str, Any]:
        """Analyze participle's linguistic anchor to subject."""
        analysis = {
            'is_dangling': False,
            'subject_token': None,
            'anchor_distance': 0,
            'anchor_clarity': 'clear'
        }
        
        # Find the main clause subject
        main_subject = self._find_main_clause_subject(participle_token, doc)
        if not main_subject:
            analysis['is_dangling'] = True
            analysis['anchor_clarity'] = 'missing_subject'
            return analysis
        
        # Check semantic compatibility between participle and subject
        semantic_match = self._check_semantic_anchor_match(participle_token, main_subject)
        if not semantic_match:
            analysis['is_dangling'] = True
            analysis['anchor_clarity'] = 'semantic_mismatch'
        
        analysis['subject_token'] = main_subject
        analysis['anchor_distance'] = abs(participle_token.i - main_subject.i)
        
        return analysis
    
    def _find_main_clause_subject(self, participle_token, doc) -> Optional[object]:
        """Find main clause subject using syntactic analysis."""
        # Look for the root verb and its subject
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        return child
        return None
    
    def _check_semantic_anchor_match(self, participle, subject) -> bool:
        """Check semantic compatibility between participle and subject."""
        # Use morphological features to check compatibility
        participle_features = self._get_morphological_features(participle)
        subject_features = self._get_morphological_features(subject)
        
        # Check if subject can logically perform the participle action
        if self._is_animate_subject(subject) and self._is_action_requiring_agency(participle):
            return True
        
        if self._is_inanimate_subject(subject) and self._is_state_participle(participle):
            return True
        
        return False
    
    def _is_animate_subject(self, subject_token) -> bool:
        """Check if subject is animate using morphological analysis."""
        # Use SpaCy's entity recognition
        if hasattr(subject_token, 'ent_type_') and subject_token.ent_type_ in ["PERSON", "ORG"]:
            return True
        
        # Use morphological patterns for animate nouns
        lemma = subject_token.lemma_.lower()
        animate_patterns = ['engineer', 'team', 'manager', 'developer', 'analyst', 'user']
        return any(pattern in lemma for pattern in animate_patterns)
    
    def _is_inanimate_subject(self, subject_token) -> bool:
        """Check if subject is inanimate using morphological analysis."""
        return not self._is_animate_subject(subject_token)
    
    def _is_action_requiring_agency(self, participle_token) -> bool:
        """Check if participle requires an animate agent."""
        lemma = participle_token.lemma_.lower()
        agency_verbs = ['analyze', 'discover', 'initiate', 'update', 'resolve', 'implement']
        return any(verb in lemma for verb in agency_verbs)
    
    def _is_state_participle(self, participle_token) -> bool:
        """Check if participle describes a state rather than action."""
        lemma = participle_token.lemma_.lower()
        state_patterns = ['resolve', 'flag', 'affect', 'impact', 'cause']
        return any(pattern in lemma for pattern in state_patterns)
    
    def _detect_formal_transitions(self, doc) -> List[Dict[str, Any]]:
        """Detect overly formal transition words using morphological patterns."""
        issues = []
        
        formal_transitions = {
            'furthermore': {'casual_alternatives': ['also', 'additionally', 'moreover'], 'formality_score': 0.9},
            'nevertheless': {'casual_alternatives': ['however', 'but', 'still'], 'formality_score': 0.8},
            'consequently': {'casual_alternatives': ['so', 'therefore', 'as a result'], 'formality_score': 0.8},
            'subsequently': {'casual_alternatives': ['then', 'next', 'later'], 'formality_score': 0.8},
            'therefore': {'casual_alternatives': ['so', 'thus'], 'formality_score': 0.7}
        }
        
        for token in doc:
            lemma_lower = token.lemma_.lower()
            if lemma_lower in formal_transitions:
                # Check if it's at sentence beginning (formal transition position)
                if self._is_transition_position(token, doc):
                    formality_data = formal_transitions[lemma_lower]
                    issues.append({
                        'type': 'formal_transition',
                        'token': token,
                        'message': f'Consider using less formal transition than "{token.text}"',
                        'severity': 'low',
                        'formality_score': formality_data['formality_score'],
                        'alternatives': formality_data['casual_alternatives'],
                        'position': token.idx
                    })
        
        return issues
    
    def _is_transition_position(self, token, doc) -> bool:
        """Check if token is in transition position using positional analysis."""
        # Beginning of sentence
        if token.i <= 2:
            return True
        
        # After sentence-internal punctuation
        if token.i > 0 and doc[token.i - 1].text in [",", ";", ":"]:
            return True
        
        return False
    
    def _detect_complex_relatives(self, doc) -> List[Dict[str, Any]]:
        """Detect complex relative clauses using syntactic anchors."""
        issues = []
        
        for token in doc:
            if self._is_complex_relative_clause(token, doc):
                complexity_analysis = self._analyze_relative_complexity(token, doc)
                if complexity_analysis['complexity_score'] > 0.6:
                    issues.append({
                        'type': 'complex_relative',
                        'token': token,
                        'message': 'Consider simplifying this complex relative clause',
                        'severity': 'medium',
                        'complexity_analysis': complexity_analysis,
                        'position': token.idx
                    })
        
        return issues
    
    def _is_complex_relative_clause(self, token, doc) -> bool:
        """Check for complex relative clause patterns."""
        # Look for relative pronouns
        if token.lemma_.lower() in ['who', 'which', 'that'] and token.dep_ in ['nsubj', 'nsubjpass', 'dobj']:
            return True
        
        # Look for complex embedding patterns
        if token.dep_ == 'relcl':
            return True
        
        return False
    
    def _analyze_relative_complexity(self, relative_token, doc) -> Dict[str, Any]:
        """Analyze complexity of relative clause using morphological anchors."""
        analysis = {
            'complexity_score': 0.0,
            'embedded_depth': 0,
            'clause_length': 0,
            'has_nested_structure': False
        }
        
        # Find the extent of the relative clause
        clause_start = relative_token.i
        clause_end = self._find_clause_end(relative_token, doc)
        clause_length = clause_end - clause_start
        
        analysis['clause_length'] = clause_length
        
        # Calculate complexity score
        if clause_length > 8:
            analysis['complexity_score'] += 0.4
        if clause_length > 12:
            analysis['complexity_score'] += 0.3
        
        # Check for nested structures
        if self._has_nested_clauses(relative_token, doc, clause_end):
            analysis['has_nested_structure'] = True
            analysis['complexity_score'] += 0.3
        
        return analysis
    
    def _find_clause_end(self, start_token, doc) -> int:
        """Find the end of relative clause using punctuation anchors."""
        for i in range(start_token.i + 1, len(doc)):
            if doc[i].text in [",", ".", "!", "?", ";"]:
                return i
        return len(doc)
    
    def _has_nested_clauses(self, relative_token, doc, clause_end) -> bool:
        """Check for nested clause structures."""
        nested_indicators = ['which', 'that', 'who', 'where', 'when']
        for i in range(relative_token.i + 1, clause_end):
            if doc[i].lemma_.lower() in nested_indicators:
                return True
        return False
    
    def _detect_weak_causal_connections(self, doc) -> List[Dict[str, Any]]:
        """Detect weak causal connections using semantic anchors."""
        issues = []
        
        for sent_boundary in self._find_sentence_boundaries(doc):
            causal_strength = self._analyze_causal_strength(sent_boundary, doc)
            if causal_strength['needs_enhancement']:
                issues.append({
                    'type': 'weak_causal_connection',
                    'position': sent_boundary['position'],
                    'message': 'Consider strengthening the logical connection between ideas',
                    'severity': 'low',
                    'causal_analysis': causal_strength
                })
        
        return issues
    
    def _find_sentence_boundaries(self, doc) -> List[Dict[str, Any]]:
        """Find sentence boundaries for causal analysis."""
        boundaries = []
        for i, token in enumerate(doc):
            if token.text in [".", "!", "?"] and i < len(doc) - 3:
                boundaries.append({
                    'position': i,
                    'before_token': token,
                    'after_start': min(i + 1, len(doc) - 1)
                })
        return boundaries
    
    def _analyze_causal_strength(self, boundary, doc) -> Dict[str, Any]:
        """Analyze causal connection strength using morphological patterns."""
        analysis = {
            'needs_enhancement': False,
            'causal_indicators': [],
            'connection_strength': 0.0
        }
        
        # Look for causal indicators in the next sentence
        next_sent_start = boundary['after_start']
        causal_words = ['because', 'since', 'therefore', 'thus', 'consequently', 'as a result', 'due to']
        
        # Check first 5 tokens of next sentence
        for i in range(next_sent_start, min(next_sent_start + 5, len(doc))):
            token_text = doc[i].lemma_.lower()
            if any(causal in token_text for causal in causal_words):
                analysis['causal_indicators'].append(token_text)
                analysis['connection_strength'] += 0.3
        
        # If no causal indicators and sentences seem related, suggest enhancement
        if analysis['connection_strength'] < 0.2:
            semantic_relation = self._check_semantic_relation(boundary, doc)
            if semantic_relation:
                analysis['needs_enhancement'] = True
        
        return analysis
    
    def _check_semantic_relation(self, boundary, doc) -> bool:
        """Check if sentences are semantically related using morphological overlap."""
        if boundary['after_start'] >= len(doc) - 3:
            return False
        
        # Get key content words from both sentences
        before_words = self._extract_content_words(doc, max(0, boundary['position'] - 10), boundary['position'])
        after_words = self._extract_content_words(doc, boundary['after_start'], min(boundary['after_start'] + 10, len(doc)))
        
        # Check for lexical overlap
        overlap = len(set(before_words) & set(after_words))
        return overlap >= 1
    
    def _extract_content_words(self, doc, start, end) -> List[str]:
        """Extract content words using morphological filtering."""
        content_words = []
        for i in range(start, end):
            if i < len(doc):
                token = doc[i]
                if (token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and 
                    not token.is_stop and 
                    token.is_alpha):
                    content_words.append(token.lemma_.lower())
        return content_words
    
    def _detect_flow_issues_morphological_fallback(self, sentence: str) -> List[Dict[str, Any]]:
        """Fallback detection using morphological patterns when SpaCy unavailable."""
        issues = []
        
        # Pattern 1: Dangling participles
        dangling_pattern = r'^([A-Z][a-z]*ed|[A-Z][a-z]*ing)\s*,?\s*(.+)'
        if re.match(dangling_pattern, sentence.strip()):
            issues.append({
                'type': 'dangling_participle',
                'message': 'Possible dangling participle at sentence start',
                'severity': 'medium',
                'position': 0
            })
        
        # Pattern 2: Formal transitions
        formal_words = ['Furthermore', 'Nevertheless', 'Consequently', 'Subsequently']
        for word in formal_words:
            if sentence.strip().startswith(word):
                issues.append({
                    'type': 'formal_transition',
                    'message': f'Consider less formal alternative to "{word}"',
                    'severity': 'low',
                    'position': sentence.find(word)
                })
        
        return issues
    
    def _generate_flow_suggestions(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions based on issue type and morphological analysis."""
        issue_type = issue.get('type', '')
        suggestions = []
        
        if issue_type == 'dangling_participle':
            suggestions.extend(self._generate_dangling_participle_suggestions(issue, doc))
        elif issue_type == 'formal_transition':
            suggestions.extend(self._generate_formal_transition_suggestions(issue, doc))
        elif issue_type == 'complex_relative':
            suggestions.extend(self._generate_complex_relative_suggestions(issue, doc))
        elif issue_type == 'weak_causal_connection':
            suggestions.extend(self._generate_causal_connection_suggestions(issue, doc))
        
        return suggestions
    
    def _generate_dangling_participle_suggestions(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions for dangling participle fixes."""
        suggestions = [
            "Move the participial phrase closer to the word it modifies",
            "Rewrite to clearly identify what the participle describes",
            "Consider: 'These issues, previously thought resolved, reappeared...'",
            "Use active voice to clarify the relationship"
        ]
        
        if doc and 'anchor_analysis' in issue:
            anchor = issue['anchor_analysis']
            if anchor.get('subject_token'):
                subject_text = anchor['subject_token'].text
                suggestions.append(f"Clarify that '{subject_text}' is what the participle modifies")
        
        return suggestions
    
    def _generate_formal_transition_suggestions(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions for formal transition alternatives."""
        suggestions = []
        
        if 'alternatives' in issue:
            alternatives = issue['alternatives']
            suggestions.append(f"Try: {', '.join(alternatives)}")
        
        suggestions.extend([
            "Use simpler, more direct transitions",
            "Consider omitting the transition if the connection is clear",
            "Match formality level to your audience"
        ])
        
        return suggestions
    
    def _generate_complex_relative_suggestions(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions for simplifying complex relatives."""
        suggestions = [
            "Break into two sentences for clarity",
            "Use coordination instead of subordination",
            "Place the relative clause closer to its antecedent",
            "Consider: 'Lead Engineer John initiated a rollback. He had worked on the initial deployment.'"
        ]
        
        if 'complexity_analysis' in issue:
            analysis = issue['complexity_analysis']
            if analysis.get('clause_length', 0) > 10:
                suggestions.append("Shorten the relative clause by removing unnecessary details")
        
        return suggestions
    
    def _generate_causal_connection_suggestions(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions for strengthening causal connections."""
        suggestions = [
            "Add explicit causal connectors: 'because', 'therefore', 'as a result'",
            "Use coordinating conjunctions to show relationships",
            "Consider: 'which compounded the complications'",
            "Make the logical connection between ideas explicit"
        ]
        
        if 'causal_analysis' in issue:
            analysis = issue['causal_analysis']
            if not analysis.get('causal_indicators'):
                suggestions.append("No causal indicators found - consider adding transitional phrases")
        
        return suggestions 