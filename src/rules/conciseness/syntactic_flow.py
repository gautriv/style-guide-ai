"""
Syntactic flow optimization detection module for conciseness analysis.
Detects wordy constructions that can be made more direct using pure SpaCy morphological analysis.
Example: 'As a result, the team should have caught this error earlier' → 'The team should have caught this error earlier'
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class SyntacticFlowDetector:
    """Detector for syntactic flow optimization using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_syntactic_flow_issues(self, doc) -> List[Dict[str, Any]]:
        """Detect syntactic flow optimization opportunities using SpaCy morphological analysis."""
        issues = []
        
        for sentence in doc.sents:
            # Method 1: Detect wordy discourse marker constructions
            discourse_wordiness = self._detect_wordy_discourse_constructions(sentence, doc)
            issues.extend(discourse_wordiness)
            
            # Method 2: Detect indirect subject-verb constructions
            indirect_constructions = self._detect_indirect_constructions(sentence, doc)
            issues.extend(indirect_constructions)
            
            # Method 3: Detect overly complex syntactic patterns
            complex_patterns = self._detect_complex_syntactic_patterns(sentence, doc)
            issues.extend(complex_patterns)
        
        return issues
    
    def _detect_wordy_discourse_constructions(self, sentence, doc) -> List[Dict[str, Any]]:
        """Detect wordy discourse marker constructions using SpaCy dependency analysis."""
        issues = []
        
        # Look for discourse markers at sentence start followed by complex constructions
        for token in sentence:
            if self._is_sentence_initial_discourse_marker(token, sentence):
                wordiness_analysis = self._analyze_discourse_marker_wordiness(token, sentence, doc)
                if wordiness_analysis['is_wordy']:
                    issues.append({
                        'type': 'wordy_discourse_construction',
                        'discourse_marker': wordiness_analysis['marker_text'],
                        'wordy_construction': wordiness_analysis['construction_text'],
                        'suggested_simplification': wordiness_analysis['simplified_version'],
                        'confidence_score': wordiness_analysis['confidence_score'],
                        'position': token.idx
                    })
        
        return issues
    
    def _detect_indirect_constructions(self, sentence, doc) -> List[Dict[str, Any]]:
        """Detect indirect subject-verb constructions using SpaCy analysis."""
        issues = []
        
        # Look for constructions that unnecessarily distance subject from verb
        root_verb = self._find_sentence_root_verb(sentence)
        if root_verb:
            indirectness_analysis = self._analyze_construction_directness(root_verb, sentence, doc)
            if indirectness_analysis['is_indirect']:
                issues.append({
                    'type': 'indirect_construction',
                    'indirect_pattern': indirectness_analysis['pattern_type'],
                    'suggested_directness': indirectness_analysis['direct_version'],
                    'complexity_reduction': indirectness_analysis['complexity_reduction'],
                    'position': root_verb.idx
                })
        
        return issues
    
    def _detect_complex_syntactic_patterns(self, sentence, doc) -> List[Dict[str, Any]]:
        """Detect overly complex syntactic patterns using SpaCy dependency analysis."""
        issues = []
        
        # Calculate syntactic complexity
        complexity_analysis = self._analyze_syntactic_complexity(sentence)
        
        if complexity_analysis['is_overly_complex']:
            issues.append({
                'type': 'complex_syntactic_pattern',
                'complexity_score': complexity_analysis['complexity_score'],
                'complexity_factors': complexity_analysis['factors'],
                'suggested_simplifications': complexity_analysis['simplifications'],
                'position': sentence.start_char
            })
        
        return issues
    
    def _is_sentence_initial_discourse_marker(self, token, sentence) -> bool:
        """Check if token is sentence-initial discourse marker using SpaCy analysis."""
        # Must be at sentence start or very near start
        if token.i - sentence.start > 2:
            return False
        
        lemma = token.lemma_.lower()
        
        # Multi-word markers starting with prepositions
        if token.pos_ == "ADP" and lemma in ['as', 'due']:
            return True
        
        # Single-word adverbial markers
        if (token.pos_ == "ADV" and 
            lemma in ['consequently', 'therefore', 'furthermore', 'moreover']):
            return True
        
        return False
    
    def _analyze_discourse_marker_wordiness(self, marker_token, sentence, doc) -> Dict[str, Any]:
        """Analyze discourse marker for wordiness using SpaCy dependency analysis."""
        analysis = {
            'is_wordy': False,
            'marker_text': '',
            'construction_text': '',
            'simplified_version': '',
            'confidence_score': 0.0
        }
        
        # Extract full marker text
        marker_span = self._extract_discourse_marker_span(marker_token, sentence)
        analysis['marker_text'] = marker_span
        
        # Analyze the main clause after the discourse marker
        main_clause_analysis = self._analyze_main_clause_after_marker(marker_token, sentence)
        
        if main_clause_analysis['can_be_direct']:
            analysis['is_wordy'] = True
            analysis['construction_text'] = sentence.text
            analysis['simplified_version'] = main_clause_analysis['direct_version']
            analysis['confidence_score'] = main_clause_analysis['confidence']
        
        return analysis
    
    def _extract_discourse_marker_span(self, start_token, sentence) -> str:
        """Extract full discourse marker span using SpaCy token analysis."""
        tokens = [start_token]
        
        # Handle multi-word markers
        if start_token.lemma_.lower() == 'as':
            # Look for "As a result"
            next_tokens = list(sentence)[start_token.i + 1 - sentence.start:sentence.end - sentence.start]
            if (len(next_tokens) >= 2 and 
                next_tokens[0].text.lower() == 'a' and 
                next_tokens[1].lemma_.lower() == 'result'):
                return 'As a result'
        
        return start_token.text
    
    def _analyze_main_clause_after_marker(self, marker_token, sentence) -> Dict[str, Any]:
        """Analyze main clause after discourse marker using SpaCy dependency analysis."""
        analysis = {
            'can_be_direct': False,
            'direct_version': '',
            'confidence': 0.0
        }
        
        # Find the main verb of the clause
        main_verb = self._find_main_verb_after_marker(marker_token, sentence)
        
        if main_verb:
            # Check if the clause is self-contained and can stand alone
            clause_independence = self._assess_clause_independence(main_verb, sentence)
            
            if clause_independence['is_independent']:
                # Generate direct version by removing discourse marker
                direct_text = self._generate_direct_version(marker_token, sentence)
                
                analysis['can_be_direct'] = True
                analysis['direct_version'] = direct_text
                analysis['confidence'] = clause_independence['independence_score']
        
        return analysis
    
    def _find_main_verb_after_marker(self, marker_token, sentence):
        """Find main verb after discourse marker using SpaCy dependency analysis."""
        # Look for the root verb or main verb in the sentence
        for token in sentence:
            if (token.pos_ == "VERB" and 
                token.dep_ in ["ROOT", "ccomp", "xcomp"] and
                token.i > marker_token.i):
                return token
        
        return None
    
    def _assess_clause_independence(self, verb_token, sentence) -> Dict[str, Any]:
        """Assess if clause can stand independently using SpaCy dependency analysis."""
        independence = {
            'is_independent': False,
            'independence_score': 0.0
        }
        
        # Check for complete subject-verb-object structure
        has_subject = any(child.dep_ in ["nsubj", "nsubjpass"] for child in verb_token.children)
        has_object_or_complement = any(child.dep_ in ["dobj", "attr", "acomp"] for child in verb_token.children)
        
        if has_subject:
            independence['independence_score'] += 0.5
            
            if has_object_or_complement or verb_token.lemma_ in ['be', 'seem', 'appear']:
                independence['independence_score'] += 0.3
                independence['is_independent'] = True
        
        return independence
    
    def _generate_direct_version(self, marker_token, sentence) -> str:
        """Generate direct version by removing discourse marker using SpaCy token analysis."""
        # Find where the main content starts after the discourse marker
        content_start = self._find_content_start_after_marker(marker_token, sentence)
        
        if content_start:
            # Extract everything from content start to sentence end
            tokens_after_marker = list(sentence)[content_start.i - sentence.start:]
            direct_text = ' '.join(token.text for token in tokens_after_marker)
            
            # Capitalize first word
            if direct_text:
                direct_text = direct_text[0].upper() + direct_text[1:] if len(direct_text) > 1 else direct_text.upper()
            
            return direct_text
        
        return sentence.text
    
    def _find_content_start_after_marker(self, marker_token, sentence):
        """Find where main content starts after discourse marker."""
        # Skip past the discourse marker and any commas
        for token in sentence:
            if (token.i > marker_token.i and 
                token.pos_ not in ["PUNCT"] and
                token.lemma_.lower() not in ['a', 'the', 'result']):
                return token
        
        return None
    
    def _find_sentence_root_verb(self, sentence):
        """Find root verb of sentence using SpaCy dependency analysis."""
        for token in sentence:
            if token.dep_ == "ROOT" and token.pos_ in ["VERB", "AUX"]:
                return token
        return None
    
    def _analyze_construction_directness(self, root_verb, sentence, doc) -> Dict[str, Any]:
        """Analyze construction directness using SpaCy dependency analysis."""
        analysis = {
            'is_indirect': False,
            'pattern_type': 'unknown',
            'direct_version': '',
            'complexity_reduction': 0.0
        }
        
        # Check for modal verb + passive construction indirectness
        if self._is_modal_passive_construction(root_verb, sentence):
            analysis['is_indirect'] = True
            analysis['pattern_type'] = 'modal_passive_indirectness'
            analysis['direct_version'] = self._suggest_modal_passive_directness(root_verb, sentence)
            analysis['complexity_reduction'] = 0.3
        
        return analysis
    
    def _is_modal_passive_construction(self, verb, sentence) -> bool:
        """Check for modal + passive construction using SpaCy analysis."""
        # Check if this is a modal auxiliary
        if verb.tag_ == "MD":  # Modal auxiliary
            # Look for passive construction in dependents
            for child in verb.children:
                if (child.pos_ == "VERB" and 
                    any(grandchild.dep_ == "nsubjpass" for grandchild in child.children)):
                    return True
        
        return False
    
    def _suggest_modal_passive_directness(self, modal_verb, sentence) -> str:
        """Suggest more direct version of modal passive construction."""
        # Find the passive subject
        passive_subject = None
        main_verb = None
        
        for child in modal_verb.children:
            if child.pos_ == "VERB":
                main_verb = child
                for grandchild in child.children:
                    if grandchild.dep_ == "nsubjpass":
                        passive_subject = grandchild
                        break
                break
        
        if passive_subject and main_verb:
            # Suggest active construction
            return f"{passive_subject.text} {modal_verb.text} {main_verb.lemma_}"
        
        return sentence.text
    
    def _analyze_syntactic_complexity(self, sentence) -> Dict[str, Any]:
        """Analyze syntactic complexity using SpaCy dependency analysis."""
        complexity = {
            'is_overly_complex': False,
            'complexity_score': 0.0,
            'factors': [],
            'simplifications': []
        }
        
        # Factor 1: Dependency depth
        max_depth = self._calculate_max_dependency_depth(sentence)
        if max_depth > 4:
            complexity['complexity_score'] += (max_depth - 4) * 0.2
            complexity['factors'].append(f'deep_dependencies_{max_depth}')
        
        # Factor 2: Number of subordinate clauses
        subordinate_clauses = self._count_subordinate_clauses(sentence)
        if subordinate_clauses > 2:
            complexity['complexity_score'] += (subordinate_clauses - 2) * 0.3
            complexity['factors'].append(f'multiple_subordinate_clauses_{subordinate_clauses}')
        
        # Factor 3: Long noun phrases
        long_noun_phrases = self._count_long_noun_phrases(sentence)
        if long_noun_phrases > 1:
            complexity['complexity_score'] += long_noun_phrases * 0.2
            complexity['factors'].append(f'long_noun_phrases_{long_noun_phrases}')
        
        # Determine if overly complex
        if complexity['complexity_score'] > 0.6:
            complexity['is_overly_complex'] = True
            complexity['simplifications'] = self._suggest_complexity_simplifications(complexity['factors'])
        
        return complexity
    
    def _calculate_max_dependency_depth(self, sentence) -> int:
        """Calculate maximum dependency depth using SpaCy."""
        max_depth = 0
        
        for token in sentence:
            depth = 0
            current = token
            while current.head != current and depth < 10:  # Prevent infinite loops
                depth += 1
                current = current.head
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_subordinate_clauses(self, sentence) -> int:
        """Count subordinate clauses using SpaCy dependency analysis."""
        subordinate_count = 0
        
        for token in sentence:
            if token.dep_ in ["advcl", "ccomp", "xcomp", "acl"]:
                subordinate_count += 1
        
        return subordinate_count
    
    def _count_long_noun_phrases(self, sentence) -> int:
        """Count long noun phrases using SpaCy analysis."""
        long_np_count = 0
        
        for token in sentence:
            if token.pos_ == "NOUN" and token.dep_ in ["nsubj", "dobj", "nsubjpass"]:
                # Count modifiers
                modifier_count = sum(1 for child in token.children if child.dep_ in ["amod", "compound", "nmod"])
                if modifier_count > 2:
                    long_np_count += 1
        
        return long_np_count
    
    def _suggest_complexity_simplifications(self, factors) -> List[str]:
        """Suggest simplifications based on complexity factors."""
        simplifications = []
        
        for factor in factors:
            if 'deep_dependencies' in factor:
                simplifications.append("Break into shorter sentences to reduce dependency depth")
            elif 'subordinate_clauses' in factor:
                simplifications.append("Consider separating subordinate clauses into independent sentences")
            elif 'long_noun_phrases' in factor:
                simplifications.append("Simplify complex noun phrases")
        
        return simplifications
    
    def generate_syntactic_flow_suggestions(self, issue: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for syntactic flow optimization."""
        suggestions = []
        
        issue_type = issue.get('type', '')
        
        if issue_type == 'wordy_discourse_construction':
            discourse_marker = issue.get('discourse_marker', '')
            simplified_version = issue.get('suggested_simplification', '')
            
            suggestions.append(f"Remove wordy discourse marker: '{discourse_marker}' adds unnecessary complexity")
            if simplified_version:
                suggestions.append(f"Simplify to: '{simplified_version}'")
            suggestions.append("The sentence is clear without the discourse marker")
            suggestions.append("Direct statements are more impactful than marker-heavy constructions")
        
        elif issue_type == 'indirect_construction':
            suggested_directness = issue.get('suggested_directness', '')
            if suggested_directness:
                suggestions.append(f"Use more direct construction: '{suggested_directness}'")
            suggestions.append("Reduce distance between subject and main action")
            suggestions.append("Active voice creates clearer responsibility")
        
        elif issue_type == 'complex_syntactic_pattern':
            complexity_factors = issue.get('complexity_factors', [])
            suggested_simplifications = issue.get('suggested_simplifications', [])
            
            suggestions.append("Simplify complex sentence structure")
            suggestions.extend(suggested_simplifications)
            
            if complexity_factors:
                factors_text = ', '.join(complexity_factors)
                suggestions.append(f"Complexity factors: {factors_text}")
        
        return suggestions 