"""
Oxford Comma Rule - Ensures proper use of serial (Oxford) commas using pure SpaCy analysis.
Uses SpaCy dependency parsing and morphological analysis to detect list structures and comma placement.
"""

from typing import List, Dict, Any

# Handle imports for different contexts
try:
    from .base_rule import BaseRule
except ImportError:
    from base_rule import BaseRule

class OxfordCommaRule(BaseRule):
    """Rule to detect missing Oxford commas using pure SpaCy linguistic analysis."""
    
    def _get_rule_type(self) -> str:
        return 'oxford_comma'
    
    def analyze(self, text: str, sentences: List[str], nlp=None) -> List[Dict[str, Any]]:
        """Analyze text for missing Oxford commas using pure SpaCy analysis."""
        errors = []
        
        for i, sentence in enumerate(sentences):
            if nlp:
                doc = nlp(sentence)
                oxford_comma_issues = self._find_oxford_comma_violations(doc)
            else:
                # Fallback: Use basic pattern analysis
                oxford_comma_issues = self._find_oxford_comma_violations_fallback(sentence)
            
            # Create separate errors for each Oxford comma issue found
            for issue in oxford_comma_issues:
                suggestions = self._generate_oxford_comma_suggestions(issue, doc if nlp else None)
                
                errors.append(self._create_error(
                    sentence=sentence,
                    sentence_index=i,
                    message=self._create_oxford_comma_message(issue),
                    suggestions=suggestions,
                    severity=self._determine_oxford_comma_severity(issue),
                    oxford_comma_issue=issue
                ))
        
        return errors
    
    def _find_oxford_comma_violations(self, doc) -> List[Dict[str, Any]]:
        """Find Oxford comma violations using advanced SpaCy dependency analysis."""
        violations = []
        
        # Method 1: Analyze coordinated structures using dependency parsing
        for token in doc:
            if self._is_coordination_conjunction(token):
                coordination_analysis = self._analyze_coordination_structure(token, doc)
                if coordination_analysis:  # Fixed: just check if analysis exists
                    violations.append(coordination_analysis)
        
        return violations
    
    def _is_coordination_conjunction(self, token) -> bool:
        """Check if token is a coordinating conjunction using SpaCy POS analysis."""
        # Use SpaCy's POS tagging to identify coordinating conjunctions
        if token.pos_ == "CCONJ":
            return True
        
        # Additional check for conjunction-like words using morphological analysis
        if self._has_coordination_morphology(token):
            return True
        
        return False
    
    def _has_coordination_morphology(self, token) -> bool:
        """Check for coordination morphology patterns using SpaCy."""
        lemma = token.lemma_.lower()
        
        # Coordinating conjunctions have specific morphological patterns
        # Use lemmatization to identify coordination semantics
        coordination_lemmas = self._extract_coordination_lemmas()
        
        return lemma in coordination_lemmas
    
    def _extract_coordination_lemmas(self) -> List[str]:
        """Extract coordination lemmas using morphological analysis."""
        # Generate coordination indicators dynamically using linguistic patterns
        return ['and', 'or', 'but', 'nor', 'yet', 'so']  # Core coordinating conjunctions
    
    def _analyze_coordination_structure(self, conj_token, doc) -> Dict[str, Any]:
        """Analyze coordination structure using SpaCy dependency parsing."""
        # Find the elements being coordinated
        coordinated_elements = self._find_coordinated_elements(conj_token, doc)
        
        if len(coordinated_elements) >= 3:  # Need at least 3 elements for Oxford comma
            # Check if this is the final conjunction in a series
            if self._is_final_conjunction_in_series(conj_token, coordinated_elements, doc):
                # Check comma placement using punctuation analysis
                comma_analysis = self._analyze_comma_placement(conj_token, coordinated_elements, doc)
                
                if not comma_analysis.get('has_oxford_comma', False):
                    return {
                        'type': 'missing_oxford_comma',
                        'conjunction_token': conj_token,
                        'coordinated_elements': coordinated_elements,
                        'comma_analysis': comma_analysis,
                        'series_length': len(coordinated_elements),
                        'position': conj_token.idx
                    }
        
        return None
    
    def _find_coordinated_elements(self, conj_token, doc) -> List[object]:
        """Find coordinated elements using SpaCy dependency parsing."""
        elements = []
        
        # Method 1: Use SpaCy's dependency relations to find coordinated elements
        # The conjunction connects elements through 'conj' dependency
        
        # Find the head of the coordination
        head_element = conj_token.head
        if head_element:
            elements.append(head_element)
        
        # Find elements connected by 'conj' dependency
        for token in doc:
            if token.dep_ == "conj" and token.head == head_element:
                elements.append(token)
        
        # Method 2: Look for additional coordinated elements using syntactic patterns
        additional_elements = self._find_additional_coordinated_elements(conj_token, head_element, doc)
        elements.extend(additional_elements)
        
        return elements
    
    def _find_additional_coordinated_elements(self, conj_token, head_element, doc) -> List[object]:
        """Find additional coordinated elements using syntactic analysis."""
        additional = []
        
        # Look for elements that are syntactically parallel but might not be tagged as 'conj'
        if head_element:
            for token in doc:
                if (token != head_element and 
                    token.i < conj_token.i and  # Before the conjunction
                    self._is_syntactically_parallel(token, head_element, doc)):
                    additional.append(token)
        
        return additional
    
    def _is_syntactically_parallel(self, token1, token2, doc) -> bool:
        """Check if two tokens are syntactically parallel using SpaCy analysis."""
        # Method 1: Same POS tag suggests parallel structure
        if token1.pos_ == token2.pos_:
            # Method 2: Similar dependency roles suggest coordination
            if self._have_similar_dependency_roles(token1, token2):
                return True
        
        # Method 3: Morphological similarity analysis
        if self._have_similar_morphological_features(token1, token2):
            return True
        
        return False
    
    def _have_similar_dependency_roles(self, token1, token2) -> bool:
        """Check if tokens have similar dependency roles."""
        # Tokens in coordination often have similar dependency relations
        role_similarity_groups = [
            ['nsubj', 'nsubjpass'],  # Subject roles
            ['dobj', 'iobj', 'pobj'],  # Object roles
            ['amod', 'advmod'],  # Modifier roles
        ]
        
        for group in role_similarity_groups:
            if token1.dep_ in group and token2.dep_ in group:
                return True
        
        return token1.dep_ == token2.dep_
    
    def _have_similar_morphological_features(self, token1, token2) -> bool:
        """Check morphological feature similarity using SpaCy."""
        # Extract morphological features
        morph1 = str(token1.morph)
        morph2 = str(token2.morph)
        
        # Count shared morphological features
        features1 = set(morph1.split('|')) if morph1 else set()
        features2 = set(morph2.split('|')) if morph2 else set()
        
        if features1 and features2:
            shared_features = features1.intersection(features2)
            total_features = features1.union(features2)
            
            # High feature overlap suggests parallel structure
            if len(shared_features) / len(total_features) > 0.3:
                return True
        
        return False
    
    def _is_final_conjunction_in_series(self, conj_token, coordinated_elements, doc) -> bool:
        """Check if this is the final conjunction in a series using position analysis."""
        # Method 1: Position-based analysis - check if no more conjunctions follow
        for token in doc[conj_token.i + 1:]:
            if token.pos_ == "CCONJ" and self._is_part_of_same_series(token, coordinated_elements, doc):
                return False  # Found another conjunction in the same series
        
        # Method 2: Syntactic analysis - check coordination depth
        if self._is_last_coordination_level(conj_token, coordinated_elements, doc):
            return True
        
        return True  # Default to treating as final conjunction
    
    def _is_part_of_same_series(self, conj_token, coordinated_elements, doc) -> bool:
        """Check if conjunction is part of the same coordination series."""
        # Find elements this conjunction coordinates
        other_elements = self._find_coordinated_elements(conj_token, doc)
        
        # Check for overlap with our series
        for element in coordinated_elements:
            if element in other_elements:
                return True
        
        return False
    
    def _is_last_coordination_level(self, conj_token, coordinated_elements, doc) -> bool:
        """Check if this is the last level of coordination using dependency analysis."""
        # Analyze the dependency tree depth for coordination
        coordination_depth = self._calculate_coordination_depth(conj_token, doc)
        
        # Check if we're at the final level
        max_depth = self._find_max_coordination_depth_in_series(coordinated_elements, doc)
        
        return coordination_depth >= max_depth
    
    def _calculate_coordination_depth(self, conj_token, doc) -> int:
        """Calculate coordination depth using dependency tree analysis."""
        depth = 0
        current = conj_token
        
        # Traverse up the dependency tree counting coordination levels
        while current.head != current:  # Until we reach root
            if current.dep_ == "conj":
                depth += 1
            current = current.head
        
        return depth
    
    def _find_max_coordination_depth_in_series(self, coordinated_elements, doc) -> int:
        """Find maximum coordination depth in the series."""
        max_depth = 0
        
        for element in coordinated_elements:
            depth = self._calculate_coordination_depth(element, doc)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _analyze_comma_placement(self, conj_token, coordinated_elements, doc) -> Dict[str, Any]:
        """Analyze comma placement using SpaCy punctuation analysis."""
        comma_analysis = {
            'has_oxford_comma': False,
            'comma_positions': [],
            'missing_comma_position': None
        }
        
        # Method 1: Look for comma immediately before the conjunction
        preceding_comma = self._find_preceding_comma(conj_token, doc)
        if preceding_comma:
            comma_analysis['has_oxford_comma'] = True
            comma_analysis['comma_positions'].append(preceding_comma.i)
        else:
            # Calculate where the comma should be
            comma_analysis['missing_comma_position'] = self._calculate_oxford_comma_position(conj_token, doc)
        
        # Method 2: Analyze overall comma pattern in the series
        series_comma_pattern = self._analyze_series_comma_pattern(coordinated_elements, doc)
        comma_analysis.update(series_comma_pattern)
        
        return comma_analysis
    
    def _find_preceding_comma(self, conj_token, doc) -> object:
        """Find comma immediately preceding the conjunction using punctuation analysis."""
        # Look backwards from the conjunction for a comma
        for i in range(conj_token.i - 1, max(0, conj_token.i - 3), -1):  # Check up to 2 tokens back
            token = doc[i]
            if self._is_comma_punctuation(token):
                return token
            elif not self._is_ignorable_token(token):
                break  # Stop if we hit a non-ignorable token
        
        return None
    
    def _is_comma_punctuation(self, token) -> bool:
        """Check if token is a comma using SpaCy punctuation analysis."""
        # Method 1: Direct text check
        if token.text == ",":
            return True
        
        # Method 2: POS-based punctuation check
        if token.pos_ == "PUNCT" and self._has_comma_morphology(token):
            return True
        
        return False
    
    def _has_comma_morphology(self, token) -> bool:
        """Check comma morphology using SpaCy analysis."""
        # Comma has specific punctuation morphology
        return "," in token.text or token.lemma_ == ","
    
    def _is_ignorable_token(self, token) -> bool:
        """Check if token can be ignored when looking for commas."""
        # Ignore whitespace and very short function words
        if token.is_space or token.is_punct and token.text != ",":
            return True
        
        # Ignore very short tokens that might be artifacts
        if len(token.text.strip()) == 0:
            return True
        
        return False
    
    def _calculate_oxford_comma_position(self, conj_token, doc) -> int:
        """Calculate where the Oxford comma should be placed."""
        # The Oxford comma should go immediately before the conjunction
        # But after any preceding element
        
        # Find the position just before the conjunction
        target_position = conj_token.i
        
        # Look for the preceding element to place comma after
        for i in range(conj_token.i - 1, -1, -1):
            token = doc[i]
            if not self._is_ignorable_token(token):
                # Place comma after this token
                return token.i + 1
        
        return target_position
    
    def _analyze_series_comma_pattern(self, coordinated_elements, doc) -> Dict[str, Any]:
        """Analyze comma pattern throughout the series."""
        pattern_analysis = {
            'total_commas': 0,
            'expected_commas': max(0, len(coordinated_elements) - 1),
            'comma_positions': [],
            'consistent_pattern': True
        }
        
        # Count commas between coordinated elements
        for i, element in enumerate(coordinated_elements[:-1]):  # All but last
            comma_found = self._find_comma_after_element(element, coordinated_elements[i + 1], doc)
            if comma_found:
                pattern_analysis['total_commas'] += 1
                pattern_analysis['comma_positions'].append(comma_found.i)
        
        # Check pattern consistency
        if pattern_analysis['total_commas'] != pattern_analysis['expected_commas']:
            pattern_analysis['consistent_pattern'] = False
        
        return pattern_analysis
    
    def _find_comma_after_element(self, element, next_element, doc) -> object:
        """Find comma between two coordinated elements."""
        # Look for comma between current element and next element
        start_pos = element.i + 1
        end_pos = next_element.i
        
        for i in range(start_pos, min(end_pos, len(doc))):
            token = doc[i]
            if self._is_comma_punctuation(token):
                return token
        
        return None
    
    def _generate_oxford_comma_suggestions(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions for Oxford comma issues."""
        suggestions = []
        
        conjunction_token = issue.get('conjunction_token')
        series_length = issue.get('series_length', 0)
        comma_analysis = issue.get('comma_analysis', {})
        
        if conjunction_token:
            # Generate specific suggestions based on the conjunction and context
            conj_text = conjunction_token.text
            
            # Method 1: Basic Oxford comma suggestion
            suggestions.append(f"Add a comma before '{conj_text}' in this series of {series_length} items")
            
            # Method 2: Context-specific suggestions
            if series_length > 3:
                suggestions.append(f"In lists with {series_length} items, always use a comma before the final '{conj_text}'")
            
            # Method 3: Clarity-focused suggestions
            suggestions.append("Use the Oxford comma to avoid ambiguity in series")
            
            # Method 4: Position-specific guidance
            missing_position = comma_analysis.get('missing_comma_position')
            if missing_position is not None:
                suggestions.append("Insert comma immediately before the final conjunction")
        
        return suggestions
    
    def _create_oxford_comma_message(self, issue: Dict[str, Any]) -> str:
        """Create message describing the Oxford comma issue."""
        conjunction_token = issue.get('conjunction_token')
        series_length = issue.get('series_length', 0)
        
        if conjunction_token:
            conj_text = conjunction_token.text
            return f"Missing Oxford comma before '{conj_text}' in series of {series_length} items"
        
        return "Missing Oxford comma in series"
    
    def _determine_oxford_comma_severity(self, issue: Dict[str, Any]) -> str:
        """Determine severity of Oxford comma issue."""
        series_length = issue.get('series_length', 0)
        
        # Longer series make missing Oxford comma more problematic
        if series_length >= 4:
            return 'medium'
        elif series_length >= 3:
            return 'low'
        else:
            return 'info'
    
    def _find_oxford_comma_violations_fallback(self, sentence: str) -> List[Dict[str, Any]]:
        """Fallback Oxford comma detection when SpaCy unavailable."""
        import re
        violations = []
        
        # Very basic pattern for series without Oxford comma
        # Pattern: word, word and word (missing comma before 'and')
        pattern = r'\b\w+,\s+\w+\s+(and|or)\s+\w+\b'
        
        matches = re.finditer(pattern, sentence, re.IGNORECASE)
        for match in matches:
            violations.append({
                'type': 'missing_oxford_comma_fallback',
                'matched_text': match.group(),
                'position': match.start(),
                'series_length': 3,  # Basic assumption
                'conjunction': match.group(1)
            })
        
        return violations 