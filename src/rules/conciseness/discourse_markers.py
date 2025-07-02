"""
Discourse markers redundancy detection module for conciseness analysis.
Detects redundant logical connectors using pure SpaCy morphological analysis.
Example: 'As a result' followed by 'Consequently' in nearby sentences
"""

from typing import List, Dict, Any
from .morphological_analysis import MorphologicalAnalyzer, ContextAnalyzer

class DiscourseMarkersDetector:
    """Detector for redundant discourse markers using pure SpaCy analysis."""
    
    def __init__(self):
        self.morph_analyzer = MorphologicalAnalyzer()
        self.context_analyzer = ContextAnalyzer()
    
    def detect_discourse_marker_redundancy(self, doc) -> List[Dict[str, Any]]:
        """Detect redundant discourse markers using SpaCy morphological analysis."""
        issues = []
        
        # Extract discourse markers from all sentences
        discourse_markers = self._extract_discourse_markers(doc)
        
        # Analyze for redundancy patterns
        redundancy_patterns = self._analyze_marker_redundancy(discourse_markers, doc)
        
        for pattern in redundancy_patterns:
            if pattern['is_redundant']:
                issues.append({
                    'type': 'discourse_marker_redundancy',
                    'redundant_markers': pattern['markers'],
                    'redundancy_type': pattern['redundancy_type'],
                    'suggested_resolution': pattern['suggested_resolution'],
                    'confidence_score': pattern['confidence_score'],
                    'position': pattern['first_marker']['position']
                })
        
        return issues
    
    def _extract_discourse_markers(self, doc) -> List[Dict[str, Any]]:
        """Extract discourse markers from document using SpaCy morphological analysis."""
        markers = []
        
        for sent_idx, sentence in enumerate(doc.sents):
            sentence_markers = self._find_sentence_discourse_markers(sentence, sent_idx)
            markers.extend(sentence_markers)
        
        return markers
    
    def _find_sentence_discourse_markers(self, sentence, sent_idx: int) -> List[Dict[str, Any]]:
        """Find discourse markers in sentence using SpaCy analysis."""
        markers = []
        
        # Method 1: Detect multi-word discourse markers using dependency patterns
        multi_word_markers = self._detect_multi_word_markers(sentence, sent_idx)
        markers.extend(multi_word_markers)
        
        # Method 2: Detect single-word discourse markers using POS and morphology
        single_word_markers = self._detect_single_word_markers(sentence, sent_idx)
        markers.extend(single_word_markers)
        
        return markers
    
    def _detect_multi_word_markers(self, sentence, sent_idx: int) -> List[Dict[str, Any]]:
        """Detect multi-word discourse markers using SpaCy dependency parsing."""
        markers = []
        
        # Look for patterns like "As a result", "Due to this", etc.
        for token in sentence:
            if self._is_discourse_marker_start(token):
                marker_span = self._extract_full_marker_span(token, sentence)
                if marker_span:
                    marker_type = self._classify_marker_semantics(marker_span)
                    markers.append({
                        'text': marker_span['text'],
                        'tokens': marker_span['tokens'],
                        'semantic_type': marker_type,
                        'sentence_index': sent_idx,
                        'position': token.idx,
                        'marker_length': len(marker_span['tokens'])
                    })
        
        return markers
    
    def _detect_single_word_markers(self, sentence, sent_idx: int) -> List[Dict[str, Any]]:
        """Detect single-word discourse markers using SpaCy POS and morphology."""
        markers = []
        
        for token in sentence:
            if self._is_single_word_discourse_marker(token):
                marker_type = self._classify_single_word_marker_semantics(token)
                markers.append({
                    'text': token.text,
                    'tokens': [token],
                    'semantic_type': marker_type,
                    'sentence_index': sent_idx,
                    'position': token.idx,
                    'marker_length': 1
                })
        
        return markers
    
    def _is_discourse_marker_start(self, token) -> bool:
        """Check if token starts a discourse marker using morphological analysis."""
        lemma = token.lemma_.lower()
        
        # Common discourse marker starters using morphological patterns
        if token.pos_ == "ADP" and lemma in ['as', 'due', 'because']:
            return True
        
        if token.pos_ == "SCONJ" and lemma in ['as', 'since', 'because']:
            return True
        
        return False
    
    def _extract_full_marker_span(self, start_token, sentence) -> Dict[str, Any]:
        """Extract full discourse marker span using SpaCy dependency analysis."""
        # Look for common patterns
        tokens = [start_token]
        text_parts = [start_token.text]
        
        # Pattern: "As a result"
        if start_token.lemma_.lower() == 'as':
            next_tokens = list(sentence)[start_token.i + 1 - sentence.start:sentence.end - sentence.start]
            if len(next_tokens) >= 2:
                if (next_tokens[0].text.lower() == 'a' and 
                    next_tokens[1].lemma_.lower() == 'result'):
                    tokens.extend(next_tokens[:2])
                    text_parts.extend([t.text for t in next_tokens[:2]])
                    return {
                        'text': ' '.join(text_parts),
                        'tokens': tokens
                    }
        
        # Pattern: "Due to"
        elif start_token.lemma_.lower() == 'due':
            next_tokens = list(sentence)[start_token.i + 1 - sentence.start:sentence.end - sentence.start]
            if len(next_tokens) >= 1 and next_tokens[0].text.lower() == 'to':
                tokens.append(next_tokens[0])
                text_parts.append(next_tokens[0].text)
                return {
                    'text': ' '.join(text_parts),
                    'tokens': tokens
                }
        
        return None
    
    def _is_single_word_discourse_marker(self, token) -> bool:
        """Check if token is single-word discourse marker using SpaCy analysis."""
        lemma = token.lemma_.lower()
        
        # Single-word consequential markers
        if (token.pos_ == "ADV" and 
            lemma in ['consequently', 'therefore', 'thus', 'hence', 'accordingly']):
            return True
        
        # Check position - discourse markers often at sentence start
        if token.i == token.sent.start and lemma in ['consequently', 'therefore']:
            return True
        
        return False
    
    def _classify_marker_semantics(self, marker_span) -> str:
        """Classify semantic type of discourse marker using morphological analysis."""
        text = marker_span['text'].lower()
        
        # Consequential/result markers
        if 'result' in text or 'consequence' in text:
            return 'consequential'
        
        # Causal markers
        if 'due' in text or 'because' in text:
            return 'causal'
        
        return 'general_logical'
    
    def _classify_single_word_marker_semantics(self, token) -> str:
        """Classify semantic type of single-word marker."""
        lemma = token.lemma_.lower()
        
        # Consequential markers
        if lemma in ['consequently', 'therefore', 'thus', 'hence']:
            return 'consequential'
        
        # Temporal markers
        if lemma in ['meanwhile', 'subsequently', 'previously']:
            return 'temporal'
        
        return 'general_logical'
    
    def _analyze_marker_redundancy(self, markers: List[Dict[str, Any]], doc) -> List[Dict[str, Any]]:
        """Analyze markers for redundancy patterns."""
        redundancy_patterns = []
        
        # Check for same semantic type in consecutive sentences
        for i in range(len(markers) - 1):
            current_marker = markers[i]
            next_marker = markers[i + 1]
            
            # Check if markers are in consecutive or nearby sentences
            sentence_distance = next_marker['sentence_index'] - current_marker['sentence_index']
            
            if sentence_distance <= 2:  # Within 2 sentences
                redundancy = self._check_semantic_redundancy(current_marker, next_marker, sentence_distance)
                if redundancy['is_redundant']:
                    redundancy_patterns.append(redundancy)
        
        return redundancy_patterns
    
    def _check_semantic_redundancy(self, marker1, marker2, distance: int) -> Dict[str, Any]:
        """Check if two markers are semantically redundant."""
        # Same semantic type = redundant
        if marker1['semantic_type'] == marker2['semantic_type']:
            return {
                'is_redundant': True,
                'markers': [marker1, marker2],
                'redundancy_type': 'same_semantic_function',
                'confidence_score': 0.9 if distance == 1 else 0.7,
                'suggested_resolution': self._suggest_redundancy_resolution(marker1, marker2),
                'first_marker': marker1
            }
        
        # Special case: "As a result" + "Consequently" (both consequential)
        if (self._is_consequential_marker(marker1) and 
            self._is_consequential_marker(marker2)):
            return {
                'is_redundant': True,
                'markers': [marker1, marker2],
                'redundancy_type': 'redundant_consequential_markers',
                'confidence_score': 0.95,
                'suggested_resolution': self._suggest_consequential_redundancy_resolution(marker1, marker2),
                'first_marker': marker1
            }
        
        return {'is_redundant': False}
    
    def _is_consequential_marker(self, marker) -> bool:
        """Check if marker is consequential type."""
        return marker['semantic_type'] == 'consequential'
    
    def _suggest_redundancy_resolution(self, marker1, marker2) -> Dict[str, Any]:
        """Suggest resolution for redundant markers."""
        # Generally prefer the simpler/shorter marker
        if marker1['marker_length'] <= marker2['marker_length']:
            return {
                'action': 'keep_first_remove_second',
                'keep_marker': marker1['text'],
                'remove_marker': marker2['text'],
                'reason': 'shorter_marker_preferred'
            }
        else:
            return {
                'action': 'keep_second_remove_first',
                'keep_marker': marker2['text'],
                'remove_marker': marker1['text'],
                'reason': 'shorter_marker_preferred'
            }
    
    def _suggest_consequential_redundancy_resolution(self, marker1, marker2) -> Dict[str, Any]:
        """Suggest resolution for consequential marker redundancy."""
        # For "As a result" vs "Consequently", prefer the more direct one
        if 'result' in marker1['text'].lower():
            return {
                'action': 'keep_first_remove_second',
                'keep_marker': marker1['text'],
                'remove_marker': marker2['text'],
                'reason': 'more_specific_preferred'
            }
        else:
            return {
                'action': 'keep_second_remove_first',
                'keep_marker': marker2['text'],
                'remove_marker': marker1['text'],
                'reason': 'simpler_marker_preferred'
            }
    
    def generate_discourse_marker_suggestions(self, issue: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for discourse marker redundancy."""
        suggestions = []
        
        redundant_markers = issue.get('redundant_markers', [])
        suggested_resolution = issue.get('suggested_resolution', {})
        redundancy_type = issue.get('redundancy_type', '')
        
        if len(redundant_markers) >= 2:
            marker1_text = redundant_markers[0]['text']
            marker2_text = redundant_markers[1]['text']
            
            # Primary suggestion based on resolution
            action = suggested_resolution.get('action', '')
            if action == 'keep_first_remove_second':
                suggestions.append(f"Remove redundant '{marker2_text}' - '{marker1_text}' already indicates the logical relationship")
            elif action == 'keep_second_remove_first':
                suggestions.append(f"Remove redundant '{marker1_text}' - '{marker2_text}' is more direct")
            
            # Specific guidance for consequential redundancy
            if redundancy_type == 'redundant_consequential_markers':
                suggestions.append(f"Both '{marker1_text}' and '{marker2_text}' show cause-effect relationship")
                suggestions.append("Use only one logical connector per cause-effect sequence")
                suggestions.append("Choose the clearer, more direct marker for better flow")
        
        # General discourse marker guidance
        suggestions.append("Avoid redundant logical connectors in consecutive sentences")
        suggestions.append("Each discourse marker should add unique logical value")
        
        return suggestions 