"""
Style Analyzer Module - Analyzes text content against style guide rules using NLP libraries.
Enhanced for technical writing with comprehensive readability and grade-level analysis.
"""

import re
import logging
import math
from typing import List, Dict, Any
import spacy
import nltk
import textstat
from collections import defaultdict, Counter

# Use built-in syllable estimation - no external library needed
SYLLABLES_AVAILABLE = False

logger = logging.getLogger(__name__)

class StyleAnalyzer:
    """Analyzes text against comprehensive style guide rules for technical writing."""
    
    def __init__(self):
        """Initialize the style analyzer with NLP tools."""
        try:
            # Load SpaCy model (install with: python -m spacy download en_core_web_sm)
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' not found. Please install it.")
            self.nlp = None
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('cmudict', quiet=True)  # For syllable counting
        except:
            pass
        
        # Technical writing style guide rules
        self.rules = {
            'max_sentence_length': 25,
            'target_grade_level': (9, 11),  # 9th to 11th grade target
            'min_readability_score': 60.0,
            'max_fog_index': 12.0,  # Gunning Fog Index for technical writing
            'passive_voice_threshold': 0.15,  # Max 15% passive voice
            'word_repetition_threshold': 3,
            'max_syllables_per_word': 2.5,  # Average syllables per word
            'min_sentence_variety': 0.7,  # Sentence length variety
        }
        
        # Note: All readability metrics are handled by textstat library
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive style analysis on text for technical writing."""
        if not text or not text.strip():
            return {
                'errors': [], 
                'suggestions': [], 
                'statistics': {}, 
                'technical_writing_metrics': {},
                'overall_score': 0
            }
        
        try:
            sentences = self._split_sentences(text)
            paragraphs = self._split_paragraphs(text)
            
            errors = []
            
            # Comprehensive readability analysis
            readability_issues = self._check_comprehensive_readability(text)
            errors.extend(readability_issues)
            
            # Grade level analysis
            grade_level_issues = self._check_grade_level(text)
            errors.extend(grade_level_issues)
            
            # Sentence length and variety
            length_errors = self._check_sentence_length_and_variety(sentences)
            errors.extend(length_errors)
            
            # Passive voice detection
            passive_voice_errors = self._check_passive_voice(sentences)
            errors.extend(passive_voice_errors)
            
            # Conciseness check
            conciseness_errors = self._check_conciseness(sentences)
            errors.extend(conciseness_errors)
            
            # Word choice and clarity
            clarity_errors = self._check_clarity(sentences)
            errors.extend(clarity_errors)
            
            # Technical writing specific checks
            technical_errors = self._check_technical_writing_patterns(sentences)
            errors.extend(technical_errors)
            
            # Calculate comprehensive statistics
            statistics = self._calculate_comprehensive_statistics(text, sentences, paragraphs)
            
            # Calculate technical writing specific metrics
            technical_metrics = self._calculate_technical_writing_metrics(text, sentences, errors)
            
            # Generate suggestions
            suggestions = self._generate_suggestions(errors)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(errors, statistics, technical_metrics)
            
            return {
                'errors': errors,
                'suggestions': suggestions,
                'statistics': statistics,
                'technical_writing_metrics': technical_metrics,
                'overall_score': overall_score
            }
            
        except Exception as e:
            logger.error(f"Error in style analysis: {str(e)}")
            return {
                'errors': [{'type': 'system_error', 'message': f'Analysis failed: {str(e)}'}],
                'suggestions': [],
                'statistics': {},
                'technical_writing_metrics': {},
                'overall_score': 0
            }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not self.nlp:
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _check_comprehensive_readability(self, text: str) -> List[Dict[str, Any]]:
        """Check readability scores."""
        errors = []
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            if flesch_score < self.rules['min_readability_score']:
                errors.append({
                    'type': 'readability',
                    'message': f'Text is difficult to read (Flesch score: {flesch_score:.1f}). Aim for 60+ for good readability.',
                    'suggestions': ['Use shorter sentences', 'Use simpler words', 'Break up complex ideas'],
                    'severity': 'medium',
                    'score': flesch_score
                })
        except Exception as e:
            logger.error(f"Readability check failed: {e}")
        return errors
    
    def _check_grade_level(self, text: str) -> List[Dict[str, Any]]:
        """Check grade level of the text."""
        errors = []
        try:
            grade_level = textstat.text_standard(text, float_output=True)
            if grade_level < self.rules['target_grade_level'][0] or grade_level > self.rules['target_grade_level'][1]:
                errors.append({
                    'type': 'grade_level',
                    'message': f'Text is at grade level {grade_level:.1f}. Aim for {self.rules["target_grade_level"][0]}-{self.rules["target_grade_level"][1]} for optimal readability.',
                    'suggestions': ['Use simpler words', 'Break up complex ideas'],
                    'severity': 'medium',
                    'score': grade_level
                })
        except Exception as e:
            logger.error(f"Grade level check failed: {e}")
        return errors
    
    def _check_sentence_length_and_variety(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Check for overly long sentences and sentence length variety."""
        errors = []
        for i, sentence in enumerate(sentences):
            word_count = len(sentence.split())
            if word_count > self.rules['max_sentence_length']:
                errors.append({
                    'type': 'sentence_length',
                    'message': f'Sentence is too long ({word_count} words). Consider breaking it up.',
                    'suggestions': ['Split into multiple sentences', 'Remove unnecessary words'],
                    'sentence': sentence,
                    'sentence_index': i,
                    'word_count': word_count,
                    'severity': 'medium' if word_count < 35 else 'high'
                })
        return errors
    
    def _check_passive_voice(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Detect passive voice usage."""
        errors = []
        passive_patterns = [
            r'\b(is|are|was|were|being|been)\s+\w*ed\b',
            r'\b(is|are|was|were|being|been)\s+\w*en\b'
        ]
        
        for i, sentence in enumerate(sentences):
            has_passive = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in passive_patterns)
            if has_passive:
                errors.append({
                    'type': 'passive_voice',
                    'message': 'Consider using active voice for clearer, more direct writing.',
                    'suggestions': ['Rewrite in active voice', 'Make the subject perform the action'],
                    'sentence': sentence,
                    'sentence_index': i,
                    'severity': 'low'
                })
        return errors
    
    def _check_conciseness(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Check for wordy phrases and redundancy."""
        errors = []
        wordy_phrases = {
            r'\bin order to\b': 'to',
            r'\bdue to the fact that\b': 'because',
            r'\bat this point in time\b': 'now',
            r'\ba large number of\b': 'many',
            r'\bmake a decision\b': 'decide'
        }
        
        for i, sentence in enumerate(sentences):
            found_wordy = []
            for pattern, replacement in wordy_phrases.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    found_wordy.append({'phrase': re.search(pattern, sentence, re.IGNORECASE).group(), 'replacement': replacement})
            
            if found_wordy:
                errors.append({
                    'type': 'conciseness',
                    'message': 'Consider using more concise language.',
                    'suggestions': [f"Replace '{item['phrase']}' with '{item['replacement']}'" for item in found_wordy],
                    'sentence': sentence,
                    'sentence_index': i,
                    'severity': 'low'
                })
        return errors
    
    def _check_clarity(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Check for clarity issues like jargon, complex words."""
        errors = []
        complex_words = {
            'utilize': 'use', 'facilitate': 'help', 'demonstrate': 'show',
            'implement': 'do', 'commence': 'start', 'terminate': 'end'
        }
        
        for i, sentence in enumerate(sentences):
            found_complex = []
            for complex_word, simple_word in complex_words.items():
                if re.search(r'\b' + complex_word + r'\b', sentence, re.IGNORECASE):
                    found_complex.append({'complex': complex_word, 'simple': simple_word})
            
            if found_complex:
                errors.append({
                    'type': 'clarity',
                    'message': 'Consider using simpler, clearer language.',
                    'suggestions': [f"Replace '{item['complex']}' with '{item['simple']}'" for item in found_complex],
                    'sentence': sentence,
                    'sentence_index': i,
                    'severity': 'low'
                })
        return errors
    
    def _check_technical_writing_patterns(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Check for technical writing specific patterns."""
        errors = []
        # Implement technical writing specific pattern checks
        return errors
    
    def _generate_suggestions(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on detected errors."""
        suggestions = []
        error_counts = defaultdict(int)
        for error in errors:
            error_counts[error['type']] += 1
        
        if error_counts['sentence_length'] > 2:
            suggestions.append({
                'type': 'general',
                'title': 'Sentence Length',
                'message': 'Multiple long sentences detected. Consider breaking them into shorter, clearer sentences.',
                'priority': 'high'
            })
        
        if error_counts['passive_voice'] > 3:
            suggestions.append({
                'type': 'general',
                'title': 'Active Voice',
                'message': 'Use active voice more frequently for clearer, more engaging writing.',
                'priority': 'medium'
            })
        
        return suggestions
    
    def _calculate_comprehensive_statistics(self, text: str, sentences: List[str], paragraphs: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive text statistics for technical writing."""
        words = text.split()
        stats = {
            # Basic counts
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'character_count': len(text),
            'character_count_no_spaces': len(text.replace(' ', '')),
            
            # Sentence analysis
            'avg_sentence_length': 0,
            'median_sentence_length': 0,
            'sentence_length_variety': 0,
            'longest_sentence': 0,
            'shortest_sentence': 0,
            
            # Word analysis
            'avg_word_length': 0,
            'avg_syllables_per_word': 0,
            'complex_words_count': 0,
            'complex_words_percentage': 0,
            
            # Readability scores
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'gunning_fog_index': 0,
            'smog_index': 0,
            'coleman_liau_index': 0,
            'automated_readability_index': 0,
            'dale_chall_readability': 0,
            'linsear_write_formula': 0,
            
            # Technical writing metrics
            'passive_voice_percentage': 0,
            'grade_level_assessment': '',
            'readability_grade_target_met': False,
            'technical_complexity_score': 0,
            
            # Language patterns
            'most_common_words': [],
            'word_frequency_distribution': {},
            'sentence_types': {'simple': 0, 'compound': 0, 'complex': 0}
        }
        
        try:
            if sentences:
                sentence_lengths = [len(s.split()) for s in sentences]
                stats['avg_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths)
                stats['median_sentence_length'] = sorted(sentence_lengths)[len(sentence_lengths) // 2]
                stats['longest_sentence'] = max(sentence_lengths)
                stats['shortest_sentence'] = min(sentence_lengths)
                
                # Calculate sentence length variety (coefficient of variation)
                if stats['avg_sentence_length'] > 0:
                    variance = sum((x - stats['avg_sentence_length']) ** 2 for x in sentence_lengths) / len(sentence_lengths)
                    std_dev = math.sqrt(variance)
                    stats['sentence_length_variety'] = std_dev / stats['avg_sentence_length']
            
            # Word analysis
            if words:
                word_lengths = [len(word.strip('.,!?;:"()[]{}')) for word in words]
                stats['avg_word_length'] = sum(word_lengths) / len(word_lengths)
                
                # Count complex words (3+ syllables)
                complex_words = 0
                total_syllables = 0
                
                for word in words:
                    clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
                    if clean_word:
                        syllable_count = self._estimate_syllables(clean_word)
                        total_syllables += syllable_count
                        if syllable_count >= 3:
                            complex_words += 1
                
                stats['complex_words_count'] = complex_words
                stats['complex_words_percentage'] = (complex_words / len(words)) * 100 if words else 0
                stats['avg_syllables_per_word'] = total_syllables / len(words) if words else 0
            
            # Comprehensive readability analysis
            stats['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            stats['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            stats['gunning_fog_index'] = textstat.gunning_fog(text)
            stats['smog_index'] = textstat.smog_index(text)
            stats['coleman_liau_index'] = textstat.coleman_liau_index(text)
            stats['automated_readability_index'] = textstat.automated_readability_index(text)
            stats['dale_chall_readability'] = textstat.dale_chall_readability_score(text)
            stats['linsear_write_formula'] = textstat.linsear_write_formula(text)
            
            # Grade level assessment
            grade_level = textstat.text_standard(text, float_output=True)
            stats['grade_level_assessment'] = f"{grade_level:.1f}"
            stats['readability_grade_target_met'] = (
                self.rules['target_grade_level'][0] <= grade_level <= self.rules['target_grade_level'][1]
            )
            
            # Calculate passive voice percentage
            passive_count = sum(1 for error in self._check_passive_voice(sentences) if error['type'] == 'passive_voice')
            stats['passive_voice_percentage'] = (passive_count / len(sentences) * 100) if sentences else 0
            
            # Technical complexity score (custom formula)
            complexity_factors = [
                stats['avg_sentence_length'] / 20,  # Normalize to ~20 words
                stats['avg_syllables_per_word'] / 2,  # Normalize to ~2 syllables
                stats['complex_words_percentage'] / 10,  # Normalize to ~10%
                (100 - stats['flesch_reading_ease']) / 100  # Invert Flesch score
            ]
            stats['technical_complexity_score'] = sum(complexity_factors) / len(complexity_factors) * 100
            
            # Word frequency analysis
            word_counter = Counter(word.lower().strip('.,!?;:"()[]{}') for word in words)
            stats['most_common_words'] = word_counter.most_common(10)
            stats['word_frequency_distribution'] = dict(word_counter.most_common(20))
            
            # Sentence type analysis (simplified)
            for sentence in sentences:
                if ';' in sentence or ' and ' in sentence or ' or ' in sentence:
                    if ',' in sentence and len(sentence.split(',')) > 2:
                        stats['sentence_types']['complex'] += 1
                    else:
                        stats['sentence_types']['compound'] += 1
                else:
                    stats['sentence_types']['simple'] += 1
                    
        except Exception as e:
            logger.error(f"Error calculating comprehensive statistics: {e}")
        
        return stats
    
    def _estimate_syllables(self, word: str) -> int:
        """Fallback syllable estimation when syllables library is not available."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count
    
    def _calculate_technical_writing_metrics(self, text: str, sentences: List[str], errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate technical writing specific metrics."""
        metrics = {
            # Core readability metrics
            'flesch_reading_ease': 0,
            'flesch_kincaid_grade': 0,
            'gunning_fog_index': 0,
            'smog_index': 0,
            'coleman_liau_index': 0,
            'automated_readability_index': 0,
            'dale_chall_readability': 0,
            
            # Grade level analysis
            'estimated_grade_level': 0,
            'grade_level_category': '',
            'meets_target_grade': False,
            
            # Technical writing assessments
            'sentence_complexity_score': 0,
            'vocabulary_complexity_score': 0,
            'overall_readability_rating': '',
            'passive_voice_ratio': 0,
            'avg_syllables_per_word': 0,
            
            # Recommendations
            'improvement_priority': [],
            'readability_recommendations': []
        }
        
        try:
            # Calculate all readability metrics
            metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            metrics['gunning_fog_index'] = textstat.gunning_fog(text)
            metrics['smog_index'] = textstat.smog_index(text)
            metrics['coleman_liau_index'] = textstat.coleman_liau_index(text)
            metrics['automated_readability_index'] = textstat.automated_readability_index(text)
            metrics['dale_chall_readability'] = textstat.dale_chall_readability_score(text)
            
            # Grade level assessment
            grade_level = textstat.text_standard(text, float_output=True)
            metrics['estimated_grade_level'] = grade_level
            metrics['meets_target_grade'] = (
                self.rules['target_grade_level'][0] <= grade_level <= self.rules['target_grade_level'][1]
            )
            
            # Categorize grade level
            if grade_level <= 8:
                metrics['grade_level_category'] = 'Elementary/Middle School'
            elif grade_level <= 12:
                metrics['grade_level_category'] = 'High School'
            elif grade_level <= 16:
                metrics['grade_level_category'] = 'College Level'
            else:
                metrics['grade_level_category'] = 'Graduate Level'
            
            # Overall readability rating based on Flesch Reading Ease
            flesch_score = metrics['flesch_reading_ease']
            if flesch_score >= 90:
                metrics['overall_readability_rating'] = 'Very Easy'
            elif flesch_score >= 80:
                metrics['overall_readability_rating'] = 'Easy'
            elif flesch_score >= 70:
                metrics['overall_readability_rating'] = 'Fairly Easy'
            elif flesch_score >= 60:
                metrics['overall_readability_rating'] = 'Standard'
            elif flesch_score >= 50:
                metrics['overall_readability_rating'] = 'Fairly Difficult'
            elif flesch_score >= 30:
                metrics['overall_readability_rating'] = 'Difficult'
            else:
                metrics['overall_readability_rating'] = 'Very Difficult'
            
            # Calculate passive voice ratio
            passive_errors = [e for e in errors if e.get('type') == 'passive_voice']
            metrics['passive_voice_ratio'] = len(passive_errors) / len(sentences) if sentences else 0
            
            # Generate improvement priorities
            if not metrics['meets_target_grade']:
                if grade_level > self.rules['target_grade_level'][1]:
                    metrics['improvement_priority'].append('Reduce complexity')
                    metrics['readability_recommendations'].extend([
                        'Use shorter sentences',
                        'Replace complex words with simpler alternatives',
                        'Break up long paragraphs'
                    ])
                else:
                    metrics['improvement_priority'].append('Increase sophistication')
            
            if flesch_score < 60:
                metrics['improvement_priority'].append('Improve readability')
                metrics['readability_recommendations'].extend([
                    'Simplify sentence structure',
                    'Use active voice more frequently',
                    'Reduce average sentence length'
                ])
            
            if metrics['gunning_fog_index'] > 12:
                metrics['improvement_priority'].append('Reduce fog index')
                metrics['readability_recommendations'].append('Eliminate jargon and complex terminology')
            
        except Exception as e:
            logger.error(f"Error calculating technical writing metrics: {e}")
        
        return metrics
    
    def _calculate_overall_score(self, errors: List[Dict[str, Any]], statistics: Dict[str, Any], technical_metrics: Dict[str, Any]) -> float:
        """Calculate overall style score (0-100) with technical writing focus."""
        base_score = 100.0
        
        # Deduct points for errors
        for error in errors:
            severity = error.get('severity', 'medium')
            if severity == 'high':
                base_score -= 8
            elif severity == 'medium':
                base_score -= 5
            else:
                base_score -= 2
        
        # Technical writing specific adjustments
        try:
            # Grade level assessment (major factor)
            if technical_metrics.get('meets_target_grade', False):
                base_score += 10  # Bonus for meeting target grade level
            else:
                grade_level = technical_metrics.get('estimated_grade_level', 0)
                target_min, target_max = self.rules['target_grade_level']
                if grade_level > target_max:
                    # Penalty for being too complex
                    penalty = min(15, (grade_level - target_max) * 3)
                    base_score -= penalty
                elif grade_level < target_min:
                    # Smaller penalty for being too simple
                    penalty = min(5, (target_min - grade_level) * 2)
                    base_score -= penalty
            
            # Readability score adjustments
            flesch_score = statistics.get('flesch_reading_ease', 0)
            if flesch_score >= 70:
                base_score += 5  # Bonus for good readability
            elif flesch_score >= 60:
                base_score += 2  # Small bonus for acceptable readability
            elif flesch_score < 40:
                base_score -= 10  # Penalty for poor readability
            elif flesch_score < 50:
                base_score -= 5  # Smaller penalty for difficult readability
            
            # Gunning Fog Index penalty
            fog_index = statistics.get('gunning_fog_index', 0)
            if fog_index > 15:
                base_score -= 8  # High penalty for very complex text
            elif fog_index > 12:
                base_score -= 5  # Moderate penalty for complex text
            
            # Passive voice penalty
            passive_percentage = statistics.get('passive_voice_percentage', 0)
            if passive_percentage > 25:
                base_score -= 8  # High penalty for excessive passive voice
            elif passive_percentage > 15:
                base_score -= 4  # Moderate penalty for high passive voice
            
            # Sentence length variety bonus
            sentence_variety = statistics.get('sentence_length_variety', 0)
            if sentence_variety > 0.5:  # Good variety
                base_score += 3
            elif sentence_variety < 0.2:  # Poor variety
                base_score -= 3
            
            # Complex words penalty
            complex_percentage = statistics.get('complex_words_percentage', 0)
            if complex_percentage > 20:
                base_score -= 6  # High penalty for too many complex words
            elif complex_percentage > 15:
                base_score -= 3  # Moderate penalty
            
            # Technical complexity score adjustment
            complexity_score = statistics.get('technical_complexity_score', 0)
            if complexity_score > 75:
                base_score -= 5  # Penalty for high complexity
            elif complexity_score < 40:
                base_score += 3  # Bonus for appropriate simplicity
            
        except Exception as e:
            logger.error(f"Error in technical writing score calculation: {e}")
        
        # Ensure score is within bounds
        final_score = max(0, min(100, base_score))
        
        # Round to 1 decimal place
        return round(final_score, 1) 