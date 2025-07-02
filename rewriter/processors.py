"""
Text Processing Module
Handles text cleaning, validation, and post-processing of AI-generated content.
"""

import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles cleaning and post-processing of AI-generated text."""
    
    def __init__(self):
        """Initialize the text processor."""
        pass
    
    def clean_generated_text(self, generated_text: str, original_text: str) -> str:
        """Clean and validate generated text, extracting only the rewritten content."""
        if not generated_text:
            logger.warning("Empty generated text")
            return original_text
        
        cleaned = generated_text.strip()
        logger.info(f"Raw AI response: '{cleaned[:200]}...'")
        
        # Remove meta-commentary and explanations more aggressively
        
        # Split into paragraphs and find the actual content
        paragraphs = cleaned.split('\n\n')
        content_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Skip paragraphs that are clearly meta-commentary
            meta_indicators = [
                'note:', 'i\'ve rewritten', 'i have rewritten', 'i applied', 'i made',
                'the following changes', 'here are the improvements', 'improvements made:',
                'changes made:', 'key improvements:', 'i converted', 'i removed',
                'i shortened', 'i replaced', 'the rewrite', 'this rewrite',
                'to address the issues', 'as requested', 'per your instructions'
            ]
            
            # Check if paragraph starts with meta-commentary
            para_lower = para.lower()
            is_meta = any(para_lower.startswith(indicator) for indicator in meta_indicators)
            
            # Also check if paragraph contains explanation patterns
            explanation_patterns = [
                'i\'ve', 'i have', 'i applied', 'i made', 'i converted', 'i removed',
                'to address', 'as you specified', 'per the guidelines', 'following the instructions'
            ]
            
            has_explanation = any(pattern in para_lower for pattern in explanation_patterns)
            
            # Skip if it's clearly meta-commentary
            if is_meta or (has_explanation and len(para.split()) < 50):
                logger.info(f"Skipping meta-commentary paragraph: '{para[:100]}...'")
                continue
            
            content_paragraphs.append(para)
        
        # Rejoin content paragraphs
        if content_paragraphs:
            cleaned = '\n\n'.join(content_paragraphs)
        
        # Remove common AI response prefixes
        prefixes_to_remove = [
            "here is the improved text:",
            "here's the improved text:",
            "improved text:",
            "rewritten text:",
            "revised text:",
            "the improved version:",
            "here is the rewrite:",
            "here's the rewrite:",
            "sure, here's",
            "certainly, here's",
            "here's a rewritten version:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Remove sentences that are clearly explanatory
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        content_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_lower = sentence.lower()
            
            # Skip explanatory sentences
            explanatory_starts = [
                'note:', 'i\'ve', 'i have', 'i applied', 'i made', 'i converted',
                'i removed', 'i shortened', 'i replaced', 'this addresses',
                'these changes', 'the rewrite', 'as requested', 'per your'
            ]
            
            is_explanatory = any(sentence_lower.startswith(start) for start in explanatory_starts)
            
            if not is_explanatory:
                content_sentences.append(sentence)
        
        if content_sentences:
            cleaned = ' '.join(content_sentences)
        
        # Remove any remaining artifacts
        cleaned = re.sub(r'\[insert[^\]]*\]', '', cleaned)  # Remove placeholder text like [insert specific examples]
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()
        
        logger.info(f"Cleaned AI response: '{cleaned[:200]}...'")
        
        # Validation
        if len(cleaned) < 10:
            logger.warning("Generated text too short after cleaning")
            return original_text
        
        # Check if it's meaningfully different from original
        if cleaned.lower().strip() == original_text.lower().strip():
            logger.warning("Generated text identical to original after cleaning")
            return original_text
        
        # Ensure proper sentence endings
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            # Find the last complete sentence
            sentences = re.split(r'[.!?]+', cleaned)
            if len(sentences) > 1:
                # Take all complete sentences except the last incomplete one
                complete_sentences = sentences[:-1]
                if complete_sentences:
                    cleaned = '. '.join(complete_sentences) + '.'
        
        logger.info(f"Final cleaned text: '{cleaned}'")
        return cleaned
    
    def rule_based_rewrite(self, content: str, errors: List[Dict[str, Any]]) -> str:
        """Fallback rule-based rewriting when AI models are not available."""
        rewritten = content
        
        try:
            for error in errors:
                error_type = error.get('type', '')
                
                if error_type == 'conciseness':
                    wordy_replacements = {
                        'in order to': 'to',
                        'due to the fact that': 'because',
                        'at this point in time': 'now',
                        'a large number of': 'many',
                        'make a decision': 'decide',
                        'for the purpose of': 'to',
                        'in spite of the fact that': 'although'
                    }
                    
                    for wordy, concise in wordy_replacements.items():
                        rewritten = re.sub(r'\b' + re.escape(wordy) + r'\b', concise, rewritten, flags=re.IGNORECASE)
                
                elif error_type == 'clarity':
                    complex_replacements = {
                        'utilize': 'use',
                        'facilitate': 'help',
                        'demonstrate': 'show',
                        'implement': 'do',
                        'commence': 'start',
                        'terminate': 'end'
                    }
                    
                    for complex_word, simple_word in complex_replacements.items():
                        rewritten = re.sub(r'\b' + complex_word + r'\b', simple_word, rewritten, flags=re.IGNORECASE)
                
                elif error_type == 'sentence_length':
                    sentence = error.get('sentence', '')
                    if sentence and sentence in rewritten:
                        if ' and ' in sentence and len(sentence.split()) > 20:
                            parts = sentence.split(' and ', 1)
                            if len(parts) == 2:
                                new_sentence = f"{parts[0].strip()}. {parts[1].strip()}"
                                rewritten = rewritten.replace(sentence, new_sentence)
            
            return rewritten
            
        except Exception as e:
            logger.error(f"Rule-based rewrite failed: {e}")
            return content
    
    def validate_text(self, text: str, original_text: str) -> Dict[str, Any]:
        """
        Validate processed text quality.
        
        Args:
            text: Processed text to validate
            original_text: Original text for comparison
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'word_count_original': len(original_text.split()),
            'word_count_processed': len(text.split()),
            'length_ratio': 0.0
        }
        
        try:
            # Check minimum length
            if len(text.strip()) < 10:
                validation['is_valid'] = False
                validation['issues'].append("Text too short after processing")
            
            # Check if identical to original
            if text.lower().strip() == original_text.lower().strip():
                validation['issues'].append("No changes made to original text")
            
            # Calculate length ratio
            if validation['word_count_original'] > 0:
                validation['length_ratio'] = validation['word_count_processed'] / validation['word_count_original']
                
                # Check for extreme length changes
                if validation['length_ratio'] > 2.0:
                    validation['issues'].append("Text significantly expanded (may contain unwanted content)")
                elif validation['length_ratio'] < 0.3:
                    validation['issues'].append("Text significantly reduced (may have lost important content)")
            
            # Check for proper sentence endings
            if text and not text.endswith(('.', '!', '?')):
                validation['issues'].append("Text does not end with proper punctuation")
            
        except Exception as e:
            logger.error(f"Text validation failed: {e}")
            validation['is_valid'] = False
            validation['issues'].append(f"Validation error: {str(e)}")
        
        return validation 