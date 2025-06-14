"""
AI Rewriter Module
Generates AI-powered rewriting suggestions based on detected style errors.
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import requests

logger = logging.getLogger(__name__)

class AIRewriter:
    """Handles AI-powered text rewriting based on style analysis errors."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", use_ollama: bool = False, ollama_model: str = "gemma:7b"):
        """Initialize the AI rewriter with a language model."""
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        # Initialize the appropriate model
        if use_ollama:
            self._test_ollama_connection()
        else:
            self._initialize_hf_model()
        
    
    def _test_ollama_connection(self):
        """Test if Ollama is running and the model is available."""
        try:
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                if self.ollama_model in available_models:
                    logger.info(f"✅ Ollama connected successfully. Using model: {self.ollama_model}")
                    self.use_ollama = True
                else:
                    logger.warning(f"⚠️ Model {self.ollama_model} not found in Ollama. Available models: {available_models}")
                    logger.info("You can pull it with: ollama pull gemma:2b")
                    self.use_ollama = False
            else:
                logger.warning("⚠️ Ollama is not responding properly")
                self.use_ollama = False
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Cannot connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            self.use_ollama = False
    
    def _initialize_hf_model(self):
        """Initialize the Hugging Face model for text generation."""
        if not HF_AVAILABLE:
            logger.warning("Transformers not available. Install with: pip install transformers torch")
            return
            
        try:
            logger.info(f"Initializing Hugging Face model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("✅ Hugging Face model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hugging Face model: {e}")
            self.generator = None
    
    def rewrite(self, content: str, errors: List[Dict[str, Any]], context: str = "sentence") -> Dict[str, Any]:
        """
        Generate AI-powered rewrite suggestions based on detected errors.
        
        Args:
            content: Original text content
            errors: List of detected errors
            context: Context level ('sentence' or 'paragraph')
            
        Returns:
            Dictionary with rewrite results
        """
        try:
            if not content or not content.strip():
                return {
                    'rewritten_text': '',
                    'improvements': [],
                    'confidence': 0.0,
                    'error': 'No content provided'
                }
            
            if not errors:
                return {
                    'rewritten_text': content,
                    'improvements': ['No errors detected'],
                    'confidence': 1.0
                }
            
            # Only use Ollama - no fallbacks
            if not self.use_ollama:
                return {
                    'rewritten_text': content,
                    'improvements': [],
                    'confidence': 0.0,
                    'error': 'Ollama is not available. AI rewriting requires a working Ollama connection.'
                }
            
            # Generate context-aware prompt
            prompt = self._generate_prompt(content, errors, context)
            
            # Generate rewrite using Ollama only
            rewritten_text = self._generate_with_ollama(prompt, content)
            
            # Check if rewrite actually made changes
            if rewritten_text == content:
                return {
                    'rewritten_text': content,
                    'improvements': [],
                    'confidence': 0.0,
                    'error': 'AI model failed to make meaningful improvements to the text'
                }
            
            # Extract improvements made
            improvements = self._extract_improvements(content, rewritten_text, errors)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(content, rewritten_text, errors)
            
            return {
                'rewritten_text': rewritten_text,
                'improvements': improvements,
                'confidence': confidence,
                'original_errors': len(errors),
                'model_used': 'ollama',
                'prompt_used': prompt[:200] + "..." if len(prompt) > 200 else prompt
            }
            
        except Exception as e:
            logger.error(f"Error in rewrite: {str(e)}")
            return {
                'rewritten_text': content,
                'improvements': [],
                'confidence': 0.0,
                'error': f'AI rewrite failed: {str(e)}'
            }
    
    def _generate_prompt(self, content: str, errors: List[Dict[str, Any]], context: str) -> str:
        """Generate a context-aware prompt based on detected errors."""
        
        # Build prompt using actual suggestions from StyleAnalyzer per sentence
        sentence_suggestions = []
        
        for error in errors:
            error_type = error.get('type', '')
            sentence = error.get('sentence', '')
            suggestions = error.get('suggestions', [])
            
            if sentence and suggestions:
                # Keep sentence-specific suggestions together
                for suggestion in suggestions:
                    sentence_suggestions.append(f"For '{sentence[:60]}...': {suggestion}")
        
        # Build enhanced prompt using actual detected suggestions
        if self.use_ollama:
            prompt = self._build_ollama_prompt(content, sentence_suggestions)
        else:
            prompt = self._build_hf_prompt(content, sentence_suggestions)
        
        return prompt
    
    def _build_ollama_prompt(self, content: str, sentence_suggestions: List[str]) -> str:
        """Build optimized prompt for Ollama/Gemma models."""
        
        if sentence_suggestions:
            suggestions_text = "\n".join(f"- {suggestion}" for suggestion in sentence_suggestions)
            prompt = f"""Apply these specific improvements:

{suggestions_text}

Original text:
{content}

Improved text:"""
        else:
            prompt = f"""Improve this text:

{content}

Improved text:"""
        
        return prompt
    
    def _build_hf_prompt(self, content: str, sentence_suggestions: List[str]) -> str:
        """Build prompt for Hugging Face models."""
        prompt_parts = [
            "Task: Improve the following text based on these specific issues:",
            "\n".join(f"- {ctx}" for ctx in sentence_suggestions),
            f"\nOriginal text: {content}",
            "\nImproved text:"
        ]
        return "\n".join(prompt_parts)
    
    def _generate_with_ollama(self, prompt: str, original_text: str) -> str:
        """Generate rewritten text using Ollama."""
        try:
            # Conservative parameters but with sufficient length for full rewrites
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Very low temperature for focused output
                    "top_p": 0.5,
                    "top_k": 10,        # Very focused vocabulary
                    "num_predict": 512,  # Increased from 100 to allow full text completion
                    "stop": ["\n\nOriginal:", "\n\nRewrite:", "###", "---"]  # Clear stop tokens
                }
            }
            
            logger.info(f"Sending prompt to Ollama: {prompt[:100]}...")
            
            response = requests.post(
                self.ollama_url,
                json=payload
                # No timeout - let the model take the time it needs
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                logger.info(f"Raw Ollama response: '{generated_text}'")
                
                # Clean and validate the output
                rewritten = self._clean_generated_text(generated_text, original_text)
                
                logger.info(f"Cleaned response: '{rewritten}'")
                
                return rewritten if rewritten else original_text
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return original_text
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return original_text
    
    def _generate_with_hf_model(self, prompt: str, original_text: str) -> str:
        """Generate rewritten text using Hugging Face model."""
        try:
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            if "Improved text:" in generated_text:
                rewritten = generated_text.split("Improved text:")[-1].strip()
            else:
                rewritten = generated_text.replace(prompt, "").strip()
            
            rewritten = self._clean_generated_text(rewritten, original_text)
            return rewritten if rewritten else original_text
            
        except Exception as e:
            logger.error(f"Hugging Face model generation failed: {e}")
            return original_text
    
    def _rule_based_rewrite(self, content: str, errors: List[Dict[str, Any]]) -> str:
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
    
    def _clean_generated_text(self, generated_text: str, original_text: str) -> str:
        """Clean and validate generated text, extracting only the rewritten content."""
        if not generated_text:
            logger.warning("Empty generated text")
            return original_text
        
        cleaned = generated_text.strip()
        logger.info(f"Raw AI response: '{cleaned[:200]}...'")
        
        # Extract the actual rewrite from common AI response patterns
        
        # Pattern 1: "Here is the improved text: [content]"
        if "here is the improved text:" in cleaned.lower():
            parts = cleaned.split(":")
            if len(parts) > 1:
                # Take everything after the first colon
                potential_rewrite = ":".join(parts[1:]).strip()
                # Remove any explanations that come after
                if "I applied" in potential_rewrite or "I made" in potential_rewrite:
                    potential_rewrite = potential_rewrite.split("I applied")[0].split("I made")[0].strip()
                if potential_rewrite:
                    cleaned = potential_rewrite
        
        # Pattern 2: "Improved text: [content]" or "Rewritten text: [content]"
        elif any(marker in cleaned.lower() for marker in ["improved text:", "rewritten text:", "revised text:"]):
            for marker in ["improved text:", "rewritten text:", "revised text:"]:
                if marker in cleaned.lower():
                    parts = cleaned.lower().split(marker)
                    if len(parts) > 1:
                        # Find the position in original case text
                        marker_pos = cleaned.lower().find(marker)
                        potential_rewrite = cleaned[marker_pos + len(marker):].strip()
                        if "I applied" in potential_rewrite or "I made" in potential_rewrite:
                            potential_rewrite = potential_rewrite.split("I applied")[0].split("I made")[0].strip()
                        if potential_rewrite:
                            cleaned = potential_rewrite
                        break
        
        # Remove meta-commentary that starts sentences
        if cleaned.lower().startswith(("sure", "here's", "i'll", "let me", "i can", "i will", "here is")):
            lines = cleaned.split('\n')
            for i, line in enumerate(lines):
                line_clean = line.strip()
                # Look for the first line that looks like actual content (not meta-commentary)
                if (len(line_clean) > 20 and 
                    not line_clean.lower().startswith(("sure", "here", "let me", "i'll", "i can", "i will", "i applied", "i made")) and
                    not line_clean.startswith(("1.", "2.", "3.", "*", "-"))):
                    cleaned = '\n'.join(lines[i:])
                    break
        
        # Remove numbered explanations and bullet points
        lines = cleaned.split('\n')
        content_lines = []
        explanation_started = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                if content_lines:  # Only add empty lines between content
                    content_lines.append('')
                continue
            
            # Check if this line starts an explanation section
            if (line_stripped.lower().startswith(("i applied", "i made", "the following changes", "here are the improvements")) or
                line_stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")) or
                line_stripped.startswith(("* ", "- ")) or
                "improvements:" in line_stripped.lower() or
                "changes made:" in line_stripped.lower()):
                explanation_started = True
                continue
            
            # If we've started explanations, skip everything
            if explanation_started:
                continue
            
            # Add content lines
            content_lines.append(line)
        
        cleaned = '\n'.join(content_lines).strip()
        
        # Remove common AI response artifacts
        artifacts_to_remove = [
            "Here is the improved text:",
            "Here's the improved text:",
            "Improved text:",
            "Rewritten text:",
            "Revised text:",
            "The improved version:",
            "Here is the rewrite:",
            "Here's the rewrite:"
        ]
        
        for artifact in artifacts_to_remove:
            if cleaned.startswith(artifact):
                cleaned = cleaned[len(artifact):].strip()
        
        # Remove trailing explanations
        stop_phrases = [
            "I applied all the specified improvements",
            "I made the following changes",
            "The following changes were made",
            "Here are the improvements",
            "Key improvements include",
            "Changes made:",
            "Improvements:"
        ]
        
        for phrase in stop_phrases:
            if phrase in cleaned:
                cleaned = cleaned.split(phrase)[0].strip()
        
        # Clean up any remaining artifacts
        cleaned = re.sub(r'\n\s*\*\s*.*$', '', cleaned, flags=re.MULTILINE | re.DOTALL)  # Remove bullet points at end
        cleaned = re.sub(r'\n\s*\d+\.\s*.*$', '', cleaned, flags=re.MULTILINE | re.DOTALL)  # Remove numbered lists at end
        
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
        
        # Final cleanup - ensure proper sentence endings
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
    
    def _extract_improvements(self, original: str, rewritten: str, errors: List[Dict[str, Any]]) -> List[str]:
        """Extract and describe the improvements made."""
        improvements = []
        
        original_words = len(original.split())
        rewritten_words = len(rewritten.split())
        
        if rewritten_words < original_words:
            improvements.append(f"Reduced word count from {original_words} to {rewritten_words}")
        
        error_types = set(error.get('type', '') for error in errors)
        
        if 'passive_voice' in error_types:
            improvements.append("Converted passive voice to active voice")
        
        if 'sentence_length' in error_types:
            improvements.append("Shortened overly long sentences")
        
        if 'conciseness' in error_types:
            improvements.append("Removed wordy phrases")
        
        if 'clarity' in error_types:
            improvements.append("Replaced complex words with simpler alternatives")
        
        if not improvements:
            improvements.append("Applied general style improvements")
        
        return improvements
    
    def _calculate_confidence(self, original: str, rewritten: str, errors: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the rewrite."""
        try:
            confidence = 0.7
            
            # Higher confidence for Ollama (local model)
            if self.use_ollama and rewritten != original:
                confidence += 0.3
            elif self.generator and rewritten != original:
                confidence += 0.2
            
            # Adjust based on number of errors addressed
            if errors:
                confidence += min(0.1, len(errors) * 0.02)
            
            # Penalize if no changes were made
            if rewritten == original:
                confidence -= 0.3
            
            # Check length ratio
            original_length = len(original.split())
            rewritten_length = len(rewritten.split())
            
            if original_length > 0:
                length_ratio = rewritten_length / original_length
                if length_ratio > 1.5 or length_ratio < 0.5:
                    confidence -= 0.2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def batch_rewrite(self, content_list: List[str], errors_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Rewrite multiple pieces of content in batch."""
        results = []
        
        for content, errors in zip(content_list, errors_list):
            result = self.rewrite(content, errors)
            results.append(result)
        
        return results 