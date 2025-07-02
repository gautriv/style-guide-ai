"""
Prompt Generation Module
Handles different prompt generation strategies for various AI models.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generates context-aware prompts based on detected errors and model type."""
    
    def __init__(self, use_ollama: bool = True):
        """Initialize the prompt generator."""
        self.use_ollama = use_ollama
    
    def generate_prompt(self, content: str, errors: List[Dict[str, Any]], context: str) -> str:
        """
        Generate a context-aware prompt based on detected errors.
        
        Args:
            content: Original text content
            errors: List of detected errors
            context: Context level ('sentence' or 'paragraph')
            
        Returns:
            Generated prompt string
        """
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
    
    def generate_self_review_prompt(self, first_rewrite: str, original_errors: List[Dict[str, Any]]) -> str:
        """Generate prompt for AI self-review and refinement (Pass 2)."""
        
        error_types = [error.get('type', '') for error in original_errors]
        error_summary = ', '.join(set(error_types))
        
        prompt = f"""You are a professional editor reviewing your own work for final polish.

YOUR FIRST REWRITE:
{first_rewrite}

ORIGINAL ISSUES ADDRESSED: {error_summary}

Please create a FINAL POLISHED VERSION that:
1. Maintains all improvements from your first rewrite
2. Enhances clarity and flow even further
3. Ensures perfect readability and professionalism
4. Keeps the original meaning intact

Be critical and look for any remaining opportunities to improve clarity, conciseness, or flow.

FINAL POLISHED VERSION:"""
        
        return prompt
    
    def _build_ollama_prompt(self, content: str, sentence_suggestions: List[str]) -> str:
        """Build optimized prompt for Ollama/Llama models."""
        
        if sentence_suggestions:
            suggestions_text = "\n".join(f"- {suggestion}" for suggestion in sentence_suggestions)
            prompt = f"""You are a professional technical writing editor. Rewrite the following text to address these specific issues:

{suggestions_text}

REWRITING GUIDELINES:
- Convert all passive voice to active voice
- Use simple, direct language instead of corporate jargon
- Break long sentences into shorter, clearer ones (15-20 words each)
- Remove unnecessary words and phrases
- Maintain the original meaning and all key information
- Write for a 9th-11th grade reading level

Original text:
{content}

Improved text:"""
        else:
            prompt = f"""You are a professional technical writing editor. Improve this text for clarity and conciseness:

REWRITING GUIDELINES:
- Use active voice throughout
- Choose simple, direct words over complex ones
- Keep sentences short and clear (15-20 words each)
- Remove unnecessary words and corporate jargon
- Maintain all original meaning and information
- Write for a 9th-11th grade reading level

Original text:
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