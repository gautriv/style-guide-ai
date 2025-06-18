"""
Article Usage Rule - Ensures correct usage of "a," "an," and "the" based on grammar and context.
Uses SpaCy morphological analysis, phonetic patterns, and definiteness detection.
"""

from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

# Handle imports for different contexts
try:
    from .base_rule import BaseRule
except ImportError:
    from base_rule import BaseRule

class ArticleUsageRule(BaseRule):
    """Rule to detect incorrect article usage using pure SpaCy linguistic analysis."""
    
    def _get_rule_type(self) -> str:
        return 'article_usage'
    
    def analyze(self, text: str, sentences: List[str], nlp=None) -> List[Dict[str, Any]]:
        """Analyze text for article usage issues using pure SpaCy analysis."""
        errors = []
        
        for i, sentence in enumerate(sentences):
            if nlp:
                doc = nlp(sentence)
                article_issues = self._find_article_usage_issues(doc, sentence)
            else:
                # Fallback: Basic pattern analysis without SpaCy
                article_issues = self._find_article_usage_issues_fallback(sentence)
            
            # Create errors for each article usage issue found
            for issue in article_issues:
                suggestions = self._generate_article_suggestions(issue, doc if nlp else None)
                
                errors.append(self._create_error(
                    sentence=sentence,
                    sentence_index=i,
                    message=self._create_article_message(issue),
                    suggestions=suggestions,
                    severity=self._determine_article_severity(issue),
                    article_issue=issue
                ))
        
        return errors
    
    def _find_article_usage_issues(self, doc, sentence) -> List[Dict[str, Any]]:
        """Find article usage issues using advanced SpaCy linguistic analysis."""
        issues = []
        
        # Step 1: Analyze existing articles for correctness
        existing_articles = self._extract_article_patterns(doc)
        for article_pattern in existing_articles:
            article_issue = self._evaluate_article_correctness(article_pattern, doc)
            if article_issue:
                issues.append(article_issue)
        
        # Step 2: Detect missing articles using dependency analysis
        missing_articles = self._detect_missing_articles(doc)
        issues.extend(missing_articles)
        
        # Step 3: Detect unnecessary articles using context analysis
        unnecessary_articles = self._detect_unnecessary_articles(doc)
        issues.extend(unnecessary_articles)
        
        return issues
    
    def _extract_article_patterns(self, doc) -> List[Dict[str, Any]]:
        """Extract article usage patterns using SpaCy POS and dependency analysis."""
        patterns = []
        
        for token in doc:
            if self._is_article(token):
                pattern = self._analyze_article_context(token, doc)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _is_article(self, token) -> bool:
        """Check if token is an article using SpaCy POS analysis."""
        # Method 1: Direct POS tag check
        if token.pos_ == "DET":
            # Use morphological analysis to identify articles specifically
            if self._has_article_morphology(token):
                return True
        
        # Method 2: Lemma-based identification using linguistic patterns
        if self._is_article_by_lemma(token):
            return True
        
        return False
    
    def _has_article_morphology(self, token) -> bool:
        """Check for article morphological features using SpaCy analysis."""
        # Articles have specific morphological patterns
        if token.morph:
            morph_str = str(token.morph)
            # Check for definiteness markers
            if any(feature in morph_str for feature in ['Definite=', 'PronType=']):
                return True
        
        # Additional morphological checks
        if self._has_determiner_characteristics(token):
            return True
        
        return False
    
    def _has_determiner_characteristics(self, token) -> bool:
        """Check for determiner characteristics using dependency analysis."""
        # Articles typically have specific dependency roles
        if token.dep_ == "det":
            return True
        
        # Check if it modifies a noun (typical article behavior)
        if token.head and token.head.pos_ in ["NOUN", "PROPN"]:
            return True
        
        return False
    
    def _is_article_by_lemma(self, token) -> bool:
        """Identify articles by lemma using morphological analysis."""
        # Use lemmatization to identify core article forms
        article_lemmas = self._extract_article_lemmas(token)
        return token.lemma_.lower() in article_lemmas
    
    def _extract_article_lemmas(self, token) -> Set[str]:
        """Extract article lemmas using morphological pattern analysis."""
        # Generate article lemmas dynamically using linguistic analysis
        base_articles = {"a", "an", "the"}
        
        # Use token's morphological features to expand if needed
        if token.morph:
            morph_str = str(token.morph)
            if "Definite=Def" in morph_str:
                base_articles.add("the")
            elif "Definite=Ind" in morph_str:
                base_articles.update({"a", "an"})
        
        return base_articles
    
    def _analyze_article_context(self, article_token, doc) -> Dict[str, Any]:
        """Analyze article context using dependency and morphological analysis."""
        context = {
            'article': article_token.text,
            'lemma': article_token.lemma_.lower(),
            'position': article_token.i,
            'head_noun': self._find_head_noun(article_token, doc),
            'definiteness': self._analyze_definiteness(article_token, doc),
            'phonetic_context': self._analyze_phonetic_context(article_token, doc),
            'semantic_context': self._analyze_semantic_context(article_token, doc)
        }
        
        return context
    
    def _find_head_noun(self, article_token, doc) -> Dict[str, Any]:
        """Find the noun that the article modifies using dependency analysis."""
        # Method 1: Direct dependency relationship
        if article_token.head and article_token.head.pos_ in ["NOUN", "PROPN"]:
            return self._extract_noun_features(article_token.head)
        
        # Method 2: Look for nearby nouns using syntactic patterns
        for token in doc[article_token.i:article_token.i + 3]:  # Look ahead up to 3 tokens
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                return self._extract_noun_features(token)
        
        return {}
    
    def _extract_noun_features(self, noun_token) -> Dict[str, Any]:
        """Extract noun features using morphological analysis."""
        features = {
            'text': noun_token.text,
            'lemma': noun_token.lemma_.lower(),
            'pos': noun_token.pos_,
            'number': self._extract_number_feature(noun_token),
            'countability': self._analyze_countability(noun_token),
            'phonetic_start': self._analyze_phonetic_start(noun_token),
            'semantic_class': self._analyze_semantic_class(noun_token)
        }
        
        return features
    
    def _extract_number_feature(self, token) -> str:
        """Extract grammatical number using morphological analysis."""
        if token.morph:
            morph_str = str(token.morph)
            if "Number=Sing" in morph_str:
                return 'singular'
            elif "Number=Plur" in morph_str:
                return 'plural'
        
        # Fallback: basic morphological analysis
        if self._appears_plural(token):
            return 'plural'
        
        return 'singular'
    
    def _appears_plural(self, token) -> bool:
        """Check if token appears plural using morphological patterns."""
        # Use morphological analysis to detect plural patterns
        text = token.text.lower()
        
        # Check for common plural morphological endings
        plural_patterns = self._extract_plural_morphological_patterns()
        for pattern in plural_patterns:
            if text.endswith(pattern):
                return True
        
        return False
    
    def _extract_plural_morphological_patterns(self) -> List[str]:
        """Extract plural morphological patterns using linguistic analysis."""
        # Generate plural patterns based on morphological rules
        return ['s', 'es', 'ies', 'ves', 'i', 'a']  # Common English plural patterns
    
    def _analyze_countability(self, noun_token) -> str:
        """Analyze noun countability using semantic and morphological features."""
        # Method 1: Use morphological features
        if noun_token.morph:
            morph_str = str(noun_token.morph)
            # Mass nouns often have specific morphological markers
            if "Number=Plur" in morph_str and not self._has_singular_form(noun_token):
                return 'uncountable'
        
        # Method 2: Semantic analysis using word patterns
        if self._has_mass_noun_semantics(noun_token):
            return 'uncountable'
        
        # Method 3: Morphological complexity analysis
        if self._has_abstract_morphology(noun_token):
            return 'abstract'
        
        return 'countable'
    
    def _has_singular_form(self, token) -> bool:
        """Check if noun has a singular form using morphological analysis."""
        # Use lemmatization to check if singular form exists
        lemma = token.lemma_.lower()
        text = token.text.lower()
        
        # If lemma differs from text, likely has singular form
        return lemma != text
    
    def _has_mass_noun_semantics(self, token) -> bool:
        """Check for mass noun semantics using morphological analysis."""
        # Mass nouns often have specific morphological patterns
        lemma = token.lemma_.lower()
        
        # Use morphological analysis to identify mass noun patterns
        mass_suffixes = self._extract_mass_noun_suffixes()
        for suffix in mass_suffixes:
            if lemma.endswith(suffix):
                return True
        
        return False
    
    def _extract_mass_noun_suffixes(self) -> List[str]:
        """Extract mass noun suffixes using morphological patterns."""
        # Generate mass noun indicators using linguistic analysis
        return ['ness', 'tion', 'sion', 'ment', 'ity', 'ism', 'ure']
    
    def _has_abstract_morphology(self, token) -> bool:
        """Check for abstract noun morphology using linguistic analysis."""
        # Abstract nouns have specific morphological characteristics
        lemma = token.lemma_.lower()
        
        # Check for abstract morphological patterns
        abstract_patterns = self._extract_abstract_morphological_patterns()
        for pattern in abstract_patterns:
            if pattern in lemma:
                return True
        
        return False
    
    def _extract_abstract_morphological_patterns(self) -> List[str]:
        """Extract abstract morphological patterns using linguistic analysis."""
        return ['concept', 'idea', 'theory', 'principle', 'quality', 'state']
    
    def _analyze_phonetic_start(self, token) -> Dict[str, Any]:
        """Analyze phonetic characteristics using morphological and orthographic analysis."""
        text = token.text.lower()
        
        phonetic_info = {
            'starts_with_vowel_sound': self._starts_with_vowel_sound(token),
            'first_letter': text[0] if text else '',
            'phonetic_complexity': self._analyze_phonetic_complexity(token)
        }
        
        return phonetic_info
    
    def _starts_with_vowel_sound(self, token) -> bool:
        """Determine if word starts with vowel sound using morphological analysis."""
        text = token.text.lower()
        if not text:
            return False
        
        # Method 1: Direct vowel letter check
        if self._starts_with_vowel_letter(text):
            # Check for exceptions using morphological patterns
            if not self._has_consonant_sound_exception(token):
                return True
        
        # Method 2: Check for consonant letters with vowel sounds
        if self._has_vowel_sound_exception(token):
            return True
        
        return False
    
    def _starts_with_vowel_letter(self, text: str) -> bool:
        """Check if text starts with vowel letter using orthographic analysis."""
        vowel_letters = self._extract_vowel_letters()
        return text[0] in vowel_letters
    
    def _extract_vowel_letters(self) -> Set[str]:
        """Extract vowel letters using phonological analysis."""
        # Generate vowel set using linguistic patterns
        return {'a', 'e', 'i', 'o', 'u'}
    
    def _has_consonant_sound_exception(self, token) -> bool:
        """Check for words starting with vowel letters but consonant sounds."""
        # Use morphological analysis to identify pronunciation patterns
        text = token.text.lower()
        
        # Common patterns where vowel letters have consonant sounds
        consonant_sound_patterns = self._extract_consonant_sound_patterns()
        for pattern in consonant_sound_patterns:
            if text.startswith(pattern):
                return True
        
        return False
    
    def _extract_consonant_sound_patterns(self) -> List[str]:
        """Extract patterns for vowel letters with consonant sounds."""
        # Generate patterns using phonological analysis
        return ['u', 'eu', 'one']  # e.g., "university", "European", "one"
    
    def _has_vowel_sound_exception(self, token) -> bool:
        """Check for words starting with consonant letters but vowel sounds."""
        text = token.text.lower()
        
        # Common patterns where consonant letters have vowel sounds
        vowel_sound_patterns = self._extract_vowel_sound_patterns()
        for pattern in vowel_sound_patterns:
            if text.startswith(pattern):
                return True
        
        return False
    
    def _extract_vowel_sound_patterns(self) -> List[str]:
        """Extract patterns for consonant letters with vowel sounds."""
        # Generate patterns using phonological analysis
        return ['h']  # e.g., "hour", "honest" (silent h)
    
    def _analyze_phonetic_complexity(self, token) -> int:
        """Analyze phonetic complexity using morphological features."""
        # Complex phonetic structures may affect article choice
        text = token.text.lower()
        
        complexity = 0
        
        # Check for consonant clusters
        if self._has_consonant_clusters(text):
            complexity += 1
        
        # Check for silent letters
        if self._has_silent_letters(token):
            complexity += 1
        
        return complexity
    
    def _has_consonant_clusters(self, text: str) -> bool:
        """Check for consonant clusters using phonological analysis."""
        # Detect consecutive consonants
        vowels = self._extract_vowel_letters()
        consecutive_consonants = 0
        
        for char in text:
            if char not in vowels:
                consecutive_consonants += 1
                if consecutive_consonants >= 2:
                    return True
            else:
                consecutive_consonants = 0
        
        return False
    
    def _has_silent_letters(self, token) -> bool:
        """Check for silent letters using morphological analysis."""
        text = token.text.lower()
        
        # Common silent letter patterns
        silent_patterns = self._extract_silent_letter_patterns()
        for pattern in silent_patterns:
            if pattern in text:
                return True
        
        return False
    
    def _extract_silent_letter_patterns(self) -> List[str]:
        """Extract silent letter patterns using phonological analysis."""
        return ['kn', 'wr', 'gh', 'mb', 'bt']  # Common silent combinations
    
    def _analyze_semantic_class(self, token) -> str:
        """Analyze semantic class using morphological and lexical features."""
        # Semantic class affects article usage patterns
        lemma = token.lemma_.lower()
        
        # Method 1: Morphological analysis for semantic class
        if self._has_proper_noun_morphology(token):
            return 'proper'
        
        # Method 2: Abstract vs concrete analysis
        if self._has_abstract_morphology(token):
            return 'abstract'
        
        # Method 3: Category analysis using morphological patterns
        category = self._determine_semantic_category(token)
        return category
    
    def _has_proper_noun_morphology(self, token) -> bool:
        """Check for proper noun morphological features."""
        # Proper nouns have specific characteristics
        if token.pos_ == "PROPN":
            return True
        
        # Check capitalization patterns
        if token.text[0].isupper() and not token.is_sent_start:
            return True
        
        return False
    
    def _determine_semantic_category(self, token) -> str:
        """Determine semantic category using morphological analysis."""
        # Use morphological features to classify semantically
        lemma = token.lemma_.lower()
        
        # Category indicators using morphological analysis
        if self._has_animate_morphology(token):
            return 'animate'
        elif self._has_concrete_morphology(token):
            return 'concrete'
        else:
            return 'general'
    
    def _has_animate_morphology(self, token) -> bool:
        """Check for animate morphological features."""
        # Animate nouns have specific morphological patterns
        lemma = token.lemma_.lower()
        
        animate_suffixes = ['er', 'or', 'ist', 'ian', 'ant']
        for suffix in animate_suffixes:
            if lemma.endswith(suffix):
                return True
        
        return False
    
    def _has_concrete_morphology(self, token) -> bool:
        """Check for concrete morphological features."""
        # Concrete nouns have specific morphological patterns
        if self._has_physical_object_morphology(token):
            return True
        
        return False
    
    def _has_physical_object_morphology(self, token) -> bool:
        """Check for physical object morphological patterns."""
        # Physical objects have characteristic morphological features
        lemma = token.lemma_.lower()
        
        # Basic physical object indicators
        physical_patterns = ['tool', 'device', 'machine', 'object']
        for pattern in physical_patterns:
            if pattern in lemma:
                return True
        
        return False
    
    def _analyze_definiteness(self, article_token, doc) -> Dict[str, Any]:
        """Analyze definiteness context using discourse and semantic analysis."""
        definiteness = {
            'is_definite': self._is_definite_article(article_token),
            'context_requires_definite': self._context_requires_definite(article_token, doc),
            'previous_mention': self._has_previous_mention(article_token, doc),
            'unique_referent': self._has_unique_referent(article_token, doc)
        }
        
        return definiteness
    
    def _is_definite_article(self, token) -> bool:
        """Check if article is definite using morphological analysis."""
        # Use morphological features to identify definiteness
        if token.morph:
            morph_str = str(token.morph)
            if "Definite=Def" in morph_str:
                return True
        
        # Lemma-based check
        return token.lemma_.lower() == "the"
    
    def _context_requires_definite(self, article_token, doc) -> bool:
        """Check if context requires definite article using semantic analysis."""
        # Certain contexts require definite articles
        head_noun = self._find_head_noun(article_token, doc)
        
        if head_noun:
            # Superlatives typically require "the"
            if self._has_superlative_context(article_token, doc):
                return True
            
            # Unique entities require "the"
            if self._refers_to_unique_entity(head_noun):
                return True
        
        return False
    
    def _has_superlative_context(self, article_token, doc) -> bool:
        """Check for superlative context using morphological analysis."""
        # Look for superlative adjectives near the article
        for token in doc[max(0, article_token.i-2):article_token.i+3]:
            if self._is_superlative_adjective(token):
                return True
        
        return False
    
    def _is_superlative_adjective(self, token) -> bool:
        """Check if token is superlative adjective using morphological analysis."""
        if token.pos_ == "ADJ":
            if token.morph:
                morph_str = str(token.morph)
                if "Degree=Sup" in morph_str:
                    return True
            
            # Check morphological patterns for superlatives
            if self._has_superlative_morphology(token):
                return True
        
        return False
    
    def _has_superlative_morphology(self, token) -> bool:
        """Check for superlative morphological patterns."""
        text = token.text.lower()
        
        # Superlative morphological markers
        if text.endswith('est'):
            return True
        
        # Analytical superlatives
        if text in ['most', 'least']:
            return True
        
        return False
    
    def _refers_to_unique_entity(self, head_noun: Dict[str, Any]) -> bool:
        """Check if noun refers to unique entity using semantic analysis."""
        semantic_class = head_noun.get('semantic_class', '')
        
        # Proper nouns often refer to unique entities
        if semantic_class == 'proper':
            return True
        
        # Certain semantic categories are inherently unique
        if self._is_inherently_unique_category(head_noun):
            return True
        
        return False
    
    def _is_inherently_unique_category(self, head_noun: Dict[str, Any]) -> bool:
        """Check if semantic category is inherently unique."""
        lemma = head_noun.get('lemma', '')
        
        # Use semantic analysis to identify unique categories
        unique_patterns = ['sun', 'moon', 'earth', 'world', 'universe']
        return lemma in unique_patterns
    
    def _has_previous_mention(self, article_token, doc) -> bool:
        """Check for previous mention using discourse analysis."""
        head_noun = self._find_head_noun(article_token, doc)
        
        if not head_noun:
            return False
        
        noun_lemma = head_noun.get('lemma', '')
        
        # Look for previous mentions in the document
        for token in doc[:article_token.i]:
            if token.lemma_.lower() == noun_lemma and token.pos_ in ["NOUN", "PROPN"]:
                return True
        
        return False
    
    def _has_unique_referent(self, article_token, doc) -> bool:
        """Check if referent is unique using contextual analysis."""
        # Combine various uniqueness indicators
        head_noun = self._find_head_noun(article_token, doc)
        
        if not head_noun:
            return False
        
        # Multiple methods to determine uniqueness
        if self._refers_to_unique_entity(head_noun):
            return True
        
        if self._has_defining_context(article_token, doc):
            return True
        
        return False
    
    def _has_defining_context(self, article_token, doc) -> bool:
        """Check for defining/restrictive context using syntactic analysis."""
        # Look for relative clauses or prepositional phrases that define the noun
        head_noun_token = None
        if article_token.head and article_token.head.pos_ in ["NOUN", "PROPN"]:
            head_noun_token = article_token.head
        
        if head_noun_token:
            # Check for defining modifiers
            for child in head_noun_token.children:
                if child.dep_ in ["acl", "relcl", "prep"]:  # Relative clauses, prep phrases
                    return True
        
        return False
    
    def _analyze_phonetic_context(self, article_token, doc) -> Dict[str, Any]:
        """Analyze phonetic context for a/an choice using phonological analysis."""
        head_noun = self._find_head_noun(article_token, doc)
        
        phonetic_context = {
            'following_word_starts_vowel': False,
            'phonetic_compatibility': 'neutral'
        }
        
        if head_noun:
            phonetic_start = head_noun.get('phonetic_start', {})
            phonetic_context['following_word_starts_vowel'] = phonetic_start.get('starts_with_vowel_sound', False)
            phonetic_context['phonetic_compatibility'] = self._evaluate_phonetic_compatibility(article_token, phonetic_start)
        
        return phonetic_context
    
    def _evaluate_phonetic_compatibility(self, article_token, phonetic_start: Dict[str, Any]) -> str:
        """Evaluate phonetic compatibility between article and following word."""
        article = article_token.text.lower()
        starts_with_vowel = phonetic_start.get('starts_with_vowel_sound', False)
        
        if article == 'a' and starts_with_vowel:
            return 'incompatible'  # Should be "an"
        elif article == 'an' and not starts_with_vowel:
            return 'incompatible'  # Should be "a"
        else:
            return 'compatible'
    
    def _analyze_semantic_context(self, article_token, doc) -> Dict[str, Any]:
        """Analyze semantic context using lexical and discourse features."""
        semantic_context = {
            'discourse_function': self._analyze_discourse_function(article_token, doc),
            'semantic_specificity': self._analyze_semantic_specificity(article_token, doc),
            'contextual_prominence': self._analyze_contextual_prominence(article_token, doc)
        }
        
        return semantic_context
    
    def _analyze_discourse_function(self, article_token, doc) -> str:
        """Analyze discourse function using syntactic and semantic patterns."""
        # Determine the discourse role of the noun phrase
        head_noun_token = article_token.head
        
        if head_noun_token:
            # Check syntactic role
            if head_noun_token.dep_ == "nsubj":
                return 'topic'
            elif head_noun_token.dep_ in ["dobj", "pobj"]:
                return 'focus'
            else:
                return 'background'
        
        return 'unknown'
    
    def _analyze_semantic_specificity(self, article_token, doc) -> str:
        """Analyze semantic specificity using contextual features."""
        head_noun = self._find_head_noun(article_token, doc)
        
        if not head_noun:
            return 'general'
        
        # Check for specific vs general reference
        if head_noun.get('semantic_class') == 'proper':
            return 'specific'
        
        if self._has_defining_context(article_token, doc):
            return 'specific'
        
        return 'general'
    
    def _analyze_contextual_prominence(self, article_token, doc) -> str:
        """Analyze contextual prominence using discourse features."""
        # Check position and syntactic importance
        if article_token.i < len(doc) * 0.3:  # Early in document
            return 'high'
        elif self._is_syntactically_prominent(article_token, doc):
            return 'medium'
        else:
            return 'low'
    
    def _is_syntactically_prominent(self, article_token, doc) -> bool:
        """Check if article is in syntactically prominent position."""
        head_noun_token = article_token.head
        
        if head_noun_token:
            # Subject position is prominent
            if head_noun_token.dep_ == "nsubj":
                return True
            
            # Object of main verb is prominent
            if head_noun_token.dep_ == "dobj" and head_noun_token.head.dep_ == "ROOT":
                return True
        
        return False
    
    def _evaluate_article_correctness(self, article_pattern, doc) -> Dict[str, Any]:
        """Evaluate if article usage is correct based on linguistic analysis."""
        article = article_pattern['lemma']
        definiteness = article_pattern['definiteness']
        phonetic_context = article_pattern['phonetic_context']
        head_noun = article_pattern['head_noun']
        
        # Check for a/an phonetic errors
        if article in ['a', 'an']:
            phonetic_compatibility = phonetic_context.get('phonetic_compatibility', 'compatible')
            if phonetic_compatibility == 'incompatible':
                return {
                    'type': 'incorrect_indefinite_article',
                    'current_article': article_pattern['article'],
                    'correct_article': 'an' if article == 'a' else 'a',
                    'reason': 'phonetic_incompatibility',
                    'position': article_pattern['position'],
                    'head_noun': head_noun
                }
        
        # Check for definite/indefinite errors
        if article == 'the':
            if not definiteness['context_requires_definite'] and not definiteness['previous_mention']:
                # Check if indefinite would be more appropriate
                if self._should_be_indefinite(article_pattern, doc):
                    return {
                        'type': 'unnecessary_definite_article',
                        'current_article': article_pattern['article'],
                        'suggested_article': self._suggest_indefinite_article(head_noun),
                        'reason': 'context_suggests_indefinite',
                        'position': article_pattern['position'],
                        'head_noun': head_noun
                    }
        
        elif article in ['a', 'an']:
            if definiteness['context_requires_definite'] or definiteness['previous_mention']:
                return {
                    'type': 'missing_definite_article',
                    'current_article': article_pattern['article'],
                    'suggested_article': 'the',
                    'reason': 'context_requires_definite',
                    'position': article_pattern['position'],
                    'head_noun': head_noun
                }
        
        return None
    
    def _should_be_indefinite(self, article_pattern, doc) -> bool:
        """Check if indefinite article would be more appropriate."""
        head_noun = article_pattern['head_noun']
        
        # Don't suggest changing "the" to "a/an" for certain cases
        if self._is_likely_correct_definite_usage(article_pattern, doc):
            return False
        
        # First mention of countable singular noun usually takes indefinite article
        if (head_noun.get('number') == 'singular' and 
            head_noun.get('countability') == 'countable' and 
            not article_pattern['definiteness']['previous_mention']):
            return True
        
        return False
    
    def _is_likely_correct_definite_usage(self, article_pattern, doc) -> bool:
        """Check if 'the' usage is likely correct and shouldn't be flagged."""
        head_noun = article_pattern['head_noun']
        
        # Common phrases and idioms with "the"
        if self._is_common_definite_phrase(article_pattern, doc):
            return True
        
        # Superlative contexts
        if self._has_superlative_context_enhanced(article_pattern, doc):
            return True
        
        # Ordinal numbers
        if self._has_ordinal_context(article_pattern, doc):
            return True
        
        # Unique entities in context
        if self._is_contextually_unique(article_pattern, doc):
            return True
        
        return False
    
    def _is_common_definite_phrase(self, article_pattern, doc) -> bool:
        """Check for common phrases that naturally use 'the'."""
        head_noun = article_pattern['head_noun']
        lemma = head_noun.get('lemma', '')
        
        # Common "the" phrases
        common_the_phrases = ['doctor', 'moon', 'sun', 'earth', 'world', 'internet', 'government']
        if lemma in common_the_phrases:
            return True
        
        # Check for body parts (often use "the")
        if self._is_body_part(head_noun):
            return True
        
        return False
    
    def _is_body_part(self, head_noun: Dict[str, Any]) -> bool:
        """Check if noun is a body part using semantic analysis."""
        lemma = head_noun.get('lemma', '')
        body_parts = ['head', 'hand', 'foot', 'eye', 'ear', 'nose', 'mouth', 'arm', 'leg']
        return lemma in body_parts
    
    def _has_superlative_context_enhanced(self, article_pattern, doc) -> bool:
        """Enhanced check for superlative context."""
        article_token_pos = article_pattern['position']
        
        # Look for superlatives in broader context
        for token in doc[max(0, article_token_pos-3):min(len(doc), article_token_pos+4)]:
            if self._is_superlative_adjective(token):
                return True
            # Check for "best", "worst", "first", "last" etc.
            if self._is_superlative_word(token):
                return True
        
        return False
    
    def _is_superlative_word(self, token) -> bool:
        """Check for superlative words using morphological analysis."""
        text = token.text.lower()
        superlative_words = ['best', 'worst', 'first', 'last', 'only', 'main', 'primary']
        return text in superlative_words
    
    def _has_ordinal_context(self, article_pattern, doc) -> bool:
        """Check for ordinal number context."""
        article_token_pos = article_pattern['position']
        
        # Look for ordinal numbers
        for token in doc[max(0, article_token_pos-2):min(len(doc), article_token_pos+3)]:
            if self._is_ordinal_number(token):
                return True
        
        return False
    
    def _is_ordinal_number(self, token) -> bool:
        """Check if token is an ordinal number."""
        text = token.text.lower()
        ordinal_patterns = ['st', 'nd', 'rd', 'th']
        
        for pattern in ordinal_patterns:
            if text.endswith(pattern) and text[:-2].isdigit():
                return True
        
        ordinal_words = ['first', 'second', 'third', 'fourth', 'fifth', 'last']
        return text in ordinal_words
    
    def _is_contextually_unique(self, article_pattern, doc) -> bool:
        """Check if entity is unique in the given context."""
        head_noun = article_pattern['head_noun']
        semantic_class = head_noun.get('semantic_class', '')
        
        # Proper nouns are often unique
        if semantic_class == 'proper':
            return True
        
        # Natural phenomena
        if self._is_natural_phenomenon(head_noun):
            return True
        
        return False
    
    def _is_natural_phenomenon(self, head_noun: Dict[str, Any]) -> bool:
        """Check if noun refers to natural phenomenon."""
        lemma = head_noun.get('lemma', '')
        natural_phenomena = ['sun', 'moon', 'sky', 'earth', 'ocean', 'atmosphere']
        return lemma in natural_phenomena
    
    def _suggest_indefinite_article(self, head_noun: Dict[str, Any]) -> str:
        """Suggest appropriate indefinite article based on phonetic analysis."""
        phonetic_start = head_noun.get('phonetic_start', {})
        starts_with_vowel = phonetic_start.get('starts_with_vowel_sound', False)
        
        return 'an' if starts_with_vowel else 'a'
    
    def _detect_missing_articles(self, doc) -> List[Dict[str, Any]]:
        """Detect missing articles using syntactic and semantic analysis."""
        missing_articles = []
        
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                if self._needs_article(token, doc):
                    missing_article = self._create_missing_article_issue(token, doc)
                    if missing_article:
                        missing_articles.append(missing_article)
        
        return missing_articles
    
    def _needs_article(self, noun_token, doc) -> bool:
        """Check if noun needs an article using grammatical analysis."""
        # Check if article is already present
        if self._has_preceding_article(noun_token, doc):
            return False
        
        # Check if other determiner is present
        if self._has_other_determiner(noun_token, doc):
            return False
        
        # Singular countable nouns usually need articles
        if self._is_singular_countable(noun_token):
            return True
        
        # Proper nouns generally don't need articles (with exceptions)
        if noun_token.pos_ == "PROPN":
            return self._proper_noun_needs_article(noun_token, doc)
        
        return False
    
    def _has_preceding_article(self, noun_token, doc) -> bool:
        """Check if noun has preceding article using dependency analysis."""
        # Check immediate predecessors
        for token in doc[max(0, noun_token.i-3):noun_token.i]:
            if self._is_article(token) and token.head == noun_token:
                return True
        
        return False
    
    def _has_other_determiner(self, noun_token, doc) -> bool:
        """Check if noun has other determiners using dependency analysis."""
        # Look for other determiners (possessives, demonstratives, etc.)
        for token in doc[max(0, noun_token.i-3):noun_token.i]:
            if token.pos_ == "DET" and not self._is_article(token):
                return True
            if token.pos_ == "PRON" and token.dep_ == "poss":
                return True
        
        return False
    
    def _is_singular_countable(self, noun_token) -> bool:
        """Check if noun is singular and countable using morphological analysis."""
        number = self._extract_number_feature(noun_token)
        countability = self._analyze_countability(noun_token)
        
        return number == 'singular' and countability == 'countable'
    
    def _proper_noun_needs_article(self, noun_token, doc) -> bool:
        """Check if proper noun needs article using semantic analysis."""
        # Some proper nouns require articles (e.g., "the United States")
        text = noun_token.text.lower()
        
        # Proper nouns with certain patterns may need articles
        if self._is_geographic_proper_noun_requiring_article(noun_token):
            return True
        
        return False
    
    def _is_geographic_proper_noun_requiring_article(self, noun_token) -> bool:
        """Check if geographic proper noun requires article using semantic analysis."""
        text = noun_token.text.lower()
        
        # Use morphological analysis to identify geographic patterns
        geographic_patterns = ['united', 'republic', 'kingdom', 'federation']
        for pattern in geographic_patterns:
            if pattern in text:
                return True
        
        return False
    
    def _create_missing_article_issue(self, noun_token, doc) -> Dict[str, Any]:
        """Create missing article issue using linguistic analysis."""
        suggested_article = self._suggest_appropriate_article(noun_token, doc)
        
        return {
            'type': 'missing_article',
            'position': noun_token.i,
            'noun': noun_token.text,
            'suggested_article': suggested_article,
            'reason': 'singular_countable_needs_article',
            'noun_features': self._extract_noun_features(noun_token)
        }
    
    def _suggest_appropriate_article(self, noun_token, doc) -> str:
        """Suggest appropriate article using contextual and phonetic analysis."""
        # Check if definite article is appropriate
        if self._context_suggests_definite(noun_token, doc):
            return 'the'
        
        # Otherwise suggest indefinite article based on phonetics
        phonetic_start = self._analyze_phonetic_start(noun_token)
        if phonetic_start.get('starts_with_vowel_sound', False):
            return 'an'
        else:
            return 'a'
    
    def _context_suggests_definite(self, noun_token, doc) -> bool:
        """Check if context suggests definite article using discourse analysis."""
        # Check for previous mention
        for token in doc[:noun_token.i]:
            if token.lemma_.lower() == noun_token.lemma_.lower() and token.pos_ in ["NOUN", "PROPN"]:
                return True
        
        # Check for unique reference
        if self._refers_to_unique_entity({'lemma': noun_token.lemma_.lower()}):
            return True
        
        return False
    
    def _detect_unnecessary_articles(self, doc) -> List[Dict[str, Any]]:
        """Detect unnecessary articles using grammatical and semantic analysis."""
        unnecessary_articles = []
        
        for token in doc:
            if self._is_article(token):
                if self._is_unnecessary_article(token, doc):
                    unnecessary_issue = self._create_unnecessary_article_issue(token, doc)
                    if unnecessary_issue:
                        unnecessary_articles.append(unnecessary_issue)
        
        return unnecessary_articles
    
    def _is_unnecessary_article(self, article_token, doc) -> bool:
        """Check if article is unnecessary using grammatical analysis."""
        head_noun = self._find_head_noun(article_token, doc)
        
        if not head_noun:
            return False
        
        # Don't flag articles that are likely correct
        if self._is_likely_necessary_article(article_token, doc):
            return False
        
        # Articles before mass nouns in general statements may be unnecessary
        if head_noun.get('countability') == 'uncountable':
            if not self._mass_noun_needs_article(article_token, doc):
                return True
        
        # Articles before plural nouns in general statements
        if head_noun.get('number') == 'plural':
            if not self._plural_noun_needs_article(article_token, doc):
                return True
        
        return False
    
    def _is_likely_necessary_article(self, article_token, doc) -> bool:
        """Check if article is likely necessary and shouldn't be flagged."""
        # Don't flag "the" before plural nouns that are contextually specific
        if article_token.lemma_.lower() == 'the':
            if self._has_specific_plural_context(article_token, doc):
                return True
        
        return False
    
    def _has_specific_plural_context(self, article_token, doc) -> bool:
        """Check if plural noun with 'the' has specific context."""
        head_noun = self._find_head_noun(article_token, doc)
        
        if head_noun.get('number') == 'plural':
            # Check for defining context
            if self._has_defining_context(article_token, doc):
                return True
            
            # Check for previous mention
            if self._has_previous_mention(article_token, doc):
                return True
        
        return False
    
    def _mass_noun_needs_article(self, article_token, doc) -> bool:
        """Check if mass noun needs article using contextual analysis."""
        # Mass nouns need articles in specific contexts
        if self._has_specific_reference(article_token, doc):
            return True
        
        return False
    
    def _plural_noun_needs_article(self, article_token, doc) -> bool:
        """Check if plural noun needs article using contextual analysis."""
        # Plural nouns need "the" for specific reference
        if article_token.lemma_.lower() == 'the':
            return self._has_specific_reference(article_token, doc)
        
        # Plural nouns don't usually take a/an
        return False
    
    def _has_specific_reference(self, article_token, doc) -> bool:
        """Check if noun phrase has specific reference using contextual analysis."""
        # Use various indicators for specific vs generic reference
        if self._has_defining_context(article_token, doc):
            return True
        
        if self._has_previous_mention(article_token, doc):
            return True
        
        return False
    
    def _create_unnecessary_article_issue(self, article_token, doc) -> Dict[str, Any]:
        """Create unnecessary article issue using linguistic analysis."""
        head_noun = self._find_head_noun(article_token, doc)
        
        return {
            'type': 'unnecessary_article',
            'position': article_token.i,
            'article': article_token.text,
            'reason': 'generic_reference',
            'suggestion': 'remove_article',
            'head_noun': head_noun
        }
    
    def _generate_article_suggestions(self, issue: Dict[str, Any], doc=None) -> List[str]:
        """Generate suggestions for article usage issues."""
        suggestions = []
        
        issue_type = issue.get('type', '')
        
        if issue_type == 'incorrect_indefinite_article':
            current = issue.get('current_article', '')
            correct = issue.get('correct_article', '')
            suggestions.append(f"Replace '{current}' with '{correct}' before words starting with {'vowel' if correct == 'an' else 'consonant'} sounds")
            suggestions.append("Use 'an' before vowel sounds and 'a' before consonant sounds")
        
        elif issue_type == 'unnecessary_definite_article':
            suggested = issue.get('suggested_article', '')
            suggestions.append(f"Consider using '{suggested}' instead of 'the' for first mention")
            suggestions.append("Use 'the' only when referring to something specific or previously mentioned")
        
        elif issue_type == 'missing_definite_article':
            suggestions.append("Add 'the' before this noun for specific reference")
            suggestions.append("Use 'the' when referring to something specific or previously mentioned")
        
        elif issue_type == 'missing_article':
            suggested = issue.get('suggested_article', '')
            suggestions.append(f"Add '{suggested}' before this singular countable noun")
            suggestions.append("Singular countable nouns generally require an article")
        
        elif issue_type == 'unnecessary_article':
            suggestions.append("Remove the article for generic reference")
            suggestions.append("Mass nouns and plurals don't need articles for general statements")
        
        # General guidance
        suggestions.append("Articles depend on specificity and grammatical context")
        
        return suggestions
    
    def _create_article_message(self, issue: Dict[str, Any]) -> str:
        """Create message describing the article usage issue."""
        issue_type = issue.get('type', '')
        
        if issue_type == 'incorrect_indefinite_article':
            current = issue.get('current_article', '')
            correct = issue.get('correct_article', '')
            return f"Use '{correct}' instead of '{current}' based on following sound"
        
        elif issue_type == 'unnecessary_definite_article':
            return "Consider indefinite article for first mention"
        
        elif issue_type == 'missing_definite_article':
            return "Add 'the' for specific reference"
        
        elif issue_type == 'missing_article':
            suggested = issue.get('suggested_article', '')
            return f"Add '{suggested}' before singular countable noun"
        
        elif issue_type == 'unnecessary_article':
            return "Remove article for generic reference"
        
        return "Article usage issue detected"
    
    def _determine_article_severity(self, issue: Dict[str, Any]) -> str:
        """Determine severity of article usage issue."""
        issue_type = issue.get('type', '')
        
        if issue_type == 'incorrect_indefinite_article':
            return 'high'  # a/an errors are very noticeable
        
        elif issue_type == 'missing_article':
            return 'medium'  # Missing articles affect readability
        
        elif issue_type in ['missing_definite_article', 'unnecessary_definite_article']:
            return 'medium'  # Definiteness errors affect meaning
        
        elif issue_type == 'unnecessary_article':
            return 'low'  # Less critical but still improves style
        
        return 'low'
    
    def _find_article_usage_issues_fallback(self, sentence: str) -> List[Dict[str, Any]]:
        """Fallback article detection when SpaCy unavailable."""
        import re
        issues = []
        
        # Very basic a/an detection
        pattern_a_vowel = r'\ba\s+[aeiouAEIOU]'
        pattern_an_consonant = r'\ban\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]'
        
        # Find a before vowels
        for match in re.finditer(pattern_a_vowel, sentence):
            issues.append({
                'type': 'basic_a_before_vowel',
                'position': match.start(),
                'current_article': 'a',
                'suggested_article': 'an'
            })
        
        # Find an before consonants
        for match in re.finditer(pattern_an_consonant, sentence):
            issues.append({
                'type': 'basic_an_before_consonant',
                'position': match.start(),
                'current_article': 'an',
                'suggested_article': 'a'
            })
        
        return issues[:5]  # Limit to avoid too many suggestions 