"""
Core morphological analysis utilities for conciseness detection.
Uses pure SpaCy morphological, syntactic, and semantic analysis.
"""

from typing import List, Dict, Any

class MorphologicalAnalyzer:
    """Core morphological analysis utilities using pure SpaCy analysis."""
    
    def calculate_morphological_complexity_score(self, token) -> float:
        """Calculate morphological complexity using SpaCy features."""
        complexity = 0.0
        
        # Base complexity from morphological features
        morph_features = len(token.morph)
        complexity += morph_features / 10.0
        
        # Derivational complexity
        if self.has_complex_derivation(token):
            complexity += 0.4
        
        # Length proxy for morphological complexity
        complexity += len(token.text) / 15.0
        
        return min(complexity, 1.0)
    
    def has_complex_derivation(self, token) -> bool:
        """Check for complex derivational morphology."""
        lemma = token.lemma_.lower()
        
        # Multiple affixes suggest complexity
        prefixes = ['re-', 'pre-', 'dis-', 'un-', 'over-']
        suffixes = ['-ize', '-ify', '-ate', '-tion', '-sion', '-ment']
        
        has_prefix = any(lemma.startswith(prefix.rstrip('-')) for prefix in prefixes)
        has_suffix = any(lemma.endswith(suffix.lstrip('-')) for suffix in suffixes)
        
        return has_prefix and has_suffix
    
    def extract_morphological_root_advanced(self, lemma) -> str:
        """Extract morphological root using advanced pattern analysis."""
        # Remove derivational suffixes dynamically
        suffixes = ['-ment', '-ary', '-tion', '-sion', '-ness', '-ity', '-back']
        
        for suffix in suffixes:
            suffix_clean = suffix.lstrip('-')
            if lemma.endswith(suffix_clean) and len(lemma) > len(suffix_clean) + 2:
                return lemma[:-len(suffix_clean)]
        
        return lemma
    
    def have_phonological_similarity(self, root1, root2) -> bool:
        """Check phonological similarity using string patterns."""
        # Simple phonological similarity check
        if len(root1) >= 3 and len(root2) >= 3:
            # Check for shared consonant clusters or vowel patterns
            consonants1 = ''.join([c for c in root1 if c not in 'aeiou'])
            consonants2 = ''.join([c for c in root2 if c not in 'aeiou'])
            
            if consonants1 and consonants2:
                # Simple similarity: shared consonant patterns
                shared = sum(1 for c in consonants1 if c in consonants2)
                similarity = shared / max(len(consonants1), len(consonants2))
                return similarity > 0.5
        
        return False
    
    def has_latinate_formality_pattern(self, token) -> bool:
        """Check for Latinate formality patterns using morphology."""
        lemma = token.lemma_.lower()
        
        # Formal Latinate patterns
        if lemma.endswith('ize') or lemma.endswith('ate') or lemma.endswith('ify'):
            return True
        
        # Complex nominalization patterns
        if lemma.endswith('tion') or lemma.endswith('sion') or lemma.endswith('ment'):
            return True
        
        return False
    
    def has_formal_morphology(self, token) -> bool:
        """Check for formal morphological patterns."""
        lemma = token.lemma_.lower()
        
        # Formal prefixes + formal suffixes indicate bureaucratic language
        formal_prefixes = ['re-', 'pre-', 'de-', 'dis-']
        formal_suffixes = ['-ment', '-tion', '-ance', '-ence']
        
        has_formal_prefix = any(lemma.startswith(prefix.rstrip('-')) for prefix in formal_prefixes)
        has_formal_suffix = any(lemma.endswith(suffix.lstrip('-')) for suffix in formal_suffixes)
        
        return has_formal_prefix or has_formal_suffix
    
    def calculate_word_formality_by_morphology(self, lemma) -> float:
        """Calculate formality using morphological features."""
        formality_score = 0.0
        
        # Length is often a proxy for formality
        formality_score += len(lemma) / 10.0
        
        # Latinate endings are more formal
        if self.has_latinate_endings(lemma):
            formality_score += 0.4
        
        # Germanic/simple words are less formal
        if self.has_simple_patterns(lemma):
            formality_score -= 0.2
        
        return max(0.0, min(1.0, formality_score))
    
    def has_latinate_endings(self, lemma) -> bool:
        """Check for Latinate endings using morphological analysis."""
        # Latinate endings often indicate formal/academic register
        if lemma.endswith('ate') or lemma.endswith('ize') or lemma.endswith('ify'):
            return True
        
        if lemma.endswith('tion') or lemma.endswith('sion'):
            return True
        
        return False
    
    def has_simple_patterns(self, lemma) -> bool:
        """Check for simple Germanic patterns using morphological analysis."""
        # Simple patterns often indicate less formal register
        if lemma.endswith('ing') or lemma.endswith('ed') or lemma.endswith('er'):
            return True
        
        return False
    
    def is_content_word(self, token) -> bool:
        """Check if token is content word using SpaCy POS analysis."""
        return token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
    
    def is_derived_word(self, token) -> bool:
        """Check if word is morphologically derived."""
        derivational_suffixes = ['-tion', '-sion', '-ment', '-ance', '-ence', '-ity', '-ness', '-ful', '-less']
        lemma = token.lemma_.lower()
        return any(lemma.endswith(suffix.lstrip('-')) for suffix in derivational_suffixes)
    
    def estimate_syllable_count(self, word) -> int:
        """Estimate syllable count for formality assessment."""
        # Simple syllable estimation
        vowels = 'aeiou'
        syllables = 0
        prev_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Minimum one syllable
        return max(1, syllables)
    
    def get_morphological_features(self, token) -> Dict[str, Any]:
        """Get comprehensive morphological features from token."""
        return {
            'pos': token.pos_,
            'tag': token.tag_,
            'lemma': token.lemma_.lower(),
            'morph_features': dict(token.morph),
            'is_alpha': token.is_alpha,
            'is_stop': token.is_stop,
            'length': len(token.text)
        }
    
    def analyze_semantic_overlap(self, token1, token2) -> float:
        """Calculate semantic overlap using SpaCy vectors."""
        if token1.has_vector and token2.has_vector:
            return token1.similarity(token2)
        return 0.0
    
    def analyze_morphological_similarity(self, token1, token2) -> float:
        """Calculate morphological similarity between tokens."""
        root1 = self.extract_morphological_root_advanced(token1.lemma_.lower())
        root2 = self.extract_morphological_root_advanced(token2.lemma_.lower())
        
        if root1 == root2:
            return 1.0
        elif self.have_phonological_similarity(root1, root2):
            return 0.7
        else:
            return 0.0
    
    def has_terminology_morphology(self, token) -> bool:
        """Check for terminology morphological patterns."""
        # Technical terms often have specific morphological patterns
        lemma = token.lemma_.lower()
        
        # Method 1: Check for technical suffixes using morphology
        technical_suffixes = self.extract_technical_suffixes()
        for suffix in technical_suffixes:
            if lemma.endswith(suffix):
                return True
        
        # Method 2: Check morphological features
        if token.morph:
            morph_str = str(token.morph)
            if any(feature in morph_str for feature in ['Number=Sing', 'Definite=Def']):
                return True
        
        return False
    
    def extract_technical_suffixes(self) -> List[str]:
        """Extract technical suffixes using morphological analysis."""
        # Generate technical suffixes dynamically based on morphological patterns
        return ['tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ogy', 'logy']
    
    def is_latinate_verb_pattern(self, verb_token) -> bool:
        """Check for Latinate verb patterns using morphological analysis."""
        lemma = verb_token.lemma_.lower()
        
        # Common Latinate verb endings
        latinate_endings = ['-ate', '-ize', '-ify']
        return any(lemma.endswith(ending) for ending in latinate_endings)
    
    def analyze_sentence_morphological_density(self, sentence) -> Dict[str, Any]:
        """Analyze morphological density of sentence."""
        total_features = 0
        total_tokens = 0
        complexity_indicators = []
        
        for token in sentence:
            if self.is_content_word(token):
                total_tokens += 1
                feature_count = len(token.morph)
                total_features += feature_count
                
                # Identify complexity indicators
                if feature_count > 4:
                    complexity_indicators.append(f"complex_morphology_{token.text}")
                
                if self.is_derived_word(token):
                    complexity_indicators.append(f"derivation_{token.text}")
        
        density_score = total_features / total_tokens if total_tokens > 0 else 0
        
        return {
            'density_score': density_score,
            'is_high_density': density_score > 3.0,  # Threshold for high density
            'complexity_indicators': complexity_indicators
        }

class ContextAnalyzer:
    """Context analysis utilities using pure SpaCy analysis."""
    
    def is_professional_context(self, doc) -> bool:
        """Enhanced professional context detection."""
        # Method 1: Use SpaCy NER for professional entities
        professional_entity_count = 0
        for ent in doc.ents:
            if self.is_professional_entity(ent):
                professional_entity_count += 1
        
        if professional_entity_count > 0:
            return True
        
        # Method 2: Use morphological analysis for professional terms
        professional_pattern_count = 0
        for token in doc:
            if self.has_professional_morphology(token):
                professional_pattern_count += 1
        
        return professional_pattern_count > len(list(doc)) * 0.05
    
    def is_professional_entity(self, entity) -> bool:
        """Check if entity is professional using SpaCy analysis."""
        # Method 1: Organizational entities
        if entity.label_ == "ORG":
            return True
        
        # Method 2: Person entities in professional context
        if entity.label_ == "PERSON":
            return True
        
        # Method 3: Financial entities indicating business context
        if entity.label_ == "MONEY":
            return True
        
        return False
    
    def has_professional_morphology(self, token) -> bool:
        """Check for professional morphology patterns."""
        lemma = token.lemma_.lower()
        
        # Professional terms often have these patterns
        if 'committee' in lemma or 'organization' in lemma or 'policy' in lemma:
            return True
        
        if 'compliance' in lemma or 'governance' in lemma or 'audit' in lemma:
            return True
        
        if 'framework' in lemma or 'stakeholder' in lemma or 'guideline' in lemma:
            return True
        
        return False
    
    def is_business_context(self, doc) -> bool:
        """Enhanced business context detection using morphological analysis."""
        # Method 1: Named entity analysis
        business_entity_count = 0
        for ent in doc.ents:
            if self.is_business_related_entity(ent):
                business_entity_count += 1
        
        if business_entity_count > 0:
            return True
        
        # Method 2: Morphological pattern analysis for business terms
        business_pattern_count = 0
        for token in doc:
            if self.has_business_morphology(token):
                business_pattern_count += 1
        
        # If significant portion has business morphology, it's business context
        return business_pattern_count > len(list(doc)) * 0.05
    
    def is_business_related_entity(self, entity) -> bool:
        """Check if entity is business-related using morphological analysis."""
        # Method 1: Organizational entities
        if entity.label_ == "ORG":
            return True
        
        # Method 2: Financial/monetary entities
        if entity.label_ == "MONEY":
            return True
        
        # Method 3: Percentage/quantitative business entities
        if entity.label_ == "PERCENT":
            return True
        
        return False
    
    def has_business_morphology(self, token) -> bool:
        """Check for business morphology patterns."""
        lemma = token.lemma_.lower()
        
        # Business terms often have these patterns
        if 'policy' in lemma or 'guideline' in lemma or 'procedure' in lemma:
            return True
        
        if 'management' in lemma or 'organization' in lemma or 'business' in lemma:
            return True
        
        if 'strategy' in lemma or 'implement' in lemma or 'process' in lemma:
            return True
        
        if 'compliance' in lemma or 'governance' in lemma or 'operation' in lemma:
            return True
        
        return False
    
    def is_academic_context(self, doc) -> bool:
        """Detect academic context using morphological analysis."""
        academic_pattern_count = 0
        
        for token in doc:
            if self.has_academic_morphology(token):
                academic_pattern_count += 1
        
        # If significant portion has academic morphology, it's academic context
        return academic_pattern_count > len(list(doc)) * 0.05
    
    def has_academic_morphology(self, token) -> bool:
        """Check for academic morphology patterns."""
        lemma = token.lemma_.lower()
        
        # Academic terms often have these patterns
        if 'research' in lemma or 'stud' in lemma or 'analys' in lemma:
            return True
        
        if 'finding' in lemma or 'methodolog' in lemma or 'literatur' in lemma:
            return True
        
        if 'theoretical' in lemma or 'empirical' in lemma or 'hypothesis' in lemma:
            return True
        
        return False
    
    def is_technical_context(self, doc) -> bool:
        """Detect technical writing context using morphological patterns."""
        # Look for technical terms using morphological analysis
        technical_pattern_count = 0
        
        for token in doc:
            if self.has_technical_morphology(token):
                technical_pattern_count += 1
        
        # If significant portion has technical morphology, it's technical context
        return technical_pattern_count > len(list(doc)) * 0.1
    
    def has_technical_morphology(self, token) -> bool:
        """Check for technical morphology patterns."""
        lemma = token.lemma_.lower()
        
        # Technical terms often have these patterns
        if 'implement' in lemma or 'system' in lemma or 'process' in lemma:
            return True
        
        if 'protocol' in lemma or 'procedur' in lemma or 'methodolog' in lemma:
            return True
        
        return False
    
    def assess_formality_level(self, doc) -> float:
        """Assess document formality level."""
        formal_indicators = 0
        total_tokens = 0
        
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                total_tokens += 1
                if self.is_formal_term(token):
                    formal_indicators += 1
        
        return formal_indicators / total_tokens if total_tokens > 0 else 0.0
    
    def is_formal_term(self, token) -> bool:
        """Check if term is formal using morphological analysis."""
        # Use morphological complexity as formality indicator
        analyzer = MorphologicalAnalyzer()
        complexity_score = analyzer.calculate_morphological_complexity_score(token)
        
        # High morphological complexity suggests formality
        if complexity_score > 0.6:
            return True
        
        # Latinate patterns suggest formality
        if analyzer.has_latinate_formality_pattern(token):
            return True
        
        return False 