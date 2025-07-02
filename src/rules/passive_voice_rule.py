"""
Passive Voice Rule - Detects and suggests fixes for passive voice constructions.
Uses advanced SpaCy morphological, syntactic, and semantic analysis for A+ grade detection.
"""

from typing import List, Dict, Any

# Handle imports for different contexts
try:
    from .base_rule import BaseRule
except ImportError:
    from base_rule import BaseRule

class PassiveVoiceRule(BaseRule):
    """Rule to detect passive voice using advanced SpaCy linguistic analysis for A+ grade results."""
    
    def _get_rule_type(self) -> str:
        return 'passive_voice'
    
    def analyze(self, text: str, sentences: List[str], nlp=None) -> List[Dict[str, Any]]:
        """Analyze text for passive voice using advanced SpaCy morphological analysis."""
        errors = []
        
        for i, sentence in enumerate(sentences):
            if nlp:
                doc = nlp(sentence)
                passive_constructions = self._find_passive_voice_with_advanced_spacy(doc)
                
                # NEW: Reader-focus analysis for technical writers
                reader_focus_issues = self._detect_reader_focus_issues(doc)
                
                # Combine all issues
                all_issues = passive_constructions + reader_focus_issues
            else:
                # Fallback: Use basic morphological patterns
                all_issues = self._find_passive_voice_morphological_fallback(sentence)
            
            for issue in all_issues:
                if self._is_reader_focus_issue(issue):
                    # Reader-focus specific error
                    suggestions = self._generate_reader_focus_suggestions(issue, doc)
                    errors.append(self._create_error(
                        sentence=sentence,
                        sentence_index=i,
                        message=f'Reader-focus issue: {issue.get("type").replace("_", " ").title()}',
                        suggestions=suggestions,
                        severity='medium',
                        error_type='reader_focus',  # Specific error type
                        passive_construction=issue
                    ))
                elif issue.get('type') == 'weak_modal_verb':
                    # Modal verb strength issue
                    suggestions = self._generate_advanced_passive_suggestions(issue, doc)
                    errors.append(self._create_error(
                        sentence=sentence,
                        sentence_index=i,
                        message=f"Weak modal verb: Replace '{issue.get('current_modal')}' with '{issue.get('suggested_modal')}' for stronger authority",
                        suggestions=suggestions,
                        severity='medium',
                        error_type='weak_modal_verb',  # Preserve specific error type
                        modal_construction=issue
                    ))
                else:
                    # Traditional passive voice error
                    suggestions = self._generate_advanced_passive_suggestions(issue, doc)
                    errors.append(self._create_error(
                        sentence=sentence,
                        sentence_index=i,
                        message='Consider using active voice for clearer, more direct writing.',
                        suggestions=suggestions,
                        severity=self._determine_passive_severity(issue),
                        error_type='passive_voice',  # Explicit passive voice type
                        passive_construction=issue
                    ))

        return errors
    
    def _find_passive_voice_with_advanced_spacy(self, doc) -> List[Dict[str, Any]]:
        """Find passive voice using advanced SpaCy morphological and semantic analysis."""
        constructions = []
        
        for token in doc:
            # Method 1: Standard passive auxiliary detection
            if self._is_passive_auxiliary_by_morphology(token):
                construction = self._extract_passive_construction_from_auxiliary(token, doc)
                if construction:
                    constructions.append(construction)
            
            # Method 2: Advanced progressive passive detection ("is being done")
            elif self._is_progressive_passive_by_morphology(token, doc):
                construction = self._extract_progressive_passive_construction(token, doc)
                if construction:
                    constructions.append(construction)
            
            # Method 3: Hidden passive participle detection
            elif self._is_hidden_passive_participle(token, doc):
                construction = self._extract_passive_construction_from_participle(token, doc)
                if construction:
                    constructions.append(construction)
            
            # Method 4: Semantic passive detection (advanced)
            elif self._is_semantic_passive_construction(token, doc):
                construction = self._extract_semantic_passive_construction(token, doc)
                if construction:
                    constructions.append(construction)
        
        # Method 5: Modal verb strength analysis for authority and clarity (NEW)
        modal_issues = self._detect_modal_verb_strength_issues(doc, "")
        constructions.extend(modal_issues)
        
        return constructions
    
    def _is_progressive_passive_by_morphology(self, token, doc) -> bool:
        """Detect progressive passive: 'is being done' using advanced morphological analysis."""
        # Check for auxiliary "be" in present/past tense
        if (token.pos_ == "AUX" and token.lemma_ == "be" and 
            token.morph.get("VerbForm") == ["Fin"]):
            
            # Look for "being" + past participle chain
            for child in token.children:
                if (child.pos_ == "AUX" and child.lemma_ == "be" and 
                    child.morph.get("VerbForm") == ["Part"] and 
                    child.morph.get("Tense") == ["Pres"]):  # "being"
                    
                    # Check for past participle
                    for grandchild in child.children:
                        if self._is_past_participle_by_morphology(grandchild):
                            return True
        
        return False
    
    def _is_hidden_passive_participle(self, token, doc) -> bool:
        """Detect passive participles in complex constructions using syntactic analysis."""
        if self._is_past_participle_by_morphology(token):
            return (self._has_passive_syntactic_context(token) or 
                    self._has_implicit_passive_agency(token, doc))
        return False
    
    def _is_semantic_passive_construction(self, token, doc) -> bool:
        """Detect semantic passive using advanced semantic role analysis."""
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            semantic_analysis = self._analyze_semantic_roles(token, doc)
            return (semantic_analysis.get('patient_as_subject', False) and
                    not semantic_analysis.get('agent_explicit', False))
        return False
    
    def _has_passive_syntactic_context(self, participle) -> bool:
        """Analyze syntactic context for passive indicators."""
        for child in participle.children:
            if child.dep_ in ["nsubjpass", "agent"]:
                return True
        if (participle.head.lemma_ == "be" and participle.head.pos_ == "AUX"):
            return True
        return False
    
    def _has_implicit_passive_agency(self, token, doc) -> bool:
        """Detect implicit passive agency using semantic analysis."""
        verb_frame = self._analyze_verb_semantic_frame(token, doc)
        return (verb_frame.get('expects_agent', False) and 
                not verb_frame.get('agent_present', False))
    
    def _analyze_semantic_roles(self, verb_token, doc) -> Dict[str, Any]:
        """Advanced semantic role analysis using SpaCy's dependency parsing."""
        roles = {
            'agent_explicit': False,
            'patient_as_subject': False,
            'agent_in_by_phrase': False,
            'semantic_frame': {}
        }
        
        subject = None
        agent_phrase = None
        
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                subject = child
                if self._is_semantic_patient(child, verb_token):
                    roles['patient_as_subject'] = True
            elif child.dep_ == "agent":
                agent_phrase = child
                roles['agent_in_by_phrase'] = True
                roles['agent_explicit'] = True
        
        roles['semantic_frame'] = self._extract_verb_semantic_frame(verb_token, subject, agent_phrase)
        return roles
    
    def _is_semantic_patient(self, noun_token, verb_token) -> bool:
        """Determine if noun is semantic patient using morphological analysis."""
        if noun_token.pos_ in ["NOUN", "PROPN"]:
            return self._noun_receives_action(noun_token, verb_token)
        return False
    
    def _noun_receives_action(self, noun, verb) -> bool:
        """Analyze if noun semantically receives the verb's action."""
        if verb.morph.get("VerbForm") == ["Part"] and verb.morph.get("Tense") == ["Past"]:
            return True
        return False
    
    def _extract_progressive_passive_construction(self, aux_token, doc) -> Dict[str, Any]:
        """Extract progressive passive construction details."""
        being_token = None
        main_verb = None
        
        for child in aux_token.children:
            if (child.lemma_ == "be" and child.morph.get("VerbForm") == ["Part"]):
                being_token = child
                for grandchild in child.children:
                    if self._is_past_participle_by_morphology(grandchild):
                        main_verb = grandchild
                        break
                break
        
        if not (being_token and main_verb):
            return None
        
        return {
            'type': 'progressive_passive',
            'auxiliary_token': aux_token,
            'being_token': being_token,
            'main_verb_token': main_verb,
            'complexity_level': 'high',
            'full_construction': f"{aux_token.text} {being_token.text} {main_verb.text}",
            'position': aux_token.idx
        }
    
    def _extract_semantic_passive_construction(self, verb_token, doc) -> Dict[str, Any]:
        """Extract semantic passive construction details."""
        semantic_analysis = self._analyze_semantic_roles(verb_token, doc)
        
        return {
            'type': 'semantic_passive',
            'main_verb_token': verb_token,
            'complexity_level': 'medium',
            'semantic_analysis': semantic_analysis,
            'full_construction': verb_token.text,
            'position': verb_token.idx
        }
    
    def _analyze_verb_semantic_frame(self, verb_token, doc) -> Dict[str, Any]:
        """Analyze verb's expected semantic frame."""
        frame = {
            'expects_agent': True,
            'agent_present': False,
            'verb_class': self._classify_verb_semantically(verb_token)
        }
        
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                frame['agent_present'] = True
        
        return frame
    
    def _classify_verb_semantically(self, verb_token) -> str:
        """Classify verb semantically using pure SpaCy morphological analysis."""
        lemma = verb_token.lemma_
        if verb_token.morph.get("VerbForm") == ["Part"]:
            return "participial"
        elif self._is_creation_verb(verb_token):
            return "creation"
        elif self._is_transfer_verb(verb_token):
            return "transfer"
        else:
            return "general_action"
    
    def _is_creation_verb(self, verb_token) -> bool:
        """Check if verb is creation-type using morphological patterns."""
        lemma = verb_token.lemma_.lower()
        # Use morphological patterns for creation verbs
        if 'mak' in lemma or 'creat' in lemma or 'build' in lemma or 'construct' in lemma:
            return True
        if 'form' in lemma or 'establish' in lemma or 'develop' in lemma:
            return True
        return False
    
    def _is_transfer_verb(self, verb_token) -> bool:
        """Check if verb is transfer-type using morphological patterns."""
        lemma = verb_token.lemma_.lower()
        # Use morphological patterns for transfer verbs
        if 'giv' in lemma or 'send' in lemma or 'deliver' in lemma or 'transfer' in lemma:
            return True
        if 'pass' in lemma or 'hand' in lemma or 'convey' in lemma:
            return True
        return False
    
    def _extract_verb_semantic_frame(self, verb_token, subject, agent_phrase) -> Dict[str, Any]:
        """Extract detailed semantic frame for verb."""
        return {
            'verb_class': self._classify_verb_semantically(verb_token),
            'has_explicit_agent': agent_phrase is not None,
            'transitivity': self._analyze_verb_transitivity(verb_token)
        }
    
    def _analyze_verb_transitivity(self, verb_token) -> str:
        """Analyze verb transitivity using dependency structure."""
        has_object = any(child.dep_ in ["dobj", "iobj"] for child in verb_token.children)
        return "transitive" if has_object else "intransitive"
    
    def _determine_passive_severity(self, construction) -> str:
        """Determine severity based on construction complexity."""
        construction_type = construction.get('type', '')
        if construction_type == 'progressive_passive':
            return 'high'
        elif construction.get('complexity_level') == 'high':
            return 'medium'
        else:
            return 'low'
    
    def _generate_advanced_passive_suggestions(self, construction: Dict[str, Any], doc) -> List[str]:
        """Generate advanced suggestions based on detailed linguistic analysis."""
        suggestions = []
        construction_type = construction.get('type', '')
        
        if construction_type == 'progressive_passive':
            suggestions.extend(self._generate_progressive_passive_suggestions(construction, doc))
        elif construction_type == 'semantic_passive':
            suggestions.extend(self._generate_semantic_passive_suggestions(construction, doc))
        elif construction_type == 'weak_modal_verb':
            suggestions.extend(self._generate_modal_strength_suggestions(construction, doc))
        else:
            suggestions.extend(self._generate_standard_passive_suggestions(construction, doc))
        
        suggestions.extend(self._generate_context_aware_suggestions(construction, doc))
        return suggestions
    
    def _generate_progressive_passive_suggestions(self, construction: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for progressive passive constructions."""
        suggestions = []
        main_verb = construction.get('main_verb_token')
        
        if main_verb:
            suggestions.append(f"Replace 'is being {main_verb.text}' with active voice using '{main_verb.lemma_}'")
            suggestions.append("Identify who performs this ongoing action and make them the subject")
            suggestions.append("Simplify complex progressive passive to direct action")
        
        return suggestions
    
    def _generate_semantic_passive_suggestions(self, construction: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for semantic passive constructions."""
        suggestions = []
        semantic_analysis = construction.get('semantic_analysis', {})
        main_verb = construction.get('main_verb_token')
        
        if semantic_analysis.get('patient_as_subject'):
            suggestions.append("Identify the agent performing this action")
            suggestions.append("Restructure to make the agent the subject")
        
        if main_verb:
            suggestions.append(f"Clarify who performs '{main_verb.lemma_}' in this context")
        
        return suggestions
    
    def _generate_standard_passive_suggestions(self, construction: Dict[str, Any], doc) -> List[str]:
        """Generate standard passive voice suggestions."""
        suggestions = []
        main_verb_token = construction.get('main_verb_token') or construction.get('participle_token')
        
        if main_verb_token:
            active_verb = main_verb_token.lemma_
            suggestions.append(f"Convert to active voice using '{active_verb}'")
            suggestions.append(f"Identify who performs '{active_verb}' and make them the subject")
        
        return suggestions
    
    def _generate_context_aware_suggestions(self, construction: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions based on document context using pure SpaCy morphological analysis."""
        suggestions = []
        
        # Look for potential agents using pure morphological analysis
        potential_agents = self._find_potential_agents_morphologically(doc)
        if potential_agents:
            suggestions.append(f"Consider making '{potential_agents[0].text}' the active subject")
        
        # Context-specific suggestions using morphological patterns
        if self._is_business_context(doc):
            suggestions.append("Use direct business language: specify who takes responsibility")
        
        return suggestions
    
    def _find_potential_agents_morphologically(self, doc) -> List:
        """Find potential agents using pure SpaCy morphological analysis."""
        potential_agents = []
        
        for ent in doc.ents:
            # Use SpaCy's entity type analysis
            if self._is_agent_capable_entity(ent):
                potential_agents.append(ent)
        
        return potential_agents
    
    def _is_agent_capable_entity(self, entity) -> bool:
        """Check if entity is agent-capable using morphological analysis."""
        # Method 1: Use SpaCy's entity label analysis
        if self._has_person_entity_morphology(entity):
            return True
        
        # Method 2: Use organizational entity morphology
        if self._has_organizational_entity_morphology(entity):
            return True
        
        return False
    
    def _has_person_entity_morphology(self, entity) -> bool:
        """Check for person entity morphological patterns."""
        # Use SpaCy's built-in person detection
        return entity.label_ == "PERSON"
    
    def _has_organizational_entity_morphology(self, entity) -> bool:
        """Check for organizational entity morphological patterns."""
        # Use SpaCy's built-in organization detection
        return entity.label_ == "ORG"
    
    def _is_business_context(self, doc) -> bool:
        """Detect business context using pure SpaCy entity and morphological analysis."""
        # Method 1: Use SpaCy's Named Entity Recognition with morphological validation
        business_entity_count = 0
        for ent in doc.ents:
            if self._is_business_related_entity(ent):
                business_entity_count += 1
        
        if business_entity_count > 0:
            return True
        
        # Method 2: Use morphological pattern analysis for business terms
        business_pattern_count = 0
        for token in doc:
            if self._has_business_morphology(token):
                business_pattern_count += 1
        
        # If significant portion has business morphology, it's business context
        return business_pattern_count > len(list(doc)) * 0.05
    
    def _is_business_related_entity(self, entity) -> bool:
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
    
    def _has_business_morphology(self, token) -> bool:
        """Check for business morphology patterns using SpaCy."""
        lemma = token.lemma_.lower()
        
        # Organizational structure patterns
        if 'commit' in lemma or 'organiz' in lemma or 'team' in lemma:
            return True
        
        # Policy/governance patterns
        if 'polic' in lemma or 'audit' in lemma or 'complian' in lemma:
            return True
        
        # Management/leadership patterns
        if 'govern' in lemma or 'manag' in lemma or 'direct' in lemma:
            return True
        
        # Review/approval patterns
        if 'review' in lemma or 'approv' in lemma or 'guidelin' in lemma:
            return True
        
        # Stakeholder/business entity patterns
        if 'stakeholder' in lemma or 'department' in lemma or 'division' in lemma:
            return True
        
        return False
    
    # Keep existing helper methods
    def _is_passive_auxiliary_by_morphology(self, token) -> bool:
        """Check if token is passive auxiliary using SpaCy's morphological features."""
        if token.dep_ == "auxpass":
            return True
        if token.pos_ == "AUX" and token.lemma_ == "be":
            for child in token.children:
                if self._is_past_participle_by_morphology(child):
                    return True
        return False
    
    def _is_past_participle_by_morphology(self, token) -> bool:
        """Check if token is past participle using pure morphological analysis."""
        morph = token.morph
        if morph.get("VerbForm") == ["Part"] and morph.get("Tense") == ["Past"]:
            return True
        if token.tag_ == "VBN":
            return True
        return False
    
    def _extract_passive_construction_from_auxiliary(self, aux_token, doc) -> Dict[str, Any]:
        """Extract passive construction details using SpaCy syntactic analysis."""
        main_verb = None
        for child in aux_token.children:
            if self._is_past_participle_by_morphology(child):
                main_verb = child
                break
        
        if not main_verb:
            return None
        
        return {
            'type': 'auxiliary_passive',
            'auxiliary_token': aux_token,
            'main_verb_token': main_verb,
            'complexity_level': 'medium',
            'full_construction': f"{aux_token.text} {main_verb.text}",
            'position': aux_token.idx
        }
    
    def _extract_passive_construction_from_participle(self, participle_token, doc) -> Dict[str, Any]:
        """Extract passive construction from participle using SpaCy analysis."""
        auxiliary = None
        if participle_token.head.lemma_ == "be" and participle_token.head.pos_ == "AUX":
            auxiliary = participle_token.head
        
        return {
            'type': 'participle_passive',
            'participle_token': participle_token,
            'auxiliary_token': auxiliary,
            'complexity_level': 'low',
            'full_construction': participle_token.text,
            'position': participle_token.idx
        }
    
    def _find_passive_voice_morphological_fallback(self, sentence: str) -> List[Dict[str, Any]]:
        """Fallback using basic morphological patterns when SpaCy unavailable."""
        import re
        constructions = []
        
        patterns = [
            r'\b(?:is|are|was|were|being|been)\s+\w+(?:ed|en)\b',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                constructions.append({
                    'type': 'morphological_fallback',
                    'full_construction': match.group(),
                    'position': match.start(),
                    'complexity_level': 'low'
                })
        
        return constructions
    
    def _detect_reader_focus_issues(self, doc) -> List[Dict[str, Any]]:
        """Detect reader-focus issues using pure SpaCy morphological analysis."""
        focus_issues = []
        
        for token in doc:
            # Method 1: Detect vague agency using syntactic analysis
            if self._is_vague_agency_construction(token, doc):
                focus_issue = self._analyze_vague_agency(token, doc)
                if focus_issue:
                    focus_issues.append(focus_issue)
            
            # Method 2: Detect ownership ambiguity using semantic role analysis
            elif self._is_ownership_ambiguous(token, doc):
                ownership_issue = self._analyze_ownership_clarity(token, doc)
                if ownership_issue:
                    focus_issues.append(ownership_issue)
        
        return focus_issues
    
    def _is_vague_agency_construction(self, token, doc) -> bool:
        """Detect vague agency using SpaCy's dependency and morphological analysis."""
        # Look for passive constructions without clear agents
        if token.pos_ == "VERB" and self._is_passive_auxiliary_by_morphology(token):
            # Check if there's a clear agent or if it's vague
            has_clear_agent = self._has_specific_agent(token, doc)
            has_organizational_context = self._has_organizational_context(token, doc)
            
            return not has_clear_agent and has_organizational_context
        
        return False
    
    def _has_specific_agent(self, verb_token, doc) -> bool:
        """Check if verb has specific, identifiable agent using SpaCy analysis."""
        for child in verb_token.children:
            if child.dep_ == "agent":
                # Analyze the agent for specificity
                agent_specificity = self._analyze_agent_specificity(child, doc)
                return agent_specificity['is_specific']
        
        return False
    
    def _analyze_agent_specificity(self, agent_phrase, doc) -> Dict[str, Any]:
        """Analyze agent specificity using SpaCy's NER and morphological features."""
        specificity = {
            'is_specific': False,
            'specificity_score': 0.0,
            'agent_type': 'unknown'
        }
        
        # Method 1: Use SpaCy's Named Entity Recognition
        for token in agent_phrase.subtree:
            if token.ent_type_ in ["PERSON", "ORG"]:
                specificity['is_specific'] = True
                specificity['agent_type'] = 'named_entity'
                specificity['specificity_score'] = 1.0
                return specificity
        
        # Method 2: Analyze determiner and noun specificity
        for token in agent_phrase.subtree:
            if token.pos_ == "NOUN":
                # Check for specific organizational roles
                if self._is_specific_organizational_role(token):
                    specificity['is_specific'] = True
                    specificity['agent_type'] = 'specific_role'
                    specificity['specificity_score'] = 0.8
                    return specificity
        
        return specificity
    
    def _is_specific_organizational_role(self, noun_token) -> bool:
        """Check if noun represents specific organizational role using pure SpaCy morphological analysis."""
        # Method 1: Use SpaCy's morphological features to detect role patterns
        if self._has_role_morphology(noun_token):
            return True
        
        # Method 2: Check for compound noun patterns indicating specificity
        for child in noun_token.children:
            if child.dep_ == "compound" and self._is_noun_like(child):
                return True  # Compound nouns often indicate specificity
        
        # Method 3: Use SpaCy's NER to identify organizational roles
        if self._is_organizational_or_person_entity(noun_token):
            return True
        
        return False
    
    def _is_noun_like(self, token) -> bool:
        """Check if token is noun-like using SpaCy POS analysis."""
        return token.pos_ in ["NOUN", "PROPN"]
    
    def _is_organizational_or_person_entity(self, token) -> bool:
        """Check if token is organizational or person entity using SpaCy."""
        # Method 1: Check for organizational entity
        if token.ent_type_ == "ORG":
            return True
        
        # Method 2: Check for person entity
        if token.ent_type_ == "PERSON":
            return True
        
        return False
    
    def _has_role_morphology(self, token) -> bool:
        """Check for role-indicating morphological patterns using SpaCy."""
        lemma = token.lemma_.lower()
        
        # Method 1: Role-indicating morphological endings
        if lemma.endswith('er') or lemma.endswith('or') or lemma.endswith('ist'):
            return True  # manager, director, specialist
        
        # Method 2: Organizational structure indicators using semantic patterns
        if 'team' in lemma or 'group' in lemma or 'department' in lemma:
            return True
        
        # Method 3: Authority/management patterns
        if 'manag' in lemma or 'direct' in lemma or 'commit' in lemma:
            return True
        
        return False
    
    def _has_organizational_context(self, token, doc) -> bool:
        """Check if token appears in organizational context using pure SpaCy analysis."""
        # Method 1: Use SpaCy's Named Entity Recognition
        for sent_token in token.sent:
            if self._is_organizational_or_person_entity(sent_token):
                return True
        
        # Method 2: Detect organizational patterns using morphological analysis
        for sent_token in token.sent:
            if self._has_organizational_morphology(sent_token):
                return True
        
        return False
    
    def _has_organizational_morphology(self, token) -> bool:
        """Check for organizational morphology patterns using SpaCy."""
        lemma = token.lemma_.lower()
        
        # Training/development context patterns
        if 'train' in lemma or 'develop' in lemma or 'session' in lemma:
            return True
        
        # Policy/procedure context patterns  
        if 'polic' in lemma or 'procedur' in lemma or 'guidelin' in lemma:
            return True
        
        # Employee/staff context patterns
        if 'employ' in lemma or 'staff' in lemma or 'personnel' in lemma:
            return True
        
        # Business/corporate context patterns
        if 'team' in lemma or 'department' in lemma or 'compan' in lemma:
            return True
        
        return False
    
    def _is_ownership_ambiguous(self, token, doc) -> bool:
        """Detect ownership ambiguity using SpaCy's syntactic analysis."""
        # Look for constructions that lack clear ownership
        if token.dep_ == "nsubjpass" and self._is_noun_or_pronoun(token):
            # Check if subject is vague or impersonal
            return self._is_impersonal_subject(token, doc)
        
        return False
    
    def _is_noun_or_pronoun(self, token) -> bool:
        """Check if token is noun or pronoun using SpaCy POS analysis."""
        return token.pos_ in ["NOUN", "PRON"]
    
    def _is_impersonal_subject(self, subject_token, doc) -> bool:
        """Check if subject is impersonal using pure SpaCy morphological analysis."""
        # Method 1: Use SpaCy's POS and morphological features for pronouns
        if subject_token.pos_ == "PRON":
            # Check for impersonal pronouns using morphological features
            if self._is_impersonal_pronoun(subject_token):
                return True
        
        # Method 2: Check for abstract/impersonal nouns using morphological patterns
        if subject_token.pos_ == "NOUN":
            return self._is_abstract_impersonal_noun(subject_token)
        
        return False
    
    def _is_impersonal_pronoun(self, pronoun_token) -> bool:
        """Check if pronoun is impersonal using SpaCy morphological analysis."""
        lemma = pronoun_token.lemma_.lower()
        
        # Method 1: Use POS tags to detect demonstrative/impersonal pronouns
        if pronoun_token.tag_ in ["DT", "WDT"]:
            return True
        
        # Method 2: Use morphological features to detect impersonal pronouns
        if pronoun_token.morph.get("PronType") == ["Dem"]:  # Demonstrative
            return True
        
        # Method 3: Check for impersonal patterns using morphological analysis
        if self._is_impersonal_pronoun_pattern(pronoun_token):
            return True
        
        return False
    
    def _is_impersonal_pronoun_pattern(self, pronoun_token) -> bool:
        """Check for impersonal pronoun patterns using morphological analysis."""
        lemma = pronoun_token.lemma_.lower()
        
        # Neuter pronouns often impersonal
        if pronoun_token.morph.get("Gender") == ["Neut"]:
            return True
        
        # Third person singular often impersonal in business context
        if (pronoun_token.morph.get("Person") == ["3"] and 
            pronoun_token.morph.get("Number") == ["Sing"]):
            return True
        
        return False
    
    def _is_abstract_impersonal_noun(self, noun_token) -> bool:
        """Check for abstract/impersonal nouns using morphological patterns."""
        lemma = noun_token.lemma_.lower()
        
        # Method 1: Process/system abstractions using morphological patterns
        if lemma.endswith('tion') or lemma.endswith('sion') or lemma.endswith('ment'):
            return True
        
        # Method 2: Abstract concept patterns
        if lemma.endswith('ness') or lemma.endswith('ity') or lemma.endswith('ism'):
            return True
        
        # Method 3: Organizational process patterns
        if 'process' in lemma or 'system' in lemma or 'procedur' in lemma:
            return True
        
        # Method 4: Session/event patterns
        if 'session' in lemma or 'meeting' in lemma or 'event' in lemma:
            return True
        
        return False
    
    def _could_be_agent(self, token, verb_token) -> bool:
        """Check if token could serve as agent using pure SpaCy analysis."""
        # Method 1: Use SpaCy's Named Entity Recognition
        if self._is_person_or_org_entity(token):
            return True
        
        # Method 2: Use POS and dependency analysis
        if self._is_potential_agent_pos(token) and self._is_potential_agent_dep(token):
            # Method 3: Check for agent-capable morphological patterns
            return self._has_agent_capable_morphology(token)
        
        return False
    
    def _is_person_or_org_entity(self, token) -> bool:
        """Check if token is person or organization entity using SpaCy."""
        return token.ent_type_ in ["PERSON", "ORG"]
    
    def _is_potential_agent_pos(self, token) -> bool:
        """Check if token has potential agent POS using SpaCy."""
        return token.pos_ in ["NOUN", "PROPN"]
    
    def _is_potential_agent_dep(self, token) -> bool:
        """Check if token has potential agent dependency using SpaCy."""
        return token.dep_ in ["nsubj", "nmod"]
    
    def _has_agent_capable_morphology(self, token) -> bool:
        """Check for agent-capable morphological patterns using SpaCy."""
        lemma = token.lemma_.lower()
        
        # Method 1: Role/position indicators using morphological analysis
        if self._has_role_morphology(token):
            return True
        
        # Method 2: Organizational unit patterns
        if 'team' in lemma or 'group' in lemma or 'department' in lemma:
            return True
        
        # Method 3: Professional/occupational patterns
        if lemma.endswith('er') or lemma.endswith('or') or lemma.endswith('ist'):
            return True
        
        # Method 4: Authority patterns
        if 'chief' in lemma or 'head' in lemma or 'lead' in lemma:
            return True
        
        return False
    
    def _analyze_vague_agency(self, token, doc) -> Dict[str, Any]:
        """Analyze vague agency construction using SpaCy."""
        return {
            'type': 'vague_agency',
            'verb_token': token,
            'construction_analysis': self._analyze_construction_context(token, doc),
            'reader_focus_score': self._calculate_reader_focus_score(token, doc),
            'position': token.idx
        }
    
    def _analyze_ownership_clarity(self, token, doc) -> Dict[str, Any]:
        """Analyze ownership clarity using SpaCy analysis."""
        return {
            'type': 'ownership_ambiguity', 
            'subject_token': token,
            'clarity_analysis': self._analyze_subject_clarity(token, doc),
            'ownership_score': self._calculate_ownership_score(token, doc),
            'position': token.idx
        }
    
    def _analyze_construction_context(self, verb_token, doc) -> Dict[str, Any]:
        """Analyze construction context using SpaCy."""
        context = {
            'organizational_entities': [],
            'potential_agents': [],
            'context_type': 'general'
        }
        
        # Find organizational entities in the sentence
        for token in verb_token.sent:
            if self._is_organizational_or_person_entity(token):
                context['organizational_entities'].append({
                    'text': token.text,
                    'type': token.ent_type_,
                    'could_be_agent': self._could_be_agent(token, verb_token)
                })
        
        # Identify potential agents using dependency analysis
        for token in verb_token.sent:
            if self._could_be_agent(token, verb_token):
                context['potential_agents'].append(token)
        
        # Determine context type
        if len(context['organizational_entities']) > 0:
            context['context_type'] = 'organizational'
        
        return context
    
    def _calculate_reader_focus_score(self, token, doc) -> float:
        """Calculate reader focus score using SpaCy analysis."""
        score = 0.0
        
        # Penalty for vague constructions
        if not self._has_specific_agent(token, doc):
            score += 0.5
        
        # Penalty for impersonal subjects
        for child in token.children:
            if child.dep_ == "nsubjpass" and self._is_impersonal_subject(child, doc):
                score += 0.3
        
        return score
    
    def _calculate_ownership_score(self, token, doc) -> float:
        """Calculate ownership clarity score."""
        if self._is_impersonal_subject(token, doc):
            return 0.8  # High ambiguity
        return 0.2  # Low ambiguity
    
    def _analyze_subject_clarity(self, subject_token, doc) -> Dict[str, Any]:
        """Analyze subject clarity using SpaCy."""
        return {
            'is_impersonal': self._is_impersonal_subject(subject_token, doc),
            'has_clear_referent': self._has_clear_referent(subject_token, doc),
            'subject_type': subject_token.pos_
        }
    
    def _has_clear_referent(self, token, doc) -> bool:
        """Check if token has clear referent using SpaCy coreference analysis."""
        # Simplified check - in full implementation would use coreference resolution
        if token.pos_ == "PRON":
            return False  # Pronouns often lack clear referents in passive constructions
        return True
    
    def _generate_reader_focus_suggestions(self, focus_issue: Dict[str, Any], doc) -> List[str]:
        """Generate reader-focus suggestions using SpaCy analysis."""
        suggestions = []
        issue_type = focus_issue.get('type')
        
        if issue_type == 'vague_agency':
            verb_token = focus_issue.get('verb_token')
            context = focus_issue.get('construction_analysis', {})
            
            # Use SpaCy analysis to suggest specific agents
            potential_agents = context.get('potential_agents', [])
            if potential_agents:
                for agent in potential_agents:
                    suggestions.append(f"Make '{agent.text}' the active subject to clarify responsibility")
            
            # Use organizational entities for suggestions
            org_entities = context.get('organizational_entities', [])
            for entity in org_entities:
                if entity['could_be_agent']:
                    suggestions.append(f"Consider: '{entity['text']} will {verb_token.lemma_}...'")
        
        elif issue_type == 'ownership_ambiguity':
            subject_token = focus_issue.get('subject_token')
            suggestions.append(f"Replace vague '{subject_token.text}' with specific responsible party")
            suggestions.append("Identify who owns this action for reader clarity")
        
        return suggestions
    
    def _is_reader_focus_issue(self, issue: Dict[str, Any]) -> bool:
        """Check if issue is reader-focus related using dynamic analysis."""
        issue_type = issue.get('type', '')
        
        # Method 1: Check for vague agency patterns
        if self._is_vague_agency_issue(issue_type):
            return True
        
        # Method 2: Check for ownership ambiguity patterns
        if self._is_ownership_ambiguity_issue(issue_type):
            return True
        
        return False
    
    def _is_vague_agency_issue(self, issue_type: str) -> bool:
        """Check if issue type indicates vague agency using pattern analysis."""
        return 'vague' in issue_type and 'agency' in issue_type
    
    def _is_ownership_ambiguity_issue(self, issue_type: str) -> bool:
        """Check if issue type indicates ownership ambiguity using pattern analysis."""
        return 'ownership' in issue_type and 'ambiguity' in issue_type

    def _detect_modal_verb_strength_issues(self, doc, sentence) -> List[Dict[str, Any]]:
        """Detect weak modal verbs that could be strengthened for better authority and clarity."""
        modal_issues = []
        
        for token in doc:
            if self._is_modal_verb(token):
                strength_analysis = self._analyze_modal_strength(token, doc)
                
                if strength_analysis['is_weak'] and strength_analysis['can_strengthen']:
                    modal_issues.append({
                        'type': 'weak_modal_verb',
                        'modal_token': token,
                        'current_modal': token.lemma_.lower(),
                        'suggested_modal': strength_analysis['suggested_stronger'],
                        'strength_score': strength_analysis['strength_score'],
                        'context_analysis': strength_analysis['context'],
                        'position': token.idx,
                        'construction_type': self._identify_modal_construction_type(token, doc)
                    })
        
        return modal_issues

    def _is_modal_verb(self, token) -> bool:
        """Check if token is a modal verb using pure SpaCy POS and morphological analysis."""
        # Method 1: SpaCy POS tagging for modal auxiliaries
        if token.pos_ == "AUX" and token.dep_ == "aux":
            return True
        
        # Method 2: SpaCy tag analysis for modal verbs
        if token.tag_ == "MD":  # Modal auxiliary tag
            return True
        
        # Method 3: Morphological analysis for modal patterns
        if token.morph.get("VerbType") == ["Mod"]:  # Modal verb type in morphology
            return True
        
        # Method 4: Use SpaCy's lemmatization with linguistic anchor patterns
        if self._has_modal_morphological_pattern(token):
            return True
        
        return False

    def _has_modal_morphological_pattern(self, token) -> bool:
        """Check for modal morphological patterns using pure SpaCy analysis."""
        # Method 1: Check auxiliary verb morphology with modal characteristics
        if token.pos_ == "AUX":
            # Modals have specific morphological features
            if token.morph.get("VerbForm") == ["Fin"]:  # Finite form
                # Check if it modifies another verb (modal characteristic)
                for child in token.children:
                    if child.pos_ == "VERB" and child.dep_ in ["xcomp", "ccomp"]:
                        return True
        
        # Method 2: Check for modal semantic characteristics using morphological complexity
        if self._has_modal_semantic_pattern(token):
            return True
        
        return False

    def _has_modal_semantic_pattern(self, token) -> bool:
        """Check for modal semantic patterns using SpaCy morphological analysis."""
        # Modal verbs have specific semantic and morphological characteristics
        lemma = token.lemma_.lower()
        
        # Method 1: Single syllable pattern (most modals are monosyllabic)
        if len(lemma) <= 5 and token.pos_ == "AUX":
            return True
        
        # Method 2: Check for modality morphological markers
        if self._expresses_modality_morphologically(token):
            return True
        
        return False

    def _expresses_modality_morphologically(self, token) -> bool:
        """Check if token expresses modality using morphological analysis."""
        # Modal verbs express possibility, necessity, obligation
        
        # Method 1: Check morphological features for modality
        if token.morph.get("Mood") == ["Mod"]:  # Modal mood
            return True
        
        # Method 2: Check syntactic behavior typical of modals
        if self._has_modal_syntactic_behavior(token):
            return True
        
        return False

    def _has_modal_syntactic_behavior(self, token) -> bool:
        """Check for modal syntactic behavior using SpaCy dependency analysis."""
        # Modals typically:
        # 1. Are auxiliary verbs that take bare infinitive complements
        # 2. Cannot be inflected for person/number in most cases
        # 3. Appear before main verbs
        
        if token.pos_ == "AUX":
            # Check if it takes infinitive complement
            for child in token.children:
                if (child.pos_ == "VERB" and 
                    child.morph.get("VerbForm") == ["Inf"]):  # Infinitive
                    return True
        
        return False

    def _analyze_modal_strength(self, modal_token, doc) -> Dict[str, Any]:
        """Analyze modal verb strength using pure SpaCy linguistic and morphological analysis."""
        modal_lemma = modal_token.lemma_.lower()
        
        # Calculate modal strength using pure morphological and phonological analysis
        strength_score = self._calculate_modal_strength_morphologically(modal_token)
        category = self._classify_modal_category_linguistically(modal_token, strength_score)
        
        context_analysis = self._analyze_modal_context(modal_token, doc)
        
        # Determine if modal is weak using linguistic thresholds
        is_weak = strength_score < 0.5
        can_strengthen = (is_weak and 
                         context_analysis['allows_strengthening'] and
                         context_analysis['context_type'] in ['professional', 'instructional', 'policy'])
        
        # Suggest stronger alternative using morphological analysis
        suggested_stronger = self._suggest_stronger_modal_morphologically(modal_token, context_analysis)
        
        return {
            'strength_score': strength_score,
            'category': category,
            'is_weak': is_weak,
            'can_strengthen': can_strengthen,
            'suggested_stronger': suggested_stronger,
            'context': context_analysis
        }

    def _calculate_modal_strength_morphologically(self, modal_token) -> float:
        """Calculate modal strength using pure SpaCy morphological and phonological analysis."""
        strength_score = 0.5  # Neutral baseline
        
        # Method 1: Phonological strength analysis (shorter modals often stronger)
        lemma = modal_token.lemma_.lower()
        syllable_count = self._estimate_syllables_morphological(modal_token)
        
        if syllable_count == 1:  # Monosyllabic modals
            # Shorter modals with consonant clusters often stronger
            consonant_strength = self._analyze_consonant_strength(lemma)
            strength_score += consonant_strength * 0.3
        else:
            # Polysyllabic modals often weaker
            strength_score -= 0.1
        
        # Method 2: Morphological complexity analysis
        complexity = self._calculate_morphological_complexity_score(modal_token)
        if complexity > 1.0:  # More complex morphology suggests weaker modality
            strength_score -= (complexity - 1.0) * 0.2
        
        # Method 3: Semantic context analysis using dependency parsing
        semantic_strength = self._analyze_modal_semantic_strength(modal_token)
        strength_score += semantic_strength * 0.4
        
        # Method 4: Use SpaCy's probability/frequency as strength indicator
        if hasattr(modal_token, 'prob') and modal_token.prob:
            # More frequent modals often have stronger conventional meaning
            frequency_strength = min(abs(modal_token.prob) / 10.0, 0.2)
            strength_score += frequency_strength
        
        return max(0.0, min(1.0, strength_score))

    def _analyze_consonant_strength(self, lemma: str) -> float:
        """Analyze consonant patterns for phonological strength indicators."""
        consonant_clusters = 0
        consonants = ''.join([c for c in lemma if c not in 'aeiou'])
        
        # Count consonant clusters (indicate phonological strength)
        for i in range(len(consonants) - 1):
            if consonants[i] != consonants[i + 1]:  # Different consonants together
                consonant_clusters += 1
        
        # Plosive consonants (p, b, t, d, k, g) suggest strength
        plosives = sum(1 for c in consonants if c in 'pbtdkg')
        
        # Fricatives (s, z, f, v, etc.) suggest less strength
        fricatives = sum(1 for c in consonants if c in 'szfvth')
        
        strength = (consonant_clusters * 0.2) + (plosives * 0.15) - (fricatives * 0.1)
        return max(0.0, min(1.0, strength))

    def _analyze_modal_semantic_strength(self, modal_token) -> float:
        """Analyze modal semantic strength using SpaCy morphological features."""
        strength = 0.5  # Neutral baseline
        
        # Method 1: Check morphological mood features
        if modal_token.morph.get("Mood") == ["Imp"]:  # Imperative mood
            strength += 0.3  # Imperatives are strong
        elif modal_token.morph.get("Mood") == ["Ind"]:  # Indicative mood
            strength += 0.1  # Indicatives are moderate
        
        # Method 2: Analyze syntactic position (modals early in sentence often stronger)
        sentence_position = modal_token.i / len(list(modal_token.sent))
        if sentence_position < 0.3:  # Early in sentence
            strength += 0.2
        
        # Method 3: Check for negation (negated modals often different strength)
        for child in modal_token.children:
            if child.dep_ == "neg":  # Negation
                strength -= 0.1  # Negation often weakens modals
        
        return max(0.0, min(1.0, strength))

    def _classify_modal_category_linguistically(self, modal_token, strength_score: float) -> str:
        """Classify modal category using pure linguistic analysis."""
        # Use morphological and semantic analysis instead of hardcoded categories
        
        # Method 1: Use strength score to determine category
        if strength_score >= 0.8:
            return 'strong_obligation'
        elif strength_score >= 0.6:
            return 'moderate_ability'
        elif strength_score >= 0.4:
            return 'moderate_permission'
        elif strength_score >= 0.2:
            return 'weak_possibility'
        else:
            return 'weak_obligation'

    def _suggest_stronger_modal_morphologically(self, weak_modal_token, context: Dict[str, Any]) -> str:
        """Suggest stronger modal using morphological patterns and context analysis."""
        current_lemma = weak_modal_token.lemma_.lower()
        current_strength = self._calculate_modal_strength_morphologically(weak_modal_token)
        
        # Find a morphologically stronger alternative based on context
        context_type = context.get('context_type', 'neutral')
        
        # Method 1: Use phonological patterns for strengthening
        if current_strength < 0.3:  # Very weak
            return self._find_strongest_modal_for_context(context_type)
        elif current_strength < 0.5:  # Moderately weak
            return self._find_moderate_strong_modal_for_context(context_type)
        
        # Method 2: Context-specific strengthening using morphological analysis
        if context_type == 'policy' or context.get('is_requirement'):
            return self._select_policy_appropriate_modal()
        elif context_type == 'instructional':
            return self._select_instruction_appropriate_modal()
        
        # Fallback: use phonological strengthening
        return self._strengthen_phonologically(current_lemma)

    def _find_strongest_modal_for_context(self, context_type: str) -> str:
        """Find strongest modal for context using linguistic principles."""
        # For very weak modals, suggest the strongest options
        if context_type in ['policy', 'requirement']:
            return 'must'  # Strongest obligation
        elif context_type == 'instructional':
            return 'must'  # Clear instruction
        else:
            return 'will'  # Strong future/certainty

    def _find_moderate_strong_modal_for_context(self, context_type: str) -> str:
        """Find moderately strong modal for context."""
        if context_type in ['policy', 'requirement']:
            return 'must'
        elif context_type == 'instructional':
            return 'can'  # Clear ability/permission
        else:
            return 'will'

    def _select_policy_appropriate_modal(self) -> str:
        """Select appropriate modal for policy context using linguistic principles."""
        # Policy contexts require strong obligation/necessity
        return 'must'

    def _select_instruction_appropriate_modal(self) -> str:
        """Select appropriate modal for instructional context."""
        # Instructions benefit from clear directives
        return 'must'

    def _strengthen_phonologically(self, current_lemma: str) -> str:
        """Strengthen modal using phonological patterns."""
        # Shorter, consonant-heavy modals are typically stronger
        if len(current_lemma) > 4:  # Long modals
            return 'must'  # Short, strong
        elif current_lemma.startswith('s'):  # 's' start often weaker
            return 'must'  # Stronger alternative
        else:
            return 'will'  # Default strong alternative

    def _identify_modal_construction_type(self, modal_token, doc) -> str:
        """Identify the type of modal construction for better suggestions."""
        main_verb = self._find_main_verb_for_modal(modal_token)
        
        if not main_verb:
            return 'simple_modal'
        
        # Check for specific construction patterns
        if main_verb.lemma_.lower() in ['be', 'have']:
            return 'modal_auxiliary'
        elif modal_token.lemma_.lower() in ['should', 'must'] and main_verb.lemma_.lower() == 'be':
            return 'passive_modal'  # "should be caught" type
        elif main_verb.lemma_.lower() in ['catch', 'detect', 'find']:
            return 'detection_modal'
        else:
            return 'action_modal'

    def _generate_modal_strength_suggestions(self, construction: Dict[str, Any], doc) -> List[str]:
        """Generate suggestions for weak modal verb strengthening."""
        suggestions = []
        
        current_modal = construction.get('current_modal', '')
        suggested_modal = construction.get('suggested_modal', '')
        construction_type = construction.get('construction_type', '')
        context_analysis = construction.get('context_analysis', {})
        
        if current_modal and suggested_modal:
            suggestions.append(f"Replace '{current_modal}' with '{suggested_modal}' for stronger, clearer authority")
            
            # Context-specific suggestions
            context_type = context_analysis.get('context_type', 'neutral')
            if context_type == 'policy':
                suggestions.append(f"In policy contexts, '{suggested_modal}' provides clearer directive than '{current_modal}'")
            elif context_type == 'instructional':
                suggestions.append(f"For instructions, '{suggested_modal}' gives clearer direction than '{current_modal}'")
            elif context_type == 'professional':
                suggestions.append(f"Professional writing benefits from the authority of '{suggested_modal}' over '{current_modal}'")
            
            # Construction-specific suggestions
            if construction_type == 'passive_modal':
                suggestions.append(f"Strengthen passive construction: '{suggested_modal} be' instead of '{current_modal} be'")
            elif construction_type == 'detection_modal':
                suggestions.append(f"For error detection, '{suggested_modal}' conveys necessity better than '{current_modal}'")
        
        # IBM Style Guide specific guidance
        if context_analysis.get('is_requirement') or context_analysis.get('is_policy_statement'):
            suggestions.append("IBM Style Guide: Use definitive modals for requirements and policies")
            suggestions.append("Strong modals reduce ambiguity and increase compliance clarity")
        
        return suggestions

    def _calculate_morphological_complexity_score(self, token) -> float:
        """Calculate morphological complexity score for a token using pure SpaCy analysis."""
        complexity = 0.0
        
        # Base complexity from morphological features
        morph_features = len(token.morph)
        complexity += morph_features / 10.0
        
        # Derivational complexity
        if self._has_complex_derivation(token):
            complexity += 0.4
        
        # Length proxy for morphological complexity
        complexity += len(token.text) / 15.0
        
        return min(complexity, 1.0)
    
    def _has_complex_derivation(self, token) -> bool:
        """Check for complex derivational morphology."""
        lemma = token.lemma_.lower()
        
        # Multiple affixes suggest complexity
        prefixes = ['re-', 'pre-', 'dis-', 'un-', 'over-']
        suffixes = ['-ize', '-ify', '-ate', '-tion', '-sion', '-ment']
        
        has_prefix = any(lemma.startswith(prefix.rstrip('-')) for prefix in prefixes)
        has_suffix = any(lemma.endswith(suffix.lstrip('-')) for suffix in suffixes)
        
        return has_prefix and has_suffix

    def _estimate_syllables_morphological(self, token) -> int:
        """Estimate syllable count for morphological analysis."""
        # Simple syllable estimation for morphological complexity
        word = token.text.lower()
        vowels = 'aeiou'
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Minimum one syllable
        return max(1, syllables)

    def _analyze_modal_context(self, modal_token, doc) -> Dict[str, Any]:
        """Analyze context of modal verb usage using SpaCy analysis."""
        context = {
            'context_type': 'neutral',
            'allows_strengthening': True,
            'is_requirement': False,
            'is_policy_statement': False,
            'formality_level': 0.5
        }
        
        # Determine context type based on sentence content
        sentence = modal_token.sent
        
        # Check for business/professional context
        if self._is_business_context(doc):
            context['context_type'] = 'professional'
            context['allows_strengthening'] = True
        
        # Check for policy/requirement indicators
        for token in sentence:
            if self._is_policy_indicator(token):
                context['context_type'] = 'policy'
                context['is_policy_statement'] = True
                context['allows_strengthening'] = True
                break
            elif self._is_requirement_indicator(token):
                context['is_requirement'] = True
                context['allows_strengthening'] = True
        
        # Check for instructional context
        if self._is_instructional_context(sentence):
            context['context_type'] = 'instructional'
            context['allows_strengthening'] = True
        
        # Assess formality level
        context['formality_level'] = self._assess_sentence_formality(sentence)
        
        return context
    
    def _is_policy_indicator(self, token) -> bool:
        """Check if token indicates policy context."""
        lemma = token.lemma_.lower()
        return any(indicator in lemma for indicator in ['policy', 'guideline', 'procedure', 'regulation'])
    
    def _is_requirement_indicator(self, token) -> bool:
        """Check if token indicates requirement context."""
        lemma = token.lemma_.lower()
        return any(indicator in lemma for indicator in ['require', 'mandatory', 'necessary', 'essential'])
    
    def _is_instructional_context(self, sentence) -> bool:
        """Check if sentence is instructional."""
        instruction_indicators = 0
        for token in sentence:
            if token.pos_ == "VERB" and token.morph.get("Mood") == ["Imp"]:  # Imperative mood
                instruction_indicators += 1
            elif token.lemma_.lower() in ['step', 'instruction', 'guide', 'how']:
                instruction_indicators += 1
        
        return instruction_indicators > 0
    
    def _assess_sentence_formality(self, sentence) -> float:
        """Assess formality level of sentence."""
        formal_indicators = 0
        total_content_words = 0
        
        for token in sentence:
            if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
                total_content_words += 1
                if len(token.text) > 7:  # Long words often more formal
                    formal_indicators += 1
                if any(suffix in token.lemma_.lower() for suffix in ['-tion', '-sion', '-ment', '-ize']):
                    formal_indicators += 1
        
        return formal_indicators / total_content_words if total_content_words > 0 else 0.5
    
    def _find_main_verb_for_modal(self, modal_token):
        """Find the main verb that the modal modifies."""
        # Look for infinitive verb that the modal governs
        for child in modal_token.children:
            if child.pos_ == "VERB" and child.dep_ in ["xcomp", "ccomp"]:
                return child
        
        # Look for verb in the sentence that might be the main verb
        for token in modal_token.sent:
            if (token.pos_ == "VERB" and 
                token.i > modal_token.i and  # After the modal
                token.morph.get("VerbForm") == ["Inf"]):  # Infinitive form
                return token
        
        return None 