"""
Rules Registry - Automatically discovers and loads all writing rules.
This system allows for easy addition of new rules without modifying the main analyzer.
"""

import importlib
import importlib.util
import pkgutil
import os
from typing import List, Dict, Any

# Import base rule with proper path handling
try:
    from .base_rule import BaseRule
except ImportError:
    # Fallback for when running from different contexts
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from base_rule import BaseRule

class RulesRegistry:
    """Registry that automatically discovers and manages all writing rules."""
    
    def __init__(self):
        self.rules = {}
        self._load_all_rules()
    
    def _load_all_rules(self):
        """Automatically discover and load all rule modules."""
        try:
            # Get the current directory (rules directory)
            rules_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Find all Python files ending with '_rule.py'
            for filename in os.listdir(rules_dir):
                if filename.endswith('_rule.py') and filename != 'base_rule.py':
                    module_name = filename[:-3]  # Remove .py extension
                    
                    try:
                        # Import the module
                        if __package__:
                            # When imported as part of package
                            module = importlib.import_module(f'.{module_name}', __package__)
                        else:
                            # When running standalone
                            spec = importlib.util.spec_from_file_location(
                                module_name, 
                                os.path.join(rules_dir, filename)
                            )
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                        
                        # Find the rule class (should end with 'Rule')
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                issubclass(attr, BaseRule) and 
                                attr != BaseRule):
                                
                                # Instantiate the rule
                                rule_instance = attr()
                                self.rules[rule_instance.rule_type] = rule_instance
                                print(f"✅ Loaded rule: {rule_instance.rule_type}")
                                
                    except Exception as e:
                        print(f"⚠️ Failed to load rule {module_name}: {e}")
                        
        except Exception as e:
            print(f"⚠️ Failed to initialize rules registry: {e}")
    
    def get_rule(self, rule_type: str) -> BaseRule:
        """Get a specific rule by type."""
        return self.rules.get(rule_type)
    
    def get_all_rules(self) -> Dict[str, BaseRule]:
        """Get all loaded rules."""
        return self.rules
    
    def analyze_with_all_rules(self, text: str, sentences: List[str], nlp=None) -> List[Dict[str, Any]]:
        """Run analysis with all discovered rules."""
        all_errors = []
        
        for rule in self.rules.values():
            try:
                rule_errors = rule.analyze(text, sentences, nlp)
                
                # Ensure all errors are JSON serializable
                serializable_errors = []
                for error in rule_errors:
                    # Use base rule's serialization method
                    serializable_error = rule._make_serializable(error)
                    serializable_errors.append(serializable_error)
                
                all_errors.extend(serializable_errors)
                
            except Exception as e:
                print(f"❌ Error in rule {rule.__class__.__name__}: {e}")
                # Add a system error that is guaranteed to be serializable
                all_errors.append({
                    'type': 'system_error',
                    'message': f'Rule {rule.__class__.__name__} failed: {str(e)}',
                    'suggestions': ['Check rule implementation'],
                    'sentence': '',
                    'sentence_index': -1,
                    'severity': 'low'
                })
        
        return all_errors

# Global registry instance
registry = RulesRegistry() 