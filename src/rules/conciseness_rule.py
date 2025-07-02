"""
Conciseness Rule - Backward compatibility import from modularized structure.
The actual implementation is now in src/rules/conciseness/ directory.

This maintains backward compatibility for existing code that imports:
from src.rules.conciseness_rule import ConcisenessRule
"""

# Import the modularized ConcisenessRule for backward compatibility
from .conciseness import ConcisenessRule

# Expose the class for existing imports
__all__ = ['ConcisenessRule'] 