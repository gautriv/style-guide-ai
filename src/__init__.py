# Style Guide Application - Source Package 

# Backward compatibility: Re-export the new modular AIRewriter
# This allows existing imports like "from src.ai_rewriter import AIRewriter" to continue working
try:
    from ..rewriter import AIRewriter
    # Make AIRewriter available as if it were still in ai_rewriter module
    import sys
    from types import ModuleType
    
    # Create a fake ai_rewriter module for backward compatibility
    ai_rewriter_module = ModuleType('ai_rewriter')
    ai_rewriter_module.AIRewriter = AIRewriter
    sys.modules['src.ai_rewriter'] = ai_rewriter_module
    
except ImportError:
    # If the new rewriter package is not available, try the old one
    try:
        from .ai_rewriter import AIRewriter
    except ImportError:
        # Neither version available
        pass 