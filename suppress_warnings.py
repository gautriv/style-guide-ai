#!/usr/bin/env python3
"""
Warning Suppression Configuration
Suppress common deprecation warnings to clean up terminal output
"""

import warnings
import os
import sys

def suppress_warnings():
    """Suppress common deprecation warnings"""
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
    warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node is deprecated.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="textstat")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="textstat")
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    # Suppress PyTorch warnings
    warnings.filterwarnings("ignore", message=".*weights_only.*")
    warnings.filterwarnings("ignore", message=".*FutureWarning.*")
    
    # Set environment variables to reduce noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only show errors from transformers
    
    # Suppress NLTK warnings
    import nltk
    nltk.data.path.append('.')
    
    print("🔇 Warning suppression activated - cleaner terminal output enabled!")

def configure_logging():
    """Configure logging to reduce noise"""
    import logging
    
    # Set specific loggers to WARNING level to reduce INFO noise
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    print("📝 Logging configured for minimal noise!")

if __name__ == "__main__":
    suppress_warnings()
    configure_logging()
    print("✅ All warnings suppressed successfully!") 