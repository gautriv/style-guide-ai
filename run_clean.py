#!/usr/bin/env python3
"""
Clean Startup Script for Peer-Review Platform
Suppresses warnings and provides clean terminal output
"""

import warnings
import os
import sys
import logging

def setup_clean_environment():
    """Setup clean environment with suppressed warnings"""
    
    # Suppress all warnings at the Python level
    warnings.filterwarnings("ignore")
    
    # Environment variables to reduce noise
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizer warnings
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only errors from transformers
    os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings
    
    # Configure logging to reduce noise
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('textstat').setLevel(logging.ERROR)
    
    # Disable specific warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def main():
    """Main function to start the application cleanly"""
    
    # Setup clean environment first
    setup_clean_environment()
    
    print("🎯 Starting Peer-Review Platform...")
    print("🔇 Warning suppression: ACTIVE")
    print("📍 Clean terminal mode: ENABLED")
    print("-" * 50)
    
    try:
        # Import and run the main application
        from app import app
        
        print("🌐 Server starting at: http://localhost:5000")
        print("🧠 AI Integration: Ready")
        print("✅ All systems operational!")
        print("-" * 50)
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to prevent double warnings
        )
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        print("👋 Thank you for using Peer-Review Platform!")
        
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("💡 Try running: python app.py")
        sys.exit(1)

if __name__ == "__main__":
    main() 