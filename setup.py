#!/usr/bin/env python3
"""
Cross-Platform Setup Script for Peer-review platform
Automatically handles all dependencies and post-installation setup.
Compatible with: Windows, macOS, Linux, Unix, Fedora, Ubuntu, etc.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description="Running command"):
    """Run a command and handle errors gracefully."""
    try:
        logger.info(f"{description}: {command}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to {description.lower()}: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr.strip()}")
        return False

def detect_platform():
    """Detect the current platform."""
    system = platform.system().lower()
    logger.info(f"Detected platform: {system} ({platform.platform()})")
    return system

def install_system_dependencies():
    """Install system-level dependencies based on the platform."""
    system = detect_platform()
    
    if system == "linux":
        # Detect Linux distribution
        try:
            with open('/etc/os-release', 'r') as f:
                os_release = f.read().lower()
            
            if 'ubuntu' in os_release or 'debian' in os_release:
                logger.info("Installing Ubuntu/Debian dependencies...")
                commands = [
                    "sudo apt-get update",
                    "sudo apt-get install -y python3-dev python3-pip",
                    "sudo apt-get install -y libmagic1 libmagic-dev",
                    "sudo apt-get install -y build-essential"
                ]
            elif 'fedora' in os_release or 'rhel' in os_release or 'centos' in os_release:
                logger.info("Installing Fedora/RHEL/CentOS dependencies...")
                commands = [
                    "sudo dnf update -y",
                    "sudo dnf install -y python3-devel python3-pip",
                    "sudo dnf install -y file-devel",
                    "sudo dnf install -y gcc gcc-c++ make"
                ]
            elif 'arch' in os_release:
                logger.info("Installing Arch Linux dependencies...")
                commands = [
                    "sudo pacman -Syu",
                    "sudo pacman -S python python-pip file gcc make"
                ]
            else:
                logger.warning("Unknown Linux distribution. Skipping system dependencies.")
                return True
                
            for cmd in commands:
                if not run_command(cmd, f"Installing system dependencies"):
                    logger.warning(f"Failed to run: {cmd}. You may need to install dependencies manually.")
                    
        except Exception as e:
            logger.warning(f"Could not install system dependencies: {e}")
            
    elif system == "darwin":  # macOS
        logger.info("Installing macOS dependencies...")
        # Check if Homebrew is installed
        if run_command("which brew", "Checking for Homebrew"):
            run_command("brew install libmagic", "Installing libmagic via Homebrew")
        else:
            logger.warning("Homebrew not found. You may need to install libmagic manually.")
            
    elif system == "windows":
        logger.info("Windows detected. Using pip-only installation.")
        # Windows dependencies are handled via pip
        
    return True

def install_python_packages():
    """Install Python packages from requirements.txt."""
    logger.info("Installing Python packages...")
    
    # Upgrade pip first
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    
    # Install packages
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        cmd = f"{sys.executable} -m pip install -r {requirements_file}"
        if not run_command(cmd, "Installing Python packages"):
            logger.error("Failed to install Python packages. Please check requirements.txt")
            return False
    else:
        logger.error("requirements.txt not found!")
        return False
        
    return True

def setup_spacy():
    """Download and setup SpaCy language models."""
    logger.info("Setting up SpaCy language models...")
    
    # Try different methods to install SpaCy models
    models = ["en_core_web_sm", "en_core_web_md"]
    
    for model in models:
        logger.info(f"Downloading SpaCy model: {model}")
        
        # Method 1: Using spacy download command
        if run_command(f"{sys.executable} -m spacy download {model}", f"Downloading {model}"):
            logger.info(f"Successfully installed {model}")
            break
            
        # Method 2: Using pip install
        elif run_command(f"{sys.executable} -m pip install https://github.com/explosion/spacy-models/releases/download/{model}-3.7.1/{model}-3.7.1-py3-none-any.whl", f"Installing {model} via pip"):
            logger.info(f"Successfully installed {model} via pip")
            break
    else:
        logger.warning("Could not install any SpaCy models. The app will use fallback text processing.")
        
    return True

def setup_nltk():
    """Download required NLTK data."""
    logger.info("Setting up NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = [
            'punkt',
            'stopwords', 
            'averaged_perceptron_tagger',
            'wordnet',
            'omw-1.4'
        ]
        
        for data in nltk_data:
            try:
                logger.info(f"Downloading NLTK data: {data}")
                nltk.download(data, quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK data '{data}': {e}")
                
    except ImportError:
        logger.error("NLTK not installed. Please install requirements.txt first.")
        return False
        
    return True

def create_directories():
    """Create necessary directories."""
    logger.info("Creating application directories...")
    
    directories = [
        "uploads",
        "logs", 
        "data",
        "cache"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
        
    return True

def test_installation():
    """Test if the installation was successful."""
    logger.info("Testing installation...")
    
    try:
        # Test critical imports
        import flask
        import spacy
        import nltk
        import textstat
        import pandas
        import numpy
        
        logger.info("✅ All critical packages imported successfully")
        
        # Test SpaCy model
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("✅ SpaCy model loaded successfully")
        except OSError:
            logger.warning("⚠️ SpaCy model not found, but fallback will work")
            
        # Test basic functionality
        from src.style_analyzer import StyleAnalyzer
        analyzer = StyleAnalyzer()
        test_result = analyzer.analyze("This is a test sentence.")
        logger.info("✅ Style analyzer working correctly")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Starting Peer-review platform setup...")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    
    steps = [
        ("Installing system dependencies", install_system_dependencies),
        ("Installing Python packages", install_python_packages),
        ("Setting up SpaCy", setup_spacy),
        ("Setting up NLTK", setup_nltk),
        ("Creating directories", create_directories),
        ("Testing installation", test_installation)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        logger.info(f"\n--- {step_name} ---")
        if not step_function():
            failed_steps.append(step_name)
            logger.error(f"❌ {step_name} failed")
        else:
            logger.info(f"✅ {step_name} completed")
    
    if failed_steps:
        logger.error(f"\n❌ Setup completed with errors in: {', '.join(failed_steps)}")
        logger.error("The application may still work, but some features might be limited.")
        return 1
    else:
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("\nYou can now run the application with:")
        logger.info("  python app.py")
        logger.info("\nOr visit: http://localhost:5000")
        return 0

if __name__ == "__main__":
    sys.exit(main())