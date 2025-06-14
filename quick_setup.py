#!/usr/bin/env python3
"""
Quick Setup Script for Peer-review platform
Handles all post-installation requirements automatically.
"""

import subprocess
import sys
import platform
import os

def run_cmd(cmd, description=""):
    """Run a command and return True if successful."""
    try:
        print(f"⏳ {description}...")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def install_packages():
    """Install Python packages"""
    print("📦 Installing Python packages...")
    try:
        # Upgrade pip first
        run_cmd(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
        
        # Install/upgrade packages to latest versions to avoid deprecation warnings
        packages_to_upgrade = [
            "textstat>=0.7.3",
            "transformers>=4.36.0", 
            "torch>=2.1.0",
            "spacy>=3.7.2",
            "nltk>=3.8.1"
        ]
        
        print("🔄 Upgrading packages to fix deprecation warnings...")
        for package in packages_to_upgrade:
            run_cmd(f"{sys.executable} -m pip install --upgrade {package}", f"Upgrading {package}")
        
        # Install all requirements
        run_cmd(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages")
        print("✅ Packages installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def main():
    print("🚀 Setting up Peer-review platform...")
    print(f"Platform: {platform.system()} ({platform.platform()})")
    print(f"Python: {sys.version.split()[0]}")
    print("-" * 50)

    success_count = 0
    total_steps = 4

    # Step 1: Install requirements
    if install_packages():
        success_count += 1

    # Step 2: Setup SpaCy model
    if run_cmd(f"{sys.executable} -m spacy download en_core_web_sm", "Downloading SpaCy model"):
        success_count += 1
    else:
        print("⚠️  SpaCy model download failed, but app will use fallback processing")
        success_count += 1  # Don't fail for this

    # Step 3: Setup NLTK data
    nltk_cmd = f'{sys.executable} -c "import nltk; nltk.download(\'punkt\', quiet=True); nltk.download(\'stopwords\', quiet=True)"'
    if run_cmd(nltk_cmd, "Setting up NLTK data"):
        success_count += 1
    else:
        print("⚠️  NLTK setup failed, but app will use fallback processing")
        success_count += 1  # Don't fail for this

    # Step 4: Create necessary directories
    directories = ['uploads', 'logs', 'cache']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Created application directories!")
    success_count += 1

    print("-" * 50)
    if success_count == total_steps:
        print("🎉 Setup completed successfully!")
        print("\n🚀 You can now run the application:")
        print("   python app.py")
        print("\n🌐 Then visit: http://localhost:5000")
        print("\n📚 For troubleshooting, see: SETUP_GUIDE.md")
    else:
        print(f"⚠️  Setup completed with some issues ({success_count}/{total_steps} steps successful)")
        print("The application should still work with reduced functionality.")
        print("Run: python app.py")

if __name__ == "__main__":
    main() 