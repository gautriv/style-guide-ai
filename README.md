# 🎯 Peer-review platform

**AI-Powered Technical Writing Assistant with Local Ollama Integration**

Transform your technical documentation with comprehensive style analysis, readability scoring, and AI-powered rewriting capabilities. Specifically designed for technical writers targeting 9th-11th grade readability standards.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Cross-Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/yourusername/style-guide-analyzer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Start (2 Commands)

```bash
# 1. Automated setup (handles everything)
python quick_setup.py

# 2. Start the application
python app.py
```

**Then visit:** [http://localhost:5000](http://localhost:5000) 🌐

---

## ✨ Key Features

### 📊 **Comprehensive Technical Writing Analysis**
- **Grade Level Assessment** - Targets 9th-11th grade readability
- **Multiple Readability Scores** - Flesch Reading Ease, Gunning Fog, SMOG, Coleman-Liau, ARI
- **Style Issues Detection** - Passive voice, sentence length, wordiness, clarity
- **Technical Complexity Scoring** - Custom metrics for technical documentation

### 🧠 **AI-Powered Rewriting**
- **Local Ollama Integration** - Uses Gemma 7B model for privacy-first processing
- **Context-Aware Prompts** - Generates specific improvements based on detected errors
- **Intelligent Text Cleaning** - Extracts clean rewrites from AI responses
- **Fallback Support** - Works with cloud models or rule-based improvements

### 📁 **Multi-Format Document Support**
- **AsciiDoc (.adoc)** - Technical documentation standard
- **Markdown (.md)** - Developer documentation
- **DITA (.dita)** - Structured authoring
- **Microsoft Word (.docx)** - Business documents
- **PDF files** - Existing documentation
- **Direct text input** - Quick analysis

### 🎨 **Modern Web Interface**
- **Real-time Analysis** - Instant feedback on text quality
- **Interactive Error Highlighting** - Click to see specific issues
- **Comprehensive Statistics Dashboard** - All metrics at a glance
- **Responsive Design** - Works on desktop, tablet, and mobile

---

## 🛠️ Installation & Setup

### 📋 **Prerequisites**
- **Python 3.8+** ([Download here](https://www.python.org/downloads/))
- **Git** (optional, for cloning)

### 🚀 **Option 1: Quick Setup (Recommended)**
*No admin privileges required • Works on all platforms • 2-3 minutes*

```bash
# Clone or download the project
git clone https://github.com/yourusername/style-guide-analyzer.git
cd style-guide-analyzer

# Run automated setup
python quick_setup.py

# Start the application
python app.py
```

### 🔧 **Option 2: Full System Setup**
*Requires admin privileges • Installs system dependencies • 5-10 minutes*

```bash
# For comprehensive setup with system dependencies
python setup.py

# Start the application
python app.py
```

### 📋 **Option 3: Manual Setup**
*Step-by-step control • Good for troubleshooting*

```bash
# 1. Install Python packages
pip install -r requirements.txt

# 2. Download language models
python -m spacy download en_core_web_sm

# 3. Setup NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Create directories
mkdir uploads logs cache

# 5. Run the application
python app.py
```

---

## 🖥️ Platform-Specific Notes

### 🐧 **Linux (Ubuntu/Debian)**
```bash
# If you encounter issues, install system dependencies:
sudo apt-get update
sudo apt-get install python3-dev python3-pip libmagic1
```

### 🎩 **Linux (Fedora/RHEL/CentOS)**
```bash
# System dependencies for Fedora-based systems:
sudo dnf install python3-devel python3-pip file-devel
```

### 🍎 **macOS**
```bash
# Install Homebrew if needed:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install libmagic:
brew install libmagic
```

### 🪟 **Windows**
**No additional system dependencies required!** Python packages handle everything.

---

## 🎯 Usage Guide

### 🌐 **Web Interface**
1. **Start the application:** `python app.py`
2. **Open your browser:** [http://localhost:5000](http://localhost:5000)
3. **Choose input method:**
   - **Upload files** - Drag & drop or click to select
   - **Paste text** - Direct text input for quick analysis
   - **Try sample** - Use built-in examples

### 📊 **Understanding Results**

#### **Overall Score (0-100)**
- **80-100**: Excellent technical writing
- **60-79**: Good, minor improvements needed
- **40-59**: Needs significant improvement
- **Below 40**: Major rewriting required

#### **Grade Level Assessment**
- **Target**: 9th-11th grade for technical documentation
- **✅ Green**: Within target range
- **⚠️ Yellow**: Outside target (too simple/complex)

#### **Technical Writing Metrics**
- **Flesch Reading Ease**: 60+ is good for technical content
- **Gunning Fog Index**: Below 12 for accessibility
- **Passive Voice**: Keep under 15% for active writing
- **Complex Words**: Monitor percentage for clarity

### 🤖 **AI Rewriting**
1. **Analyze your text** to detect issues
2. **Click "Rewrite with AI"** for intelligent improvements
3. **Review suggestions** and copy improved text
4. **Re-analyze** to see improvement scores

---

## ⚙️ Configuration

### 🧠 **AI Model Configuration**
Edit `src/config.py` to customize AI settings:

```python
# Ollama (Local) Configuration
USE_OLLAMA = True
OLLAMA_MODEL = "gemma:7b"  # or "gemma:2b" for faster processing
OLLAMA_BASE_URL = "http://localhost:11434"

# Hugging Face (Cloud) Fallback
HF_MODEL_NAME = "microsoft/DialoGPT-medium"
```

### 📏 **Style Guide Rules**
Customize analysis rules in `src/style_analyzer.py`:

```python
self.rules = {
    'max_sentence_length': 25,        # Maximum words per sentence
    'target_grade_level': (9, 11),    # Target readability range
    'min_readability_score': 60.0,    # Minimum Flesch score
    'passive_voice_threshold': 0.15,  # Max 15% passive voice
}
```

---

## 🔧 Advanced Setup

### 🐋 **Docker Setup** (Coming Soon)
```bash
docker build -t style-guide-analyzer .
docker run -p 5000:5000 style-guide-analyzer
```

### 🤖 **Ollama Local AI Setup**
For privacy-first AI processing:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download Gemma model
ollama pull gemma:7b

# Start Ollama service
ollama serve
```

### 🏗️ **Development Setup**
```bash
# Install with development dependencies
pip install -r requirements.txt[dev]

# Run tests
pytest

# Code formatting
black .
isort .
flake8 .
```

---

## 📁 Project Structure

```
style-guide-analyzer/
├── 📄 app.py                  # Main Flask application
├── 📁 src/                    # Core modules
│   ├── 🧠 ai_rewriter.py      # AI-powered rewriting
│   ├── 📊 style_analyzer.py   # Style analysis engine
│   ├── 📁 document_processor.py # Multi-format text extraction
│   └── ⚙️ config.py           # Configuration management
├── 📁 templates/              # Web interface templates
│   ├── 🎨 base.html           # Main UI template
│   └── 🏠 index.html          # Homepage
├── 📁 uploads/                # File upload directory
├── 📁 logs/                   # Application logs
├── 📋 requirements.txt        # Python dependencies
├── 🚀 quick_setup.py          # Quick setup script
├── 🔧 setup.py               # Full setup script
├── 📖 SETUP_GUIDE.md          # Detailed setup instructions
└── 📄 README.md              # This file
```

---

## 🧪 API Endpoints

### **POST /analyze**
Analyze text content for style issues.

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "Your text here"}'
```

### **POST /rewrite**
Generate AI-powered rewrite suggestions.

```bash
curl -X POST http://localhost:5000/rewrite \
  -H "Content-Type: application/json" \
  -d '{"content": "Text to rewrite", "errors": [], "context": "paragraph"}'
```

### **POST /upload**
Upload and process documents.

```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@document.pdf"
```

### **GET /health**
Check system status and AI model availability.

```bash
curl http://localhost:5000/health
```

---

## 🐛 Troubleshooting

### **"SpaCy model not found"**
```bash
python -m spacy download en_core_web_sm
```

### **"NLTK data missing"**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### **"Ollama connection failed"**
```bash
# Start Ollama service
ollama serve

# Check if model is available
ollama list
```

### **"Import errors"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### **"Permission denied on Linux"**
```bash
# Install system dependencies
sudo apt-get install python3-dev libmagic1
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite:** `pytest`
5. **Format your code:** `black . && isort .`
6. **Commit your changes:** `git commit -m 'Add amazing feature'`
7. **Push to the branch:** `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### 🎯 **Areas We Need Help With**
- Additional language models integration
- More document format support
- Enhanced UI/UX improvements
- Performance optimizations
- Documentation improvements

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SpaCy** - Advanced natural language processing
- **NLTK** - Natural language toolkit
- **TextStat** - Readability statistics
- **Ollama** - Local AI model serving
- **Flask** - Web framework
- **Bootstrap** - UI components

---

## 📊 Performance & Requirements

### **System Requirements**
- **RAM**: 4GB minimum, 8GB recommended (with Ollama)
- **Storage**: 2GB for dependencies + models
- **CPU**: Any modern processor (AI benefits from multi-core)

### **Supported File Formats**
| Format | Extension | Notes |
|--------|-----------|-------|
| AsciiDoc | `.adoc` | Technical documentation |
| Markdown | `.md` | Developer docs |
| DITA | `.dita` | Structured authoring |
| Word | `.docx` | Business documents |
| PDF | `.pdf` | Existing documentation |
| Plain Text | `.txt` | Simple text files |

### **Performance Benchmarks**
- **Text Analysis**: < 1 second for 1000 words
- **AI Rewriting**: 2-5 seconds (local Ollama)
- **File Processing**: < 2 seconds for typical documents
- **Memory Usage**: 100-500MB (depending on models)

---

## 🔮 Roadmap

### **Version 2.0 (Planned)**
- [ ] Real-time collaborative editing
- [ ] Custom style guide creation
- [ ] Batch document processing
- [ ] API rate limiting and authentication
- [ ] Docker containerization
- [ ] CI/CD pipeline

### **Version 2.1 (Future)**
- [ ] Integration with popular editors (VS Code, etc.)
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Team collaboration features

---

**🎉 Ready to improve your technical writing? [Get started now!](#-quick-start-2-commands)**

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md) 