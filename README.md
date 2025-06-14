# 🎯 Peer-Review Platform

**AI-Powered Technical Writing Assistant with Local Ollama Integration**

Transform your technical documentation with comprehensive style analysis, readability scoring, and AI-powered rewriting capabilities. Specifically designed for technical writers targeting 9th-11th grade readability standards.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Cross-Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-green.svg)](https://github.com/yourusername/peer-review-platform)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Complete Setup Guide (Beginner-Friendly)

### 📋 **Prerequisites**
- **Python 3.8+** ([Download here](https://www.python.org/downloads/))
- **Git** (optional, for cloning) ([Download here](https://git-scm.com/downloads))

### 🔧 **Step 1: Get the Code**

**Option A: Download ZIP**
1. Click the green "Code" button on GitHub
2. Select "Download ZIP"
3. Extract to your desired folder (e.g., `C:\Users\YourName\Desktop\Peer-Review-Platform`)

**Option B: Clone with Git**
```bash
git clone https://github.com/yourusername/peer-review-platform.git
cd peer-review-platform
```

### 🐍 **Step 2: Create Virtual Environment**

**🪟 Windows (Command Prompt/PowerShell):**
```batch
# Navigate to your project folder
cd C:\Users\YourName\Desktop\Peer-Review-Platform

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# You should see (venv) at the beginning of your prompt
```

**🐧 Linux/Ubuntu:**
```bash
# Navigate to your project folder
cd ~/Desktop/Peer-Review-Platform

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) at the beginning of your prompt
```

**🍎 macOS:**
```bash
# Navigate to your project folder
cd ~/Desktop/Peer-Review-Platform

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) at the beginning of your prompt
```

### ✅ **Step 3: Quick Setup (Automated)**

**With virtual environment activated:**
```bash
# Run the automated setup (handles everything!)
python quick_setup.py
```

This will:
- ✅ Upgrade pip and install all packages
- ✅ Download SpaCy language models
- ✅ Setup NLTK data
- ✅ Create necessary directories
- ✅ Fix deprecation warnings

### 🔇 **Step 4: Fix Terminal Warnings (Optional)**

If you see annoying warnings like `pkg_resources is deprecated`:

```bash
# Upgrade packages to eliminate warnings (one-time fix)
pip install --upgrade textstat>=0.7.3 transformers>=4.36.0 torch>=2.1.0
```

### 🎯 **Step 5: Start the Application**

**Clean startup (no warnings):**
```bash
python app.py
```

**Then visit:** [http://localhost:5000](http://localhost:5000) 🌐

---

## 🔄 **Daily Usage (After Initial Setup)**

### 🪟 **Windows Users:**
```batch
# 1. Open Command Prompt/PowerShell
# 2. Navigate to project folder
cd C:\Users\YourName\Desktop\Peer-Review-Platform

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Start application
python app.py
```

### 🐧🍎 **Linux/macOS Users:**
```bash
# 1. Open Terminal
# 2. Navigate to project folder
cd ~/Desktop/Peer-Review-Platform

# 3. Activate virtual environment
source venv/bin/activate

# 4. Start application
python app.py
```

### 🚨 **Important Notes:**
- **Always activate the virtual environment** before running the app
- Look for `(venv)` at the start of your command prompt
- If you see import errors, make sure venv is activated

---

## ⚙️ **Manual Setup (If Automated Fails)**

**With virtual environment activated:**

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip

# 2. Install all packages
pip install -r requirements.txt

# 3. Download language models
python -m spacy download en_core_web_sm

# 4. Setup NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Create directories
mkdir uploads logs cache
```

---

## ✨ Key Features

### 📊 **Comprehensive Technical Writing Analysis**
- **Grade Level Assessment** - Targets 9th-11th grade readability
- **Multiple Readability Scores** - Flesch Reading Ease, Gunning Fog, SMOG, Coleman-Liau, ARI
- **Style Issues Detection** - Passive voice, sentence length, wordiness, clarity
- **Technical Complexity Scoring** - Custom metrics for technical documentation

### 🧠 **AI-Powered Rewriting**
- **Local Ollama Integration** - Uses Gemma/Llama models for privacy-first processing
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

## 🎯 Usage Guide

### 🌐 **Web Interface**
1. **Start the application:** `python app.py` (with venv activated)
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

## 🧠 **AI Configuration**

### 🤖 **Using Local Ollama (Recommended)**

**Install Ollama:**
```bash
# Windows: Download from https://ollama.ai/
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
```

**Download AI Models:**
```bash
# Download Gemma model (small, fast)
ollama pull gemma:2b

# Or download Llama model (larger, more capable)
ollama pull llama3.1:8b

# Start Ollama service
ollama serve
```

**Configure in `src/config.py`:**
```python
USE_OLLAMA = True
OLLAMA_MODEL = "gemma:2b"  # or "llama3.1:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
```

---

## 🐛 Troubleshooting

### **Virtual Environment Issues**

**❌ "python: command not found"**
```bash
# Use python3 instead
python3 -m venv venv
```

**❌ "venv\Scripts\activate not found" (Windows)**
```batch
# Try PowerShell instead of Command Prompt
venv\Scripts\Activate.ps1
```

**❌ "Permission denied" (macOS/Linux)**
```bash
# Make sure you have write permissions
sudo chown -R $(whoami) ~/Desktop/Peer-Review-Platform
```

### **Package Installation Issues**

**❌ "SpaCy model not found"**
```bash
# With venv activated
python -m spacy download en_core_web_sm
```

**❌ "NLTK data missing"**
```bash
# With venv activated
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**❌ "Import errors"**
```bash
# Make sure virtual environment is activated
# Look for (venv) in your prompt
# If not there, activate it first
```

### **Deprecation Warnings**

**❌ "pkg_resources is deprecated"**
```bash
# Upgrade textstat
pip install --upgrade textstat>=0.7.3
```

**❌ "torch.utils._pytree._register_pytree_node is deprecated"**
```bash
# Upgrade transformers and torch
pip install --upgrade transformers>=4.36.0 torch>=2.1.0
```

### **AI/Ollama Issues**

**❌ "Ollama connection failed"**
```bash
# Start Ollama service
ollama serve

# Check if model is available
ollama list
```

**❌ "AI rewriting not working"**
- Check if Ollama is running: `ollama serve`
- Verify model is downloaded: `ollama list`
- Check `src/config.py` settings

---

## 📁 Project Structure

```
peer-review-platform/
├── 📄 app.py                  # Main Flask application
├── 📁 venv/                   # Virtual environment (you create this)
├── 📁 src/                    # Core modules
│   ├── 🧠 ai_rewriter.py      # AI-powered rewriting
│   ├── 📊 style_analyzer.py   # Style analysis engine
│   ├── 📁 document_processor.py # Multi-format text extraction
│   └── ⚙️ config.py           # Configuration management
├── 📁 templates/              # Web interface templates
│   ├── 🎨 base.html           # Main UI template
│   └── 🏠 index.html          # Homepage
├── 📁 uploads/                # File upload directory (auto-created)
├── 📁 logs/                   # Application logs (auto-created)
├── 📋 requirements.txt        # Python dependencies
├── 🚀 quick_setup.py          # Quick setup script
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

## 🎓 **Complete Beginner's Walkthrough**

### **Never used Python before? Start here:**

1. **Install Python:**
   - Go to [python.org](https://python.org/downloads)
   - Download Python 3.8 or newer
   - ✅ Check "Add Python to PATH" during installation

2. **Download this project:**
   - Click green "Code" button → "Download ZIP"
   - Extract to `Desktop/Peer-Review-Platform`

3. **Open terminal/command prompt:**
   - **Windows:** Press `Win + R`, type `cmd`, press Enter
   - **macOS:** Press `Cmd + Space`, type `terminal`, press Enter
   - **Linux:** Press `Ctrl + Alt + T`

4. **Navigate to project:**
   ```bash
   cd Desktop/Peer-Review-Platform
   ```

5. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

6. **Activate virtual environment:**
   - **Windows:** `venv\Scripts\activate`
   - **macOS/Linux:** `source venv/bin/activate`

7. **Install everything:**
   ```bash
   python quick_setup.py
   ```

8. **Start the application:**
   ```bash
   python app.py
   ```

9. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

**🎉 That's it! You're ready to improve your writing with AI!**

---

## 📊 Performance & Requirements

### **System Requirements**
- **RAM**: 4GB minimum, 8GB recommended (with Ollama)
- **Storage**: 2GB for dependencies + models
- **CPU**: Any modern processor (AI benefits from multi-core)
- **Python**: 3.8 or newer

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

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create your development environment:**
   ```bash
   git clone https://github.com/yourusername/peer-review-platform.git
   cd peer-review-platform
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Create a feature branch:** `git checkout -b feature/amazing-feature`
4. **Make your changes** and add tests
5. **Run the test suite:** `pytest`
6. **Format your code:** `black . && isort .`
7. **Commit your changes:** `git commit -m 'Add amazing feature'`
8. **Push to the branch:** `git push origin feature/amazing-feature`
9. **Open a Pull Request**

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

**🎉 Ready to improve your technical writing? Follow the setup guide above and get started!**

**Need help?** Create an issue on GitHub or check our troubleshooting section above. 