# 🛡️ Blackwall - AI Content Detection Tool

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <strong>Detect AI-generated content locally, privately, and efficiently</strong>
</p>

---

    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                                   ║
    ║                                                                                                                                   ║
    ║   ░▒▓███████▓▒░░▒▓█▓▒░       ░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓███████▓▒░░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░      ░▒▓███████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░          ║
    ║   ░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓█████████████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓████████▓▒░   ║
    ║                                                                                                                                   ║
    ║                                                                                                                                   ║
    ║                                               GenAI Detection Tool v0.1.0                                                         ║
    ║                                                                                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

## 🎯 What is Blackwall?

Blackwall is a command-line tool that detects AI-generated content in text files, images, and videos

### Why "Blackwall"?

GenAI has a lot of potential, but that means for each good opportunity it provides, there are negative ones that people will use it for, malicious intent or not. From fake news, videos, or images portraying one thing that’s not truthful. GenAI has the potential to be physically, emotionally, and mentally harmful to people.
The line between useful AI and harmful manipulation is an extremely grey line. Deepfakes can erode trust in public figures. Synthetic news articles can shift public opinion or tank a company’s stock before the truth catches up. Even without ill intent, GenAI models can amplify bias, hallucinate facts, or be used in ways the creators never anticipated

That doesn’t mean we should avoid GenAI—it means we need regulations and guardrails in place, responsible deployment and ownership from these GenAI companies, and widespread education. How we decide to align and to GenAI is how we will determine whether it becomes a force multiplier or a societal hazard

Blackwall serves as your defense against the widespread AI-generated content found on social media today, helping you verify authenticity in an age where distinguishing between human and AI creation is exponentially getting more difficult by the day

## 🚀 Key Features

- **🔒 100% Local Processing**: All analysis happens on your machine
- **📄 Multi-Format Support**: Detect AI content in text (.txt, .docx, .pdf), images (.jpg, .png), and videos (.mp4, .mov)
- **⚡ Fast Analysis**: Optimized for performance with smart caching and efficient processing
- **🎯 Confidence Scoring**: Get probability scores and confidence levels for each detection
- **🛠️ CLI-First Design**: Built for developers, researchers, and power users
- **📊 Detailed Reports**: Choose from multiple output formats (plain, JSON, detailed, minimal)
- **🔄 Batch Processing**: Analyze entire directories of files efficiently

## 🤔 Why I Created Blackwall

As AI-generated content becomes increasingly complex and prevalent, I recognized the critical need for a tool that could help identify such content while respecting user privacy. Existing solutions often require uploading files to cloud services, which isn't suitable for sensitive documents, proprietary images, or confidential videos

I built Blackwall to:
- **Protect Privacy**: Keep sensitive content analysis completely local
- **Empower Verification**: Help educators, journalists, and researchers verify content authenticity
- **Support Open Source**: Provide a foundation for the community to build upon
- **Encourage Transparency**: Make AI detection accessible to everyone

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- 4GB RAM minimum (8GB recommended)
- ~2GB free disk space for models
- FFmpeg (for video processing)

### Quick Installation

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.10 ffmpeg libmagic

# Clone the repository
git clone https://github.com/bdpetersen13/Blackwall.git
cd blackwall

# Run the quick start script
./reinstall_script.sh
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/bdpetersen13/Blackwall.git
cd blackwall

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Blackwall
pip install -e .

# Download models
python scripts/download_models.py
```

## 🎮 Usage

### Basic Usage

```bash
# Analyze a text file
blackwall --file document.txt

# Analyze an image
blackwall --file image.jpg

# Analyze a video
blackwall --file video.mp4
```

### Advanced Options

```bash
# Get detailed analysis with metadata
blackwall --file document.pdf --output detailed

# Get JSON output for integration
blackwall --file image.png --output json

# Set custom detection threshold
blackwall --file video.mp4 --threshold 0.7

# Quiet mode with minimal output
blackwall --file report.docx --quiet --output minimal
```

### Batch Processing

```bash
# Analyze all text files in a directory
blackwall-batch --directory ./documents --pattern "*.txt"

# Recursive analysis with CSV output
blackwall-batch --directory ./media --recursive --output csv

# Process images with custom threshold
blackwall-batch --directory ./images --pattern "*.jpg" --threshold 0.8
```

## 📊 Understanding Results

Blackwall provides probability scores indicating the likelihood of AI generation:

- **0-30%**: Likely human-created (GREEN)
- **30-70%**: Uncertain, further review recommended (YELLOW)  
- **70-100%**: Likely AI-generated (RED)

### Output Example

```
═══ Blackwall Detection Result ═══

📄 File: essay.docx
🔍 Type: TEXT
📊 Probability: 82.3%
🎯 Status: AI-GENERATED
⚡ Confidence: HIGH
⏱️  Processing Time: 1.24s

📝 Notes:
   • High probability of AI-generation detected
   • Found 4 common AI phrases
   • Low vocabulary diversity detected
```

## 🛠️ Architecture

Blackwall uses a modular architecture with specialized detectors:

- **Text Detector**: Transformer-based models analyzing linguistic patterns
- **Image Detector**: CNN models detecting GAN/diffusion model artifacts
- **Video Detector**: Frame-by-frame analysis with temporal consistency checks

## 🗺️ Roadmap

### Phase 1: MVP (Current)
- ✅ Basic detection for text, image, and video using prebuilt ML models
- ✅ Local processing pipeline
- ✅ CLI interface
- ✅ Multiple output formats

### Phase 2: Enhanced Accuracy
- 🔄 Experiment with different models and Fine-tune models for better accuracy
- 🔄 Support for more file formats
- 🔄 Detection of specific AI models (GPT-4, DALL-E, Midjourney, etc.)
- 🔄 Confidence calibration improvements

### Phase 3: Advanced Features
- 📱 GUI application for non-technical users
- 🔌 Plugin system for custom detectors
- 🌍 Multi-language support
- 📈 Detailed analysis reports with explanations
- 🔄 Real-time monitoring capabilities

### Phase 4: Ecosystem
- 🌐 Web interface (still local processing)
- 🔗 Integration libraries (Python, Node.js, Go)
- 📚 Model zoo with specialized detectors

## 🤝 Contributing

Blackwall is open source and welcomes contributions! Here's how you can help:

- **🐛 Report Bugs**: Open an issue with detailed information
- **💡 Suggest Features**: Share your ideas in the discussions
- **🔧 Submit PRs**: Check our contributing guidelines
- **📖 Improve Docs**: Help make our documentation better
- **🧪 Add Tests**: Increase our test coverage
- **🎨 Create Models**: Train and share detection models

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
make lint

# Format code
make format
```

## ⚠️ Limitations & Disclaimers

- **Accuracy**: Current models are proof-of-concept. False positives/negatives are expected
- **Evolution**: As AI models improve, detection becomes more challenging
- **Scope**: Designed for common AI generation methods; novel techniques may evade detection
- **Not Legal Advice**: Results should not be used as sole evidence in legal contexts


## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/bdpetersen13/Blackwall/issues)
- **Email**: petersen.brandon@sudomail.com

---

<p align="center">
  Made with ❤️ by Brandon Petersen
</p>

<p align="center">
  Some of the content in this repository has been made with GenAI, such as this README.md file and some of the code
</p>

<p align="center">
  <i>Protecting authenticity in the age of AI</i>
</p>
