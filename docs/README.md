# Blackwall - GenAI Detection Tool

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
</p>

Blackwall is a command-line tool for detecting AI-generated content in text, images, and videos. It runs completely locally without requiring any external API calls

## 🚀 Features

- **Multi-format Support**: Detect AI-generated content in text files, images, and videos
- **Local Processing**: All analysis happens on your machine - no data is sent to external servers
- **Production Ready**: Built with enterprise-grade error handling, logging, and performance optimization
- **Flexible Output**: Multiple output formats including JSON, detailed reports, and minimal output
- **Batch Processing**: Analyze entire directories of files efficiently
- **Caching**: Smart caching system for faster repeated analyses
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📋 Requirements

- Python 3.10 or higher
- FFmpeg (for video processing)
- ~2GB of free disk space for models
- 4GB RAM minimum (8GB recommended)

## 🛠️ Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/bdpetersen13/Blackwall.git
cd blackwall

# Run the quick start script
./quickstart.sh