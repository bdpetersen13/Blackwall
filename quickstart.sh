# Blackwall Quick Start Script

set -e

echo "🚀 Blackwall Quick Start"
echo "======================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install Blackwall
echo "🔨 Installing Blackwall..."
pip install -e .

# Download models
echo "🤖 Downloading models..."
python scripts/download_models.py

# Run tests
echo "🧪 Running basic tests..."
blackwall --help

echo ""
echo "✨ Installation complete!"
echo ""
echo "Usage examples:"
echo "  blackwall --file examples/sample.txt"
echo "  blackwall --file examples/image.jpg --output detailed"
echo "  blackwall --file examples/video.mp4 --verbose"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"