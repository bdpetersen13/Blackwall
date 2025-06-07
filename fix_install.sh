#!/bin/bash
# Fix installation issues

echo "🔧 Fixing Blackwall installation..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Upgrade pip
pip install --upgrade pip

# Install updated requirements
pip install -r requirements.txt

# Reinstall Blackwall
pip install -e .

echo "✅ Installation fixed!"
echo ""
echo "Now run: python scripts/download_models.py"