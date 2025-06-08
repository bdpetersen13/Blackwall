#!/bin/bash
# Complete fix for Blackwall on macOS M1

echo "🍎 Fixing Blackwall for macOS M1..."

# 1. Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# 2. Install libmagic via Homebrew
echo "📦 Installing libmagic via Homebrew..."
brew install libmagic

# 3. Activate virtual environment
if [ -d "venv" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Error: venv directory not found."
    exit 1
fi

# 4. Clean up previous installation attempts
echo "🧹 Cleaning up..."
pip uninstall -y blackwall 2>/dev/null || true
rm -rf build/ dist/ *.egg-info src/*.egg-info

# 5. Update pip and install build tools
echo "⬆️  Updating pip and build tools..."
pip install --upgrade pip setuptools wheel

# 6. Install dependencies separately to handle potential issues
echo "📚 Installing dependencies..."

# First install the problematic packages
pip install python-magic==0.4.27
pip install pydantic==2.5.0
pip install pydantic-settings==2.1.0

# Then install the rest
pip install -r requirements.txt --no-deps 2>/dev/null || true

# 7. Install blackwall in development mode
echo "🔨 Installing blackwall..."
pip install -e .

# 8. If installation failed, create manual wrapper
if ! command -v blackwall &> /dev/null; then
    echo "⚠️  Standard installation didn't work. Creating manual wrapper..."
    
    # Create wrapper script
    cat > "$VIRTUAL_ENV/bin/blackwall" << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
from blackwall.cli import main

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
EOF
    
    chmod +x "$VIRTUAL_ENV/bin/blackwall"
    
    # Create batch wrapper
    cat > "$VIRTUAL_ENV/bin/blackwall-batch" << 'EOF'
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
from blackwall.cli import batch_process

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(batch_process())
EOF
    
    chmod +x "$VIRTUAL_ENV/bin/blackwall-batch"
fi

# 9. Verify installation
echo ""
echo "🧪 Verifying installation..."

# Check Python path
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check if blackwall is available
if command -v blackwall &> /dev/null; then
    echo "✅ blackwall command found at: $(which blackwall)"
    echo ""
    echo "Testing blackwall..."
    blackwall --version || blackwall --help
    echo ""
    echo "✨ Success! You can now use:"
    echo "   blackwall --file yourfile.txt"
    echo "   blackwall --help"
else
    echo "❌ blackwall command still not found"
    echo ""
    echo "Try running:"
    echo "   python -m blackwall --help"
    echo ""
    echo "Or add this to your shell profile:"
    echo "   alias blackwall='python -m blackwall'"
fi

# 10. Final check - list what's in the bin directory
echo ""
echo "📁 Contents of venv/bin:"
ls -la "$VIRTUAL_ENV/bin" | grep -E "(blackwall|python)"