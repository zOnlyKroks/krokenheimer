#!/bin/bash

echo "🔍 Checking ChromaDB installation..."
echo ""

# Check if chroma command is in PATH
echo "1. Checking if 'chroma' is in PATH:"
if command -v chroma &> /dev/null; then
    echo "   ✅ Found: $(which chroma)"
    chroma --version
else
    echo "   ❌ Not found in PATH"
fi

echo ""

# Check if chromadb Python module is importable
echo "2. Checking if chromadb Python module is available:"
if python3 -c "import chromadb" &> /dev/null; then
    echo "   ✅ chromadb module can be imported"
    python3 -c "import chromadb; print(f'   Version: {chromadb.__version__}')"
else
    echo "   ❌ chromadb module not found"
fi

echo ""

# Check pipx installations
echo "3. Checking pipx installations:"
if command -v pipx &> /dev/null; then
    echo "   pipx is installed"
    pipx list
else
    echo "   ❌ pipx not found"
fi

echo ""

# Check common locations
echo "4. Checking common installation locations:"
if [ -f ~/.local/bin/chroma ]; then
    echo "   ✅ Found: ~/.local/bin/chroma"
fi

if [ -f ~/.chromadb-venv/bin/chroma ]; then
    echo "   ✅ Found: ~/.chromadb-venv/bin/chroma"
fi

echo ""

# Check PATH
echo "5. Current PATH:"
echo "   $PATH" | tr ':' '\n' | grep -E "local/bin|pipx"

echo ""

# Suggest fixes
echo "📋 Suggested fixes:"
echo ""
if [ -f ~/.local/bin/chroma ]; then
    echo "   ChromaDB is installed at ~/.local/bin/chroma"
    echo "   Add to PATH with:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "   Make it permanent by adding to ~/.bashrc:"
    echo "   echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    echo "   source ~/.bashrc"
fi
