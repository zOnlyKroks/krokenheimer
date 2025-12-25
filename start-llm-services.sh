#!/bin/bash

echo "🚀 Starting LLM Services..."

# Create data directory if it doesn't exist
mkdir -p ./data
mkdir -p ./chroma_data

# Check if screen is installed
if ! command -v screen &> /dev/null; then
    echo "❌ screen is not installed. Please install it:"
    echo "   macOS: brew install screen"
    echo "   Linux: sudo apt install screen"
    exit 1
fi

# Check if ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ ollama is not installed. Please install it:"
    echo "   https://ollama.com/download"
    exit 1
fi

# Check if chroma is available in any form
CHROMA_FOUND=false

if command -v chroma &> /dev/null; then
    CHROMA_FOUND=true
elif [ -f ~/.local/bin/chroma ]; then
    CHROMA_FOUND=true
elif [ -f ~/.chromadb-venv/bin/chroma ]; then
    CHROMA_FOUND=true
elif python3 -c "import chromadb" &> /dev/null 2>&1; then
    CHROMA_FOUND=true
fi

if [ "$CHROMA_FOUND" = false ]; then
    echo "❌ chromadb is not installed."
    echo ""
    echo "Installation options:"
    echo ""
    echo "Option 1 - Using pipx (recommended):"
    echo "   sudo apt install pipx"
    echo "   pipx install chromadb"
    echo "   pipx ensurepath"
    echo ""
    echo "Option 2 - Using virtual environment:"
    echo "   python3 -m venv ~/.chromadb-venv"
    echo "   ~/.chromadb-venv/bin/pip install chromadb"
    echo ""
    echo "After installation, run this script again."
    echo ""
    echo "Run ./check-chroma.sh for detailed diagnostics."
    exit 1
fi

# Start Ollama in a screen session
echo "📡 Starting Ollama server..."
if screen -list | grep -q "ollama_server"; then
    echo "⚠️  Ollama screen session already exists"
else
    screen -dmS ollama_server bash -c "ollama serve; exec bash"
    echo "✅ Ollama started in screen session 'ollama_server'"
fi

# Wait a moment for Ollama to start
sleep 2

# Check if the model exists
echo "🔍 Checking for required model..."
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "📥 Downloading llama3.2:3b model (this may take a few minutes)..."
    ollama pull llama3.2:3b
fi

# Start ChromaDB in a screen session
echo "📊 Starting ChromaDB server..."
if screen -list | grep -q "chroma_server"; then
    echo "⚠️  ChromaDB screen session already exists"
else
    # Detect how chromadb is installed and use the appropriate command
    CHROMA_CMD=""

    if command -v chroma &> /dev/null; then
        # Found in PATH
        CHROMA_CMD="chroma"
    elif [ -f ~/.local/bin/chroma ]; then
        # Installed via pipx (common location)
        CHROMA_CMD="~/.local/bin/chroma"
    elif [ -f ~/.chromadb-venv/bin/chroma ]; then
        # Installed in venv at standard location
        CHROMA_CMD="~/.chromadb-venv/bin/chroma"
    elif python3 -c "import chromadb" &> /dev/null 2>&1; then
        # Available in Python but no CLI command, use python -m
        CHROMA_CMD="python3 -m chromadb.cli"
    fi

    if [ -z "$CHROMA_CMD" ]; then
        echo "❌ Could not find chromadb. Run ./check-chroma.sh for diagnostics"
        exit 1
    fi

    screen -dmS chroma_server bash -c "$CHROMA_CMD run --path ./chroma_data; exec bash"
    echo "✅ ChromaDB started in screen session 'chroma_server'"
    echo "   Using: $CHROMA_CMD"
fi

echo ""
echo "✅ All services started successfully!"
echo ""
echo "📋 Useful commands:"
echo "   View Ollama logs:    screen -r ollama_server"
echo "   View ChromaDB logs:  screen -r chroma_server"
echo "   Detach from screen:  Ctrl+A, then D"
echo "   List all screens:    screen -list"
echo "   Stop services:       ./stop-llm-services.sh"
echo ""
echo "🤖 You can now start the bot with: npm run dev"
