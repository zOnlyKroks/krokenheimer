#!/bin/bash

set -e

echo "🐍 Setting up Python environment for REAL training..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Install it with: apt-get install python3 python3-full python3-venv"
    exit 1
fi

echo "✅ Python 3 found"

if [ ! -d "venv/bin" ]; then
    echo "Creating virtual environment..."
    rm -rf venv
    python3 -m venv venv
fi

if [ ! -f "venv/bin/python" ]; then
    echo "❌ Failed to create Python virtual environment!"
    exit 1
fi

echo "✅ Virtual environment ready"

echo "Installing PyTorch and training dependencies..."
echo "This may take a few minutes..."

venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Python:"
venv/bin/python --version
echo ""
echo "To run training:"
echo "  1. Make sure you have messages scanned: !scan"
echo "  2. Run: !retrain"
echo ""
echo "Training will take 30-60+ minutes on CPU"
echo "Check console for progress"