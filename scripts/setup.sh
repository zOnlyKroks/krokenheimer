#!/bin/bash

echo "🐍 Setting up Python environment for REAL training..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "Install it with: apt-get install python3 python3-pip python3-venv"
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing PyTorch and training dependencies..."
echo "This may take a few minutes..."

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run training:"
echo "  1. Make sure you have messages scanned: !scan"
echo "  2. Run: !retrain"
echo ""
echo "Training will take 30-60+ minutes on CPU"
echo "Check console for progress"