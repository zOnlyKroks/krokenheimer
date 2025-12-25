#!/bin/bash
# Setup script for Unsloth fine-tuning environment

echo "🚀 Setting up Unsloth training environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📋 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Unsloth (CPU-friendly version)
echo "📥 Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional dependencies
echo "📥 Installing training dependencies..."
pip install \
    torch \
    transformers \
    trl \
    datasets \
    bitsandbytes \
    accelerate \
    peft

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Activate venv: source venv/bin/activate"
echo "   2. Run training: python3 scripts/train_unsloth.py <base_model> <data.jsonl> <output_name>"
echo "   3. Or let the bot auto-train when threshold is reached!"
echo ""
echo "⚠️  Note: Training on CPU will take 3-7 days. This is normal."
echo "💡 The bot continues working normally during training."
