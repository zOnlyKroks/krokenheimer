#!/bin/bash
#
# Complete Training Reset (Lobotomy)
# Deletes all trained models and training state
#

set -e

echo "⚠️  WARNING: This will DELETE all training data!"
echo ""
echo "   - Trained models (./data/models/)"
echo "   - Training state (./data/training_state.json)"
echo "   - Training data exports (./data/training_data.jsonl)"
echo ""
echo "📊 Message history (./data/messages.db) will be PRESERVED"
echo ""
read -p "Are you sure? Type 'yes' to continue: " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "🧹 Cleaning up training data..."

# Create data directory if it doesn't exist
mkdir -p ./data

# Remove trained models directory entirely
if [ -d "./data/models" ]; then
    echo "   Deleting trained models directory..."
    rm -rf ./data/models
    echo "   ✓ Models deleted"
else
    echo "   ℹ️  No models directory found"
fi

# Remove training state
if [ -f "./data/training_state.json" ]; then
    echo "   Deleting training state..."
    rm -f ./data/training_state.json
    echo "   ✓ Training state deleted"
else
    echo "   ℹ️  No training state found"
fi

# Remove training data exports
if [ -f "./data/training_data.jsonl" ]; then
    echo "   Deleting training data export..."
    rm -f ./data/training_data.jsonl
    echo "   ✓ Training data export deleted"
else
    echo "   ℹ️  No training data export found"
fi

echo ""
echo "✅ Training reset complete!"
echo ""
echo "💡 Next steps:"
echo "   1. Stop container:    docker stop krokenheimer-bot"
echo "   2. Restart bot:       ./startup.sh"
echo "   3. Train fresh model: !llmtrain now"
echo ""
echo "📊 Your message history is preserved in ./data/messages.db"
echo "   The bot will learn from scratch using these messages."
