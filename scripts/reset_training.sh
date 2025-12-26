#!/bin/bash
#
# Complete Training Reset (Lobotomy)
# Deletes all trained models and training state
#

set -e

echo "⚠️  WARNING: This will DELETE all training data!"
echo ""
echo "   - Trained models"
echo "   - Training state"
echo "   - Training data exports"
echo ""
read -p "Are you sure? Type 'yes' to continue: " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "🧹 Cleaning up training data..."

# Remove trained models
if [ -d "./data/models" ]; then
    echo "   Deleting trained models..."
    rm -rf ./data/models/*
    echo "   ✓ Models deleted"
fi

# Remove training state
if [ -f "./data/training_state.json" ]; then
    echo "   Deleting training state..."
    rm -f ./data/training_state.json
    echo "   ✓ Training state deleted"
fi

# Remove training data exports
if [ -f "./data/training_data.jsonl" ]; then
    echo "   Deleting training data export..."
    rm -f ./data/training_data.jsonl
    echo "   ✓ Training data export deleted"
fi

echo ""
echo "✅ Training reset complete!"
echo ""
echo "💡 Next steps:"
echo "   1. Restart bot: ./startup.sh"
echo "   2. Train fresh model: !llmtrain now"
echo ""
echo "📊 Your message history is preserved in ./data/messages.db"
echo "   The bot will learn from scratch using these messages."
