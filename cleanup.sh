#!/bin/bash

echo "🧹 Cleaning up Krokenheimer bot data..."

# Remove message database
if [ -f "./data/messages.db" ]; then
    rm -f "./data/messages.db"
    echo "✅ Deleted message database"
else
    echo "⚠️  No message database found"
fi

# Get custom model name from env or use default
CUSTOM_MODEL="${LLM_CUSTOM_MODEL:-discord-bot-custom}"

# Remove Ollama custom model
echo "🗑️  Attempting to remove Ollama model: $CUSTOM_MODEL"
if ollama list | grep -q "$CUSTOM_MODEL"; then
    ollama rm "$CUSTOM_MODEL"
    echo "✅ Deleted Ollama model: $CUSTOM_MODEL"
else
    echo "⚠️  Ollama model not found: $CUSTOM_MODEL"
fi

echo ""
echo "✨ Cleanup complete! Restart the bot to:"
echo "   1. Create fresh database"
echo "   2. Scan all historical messages"
echo "   3. Train with !retrain command"