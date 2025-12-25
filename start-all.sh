#!/bin/bash

echo "🚀 Starting Complete Bot Stack..."

# Start LLM services first
./start-llm-services.sh

if [ $? -ne 0 ]; then
    echo "❌ Failed to start LLM services"
    exit 1
fi

# Wait for services to fully initialize
echo ""
echo "⏳ Waiting for services to initialize..."
sleep 3

# Start the Discord bot
echo ""
echo "🤖 Starting Discord bot..."
npm run dev
