#!/bin/bash

echo "🛑 Stopping LLM Services..."

# Stop Ollama screen session
if screen -list | grep -q "ollama_server"; then
    screen -S ollama_server -X quit
    echo "✅ Ollama server stopped"
else
    echo "⚠️  Ollama screen session not found"
fi

# Stop ChromaDB screen session
if screen -list | grep -q "chroma_server"; then
    screen -S chroma_server -X quit
    echo "✅ ChromaDB server stopped"
else
    echo "⚠️  ChromaDB screen session not found"
fi

echo ""
echo "✅ All services stopped"
