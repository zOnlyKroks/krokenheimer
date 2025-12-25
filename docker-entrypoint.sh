#!/bin/bash
set -e

echo "🚀 Starting Krokenheimer Discord Bot with LLM features..."

# Create log directory for supervisor
mkdir -p /var/log/supervisor

# Start supervisor in background to get services running
echo "📡 Starting services..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "❌ Ollama failed to start after ${max_attempts} seconds"
        exit 1
    fi
    sleep 1
done
echo "✅ Ollama is running"

# Check if model exists, if not pull it
MODEL_NAME="${OLLAMA_MODEL:-llama3.2:3b}"
echo "🔍 Checking for model: $MODEL_NAME"

# List all models
echo "📋 Available models:"
ollama list

# Check if the specific model exists (match the full name or base name)
if ollama list | grep -q "$MODEL_NAME" || ollama list | grep -q "${MODEL_NAME%%:*}"; then
    echo "✅ Model $MODEL_NAME already exists, skipping download"
else
    echo "📥 Pulling model $MODEL_NAME (this may take a few minutes)..."
    ollama pull "$MODEL_NAME"
    echo "✅ Model downloaded"
fi

# Wait for ChromaDB to be ready
echo "⏳ Waiting for ChromaDB to start..."
attempt=0
while ! curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "❌ ChromaDB failed to start after ${max_attempts} seconds"
        exit 1
    fi
    sleep 1
done
echo "✅ ChromaDB is running"

echo ""
echo "✅ All services started successfully!"
echo ""
echo "📋 Service Status:"
echo "   Ollama:     http://localhost:11434"
echo "   ChromaDB:   http://localhost:8000"
echo "   Discord Bot: Starting..."
echo ""

# Keep the container running by tailing supervisor logs
tail -f /var/log/supervisor/supervisord.log
