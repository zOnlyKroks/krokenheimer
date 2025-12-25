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

# Only pull model if explicitly requested via environment variable
if [ "$OLLAMA_AUTO_PULL" = "true" ]; then
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
else
    echo "⏭️  Skipping automatic model download (set OLLAMA_AUTO_PULL=true to enable)"
    echo "📋 Available models:"
    ollama list || echo "   (No models found - pull manually with: docker-compose exec krokenheimer ollama pull llama3.2:3b)"
fi

# Wait for ChromaDB to be ready with proper health check
echo "⏳ Waiting for ChromaDB to be ready..."
attempt=0
max_chroma_attempts=60  # 2 minutes max

while true; do
    attempt=$((attempt + 1))

    # Try to connect to ChromaDB heartbeat endpoint
    if curl -s -f http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
        echo "✅ ChromaDB is ready and accepting connections"
        break
    fi

    if [ $attempt -ge $max_chroma_attempts ]; then
        echo "❌ ChromaDB failed to become ready after ${max_chroma_attempts} attempts"
        echo "Check ChromaDB logs: supervisorctl tail chromadb"
        exit 1
    fi

    if [ $((attempt % 10)) -eq 0 ]; then
        echo "   Still waiting for ChromaDB... (${attempt}/${max_chroma_attempts})"
    fi

    sleep 2
done

echo ""
echo "✅ All services ready!"
echo ""
echo "📋 Service Status:"
echo "   Ollama:     http://localhost:11434 ✅"
echo "   ChromaDB:   http://localhost:8000 ✅"
echo "   Discord Bot: Starting now..."
echo ""

# Now start the Discord bot (autostart was set to false)
supervisorctl start discord-bot

# Wait a moment for it to start
sleep 2

# Check if bot started successfully
if supervisorctl status discord-bot | grep -q "RUNNING"; then
    echo "✅ Discord bot started successfully!"
else
    echo "❌ Discord bot failed to start"
    supervisorctl status discord-bot
fi

echo ""
echo "📋 All Services Status:"
supervisorctl status

echo ""
echo "Container is ready. Tailing logs..."

# Keep the container running by tailing supervisor logs
tail -f /var/log/supervisor/supervisord.log
