#!/bin/bash
set -e

echo "🚀 Starting Krokenheimer Discord Bot with LLM features..."

# Create log directory for supervisor
mkdir -p /var/log/supervisor

# Start supervisor
echo "📡 Starting services..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &

# Function to check if a service is running
wait_for_service() {
    local name=$1
    local command=$2
    local max_attempts=${3:-60}
    local attempt=0

    echo "⏳ Waiting for $name to start..."

    while [ $attempt -lt "$max_attempts" ]; do
        attempt=$((attempt + 1))

        if eval "$command" >/dev/null 2>&1; then
            echo "✅ $name is ready!"
            return 0
        fi

        if [ $((attempt % 10)) -eq 0 ]; then
            echo "   Still waiting for $name... (attempt $attempt/$max_attempts)"
        fi

        sleep 2
    done

    echo "❌ $name failed to start after $max_attempts attempts"
    return 1
}

# Wait for supervisord
wait_for_service "Supervisord" "[ -S /var/run/supervisor.sock ]" 30

# Check service status
echo "📋 Initial service status:"
supervisorctl status

# Wait for Ollama
wait_for_service "Ollama" "curl -s http://localhost:11434/api/tags > /dev/null"

# Optional model pull
if [ "$OLLAMA_AUTO_PULL" = "true" ]; then
    MODEL_NAME="${OLLAMA_MODEL:-llama3.2:3b}"
    echo "🔍 Checking for model: $MODEL_NAME"

    if ! ollama list | grep -q "$MODEL_NAME"; then
        echo "📥 Pulling model $MODEL_NAME..."
        ollama pull "$MODEL_NAME"
        echo "✅ Model downloaded"
    else
        echo "✅ Model $MODEL_NAME already exists"
    fi
else
    echo "⏭️  Skipping automatic model download"
    echo "📋 Available models:"
    ollama list || echo "   (No models found)"
fi

# Wait for ChromaDB (simpler check)
wait_for_service "ChromaDB" "curl -s http://localhost:8000/api/v1/heartbeat > /dev/null"

# Start Discord bot
echo "🤖 Starting Discord bot..."
supervisorctl start discord-bot

# Wait a moment and check status
sleep 3

echo ""
echo "📋 Final Service Status:"
supervisorctl status

echo ""
echo "✅ All services are running!"
echo "   Ollama:     http://localhost:11434"
echo "   ChromaDB:   http://localhost:8000"
echo "   Discord Bot: Running"

# Keep container running
echo ""
echo "📋 Tailing supervisor logs..."
tail -f /var/log/supervisor/supervisord.log