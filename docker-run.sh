#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    echo "📋 Loading environment from .env file..."
    set -a
    source .env
    set +a
else
    echo "❌ .env file not found!"
    exit 1
fi

# Check required variables
if [ -z "$BOT_TOKEN" ]; then
    echo "❌ BOT_TOKEN not set in .env file!"
    exit 1
fi

echo "✅ BOT_TOKEN is set (length: ${#BOT_TOKEN})"
echo "✅ BOT_OWNERS: ${BOT_OWNERS:-(none)}"

# Build the image
echo ""
echo "🔨 Building Docker image..."
docker build -t krokenheimer-bot .

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "✅ Build successful!"
echo ""
echo "🚀 Starting container..."

# Stop and remove old container if exists
docker stop krokenheimer-bot 2>/dev/null
docker rm krokenheimer-bot 2>/dev/null

# Run the container
docker run -d \
    --name krokenheimer-bot \
    --restart unless-stopped \
    -p 3000:3000 \
    -e BOT_TOKEN="$BOT_TOKEN" \
    -e BOT_OWNERS="$BOT_OWNERS" \
    -e LLM_ENABLED="${LLM_ENABLED:-true}" \
    -e LLM_MIN_INTERVAL_MINUTES="${LLM_MIN_INTERVAL_MINUTES:-60}" \
    -e LLM_MAX_INTERVAL_MINUTES="${LLM_MAX_INTERVAL_MINUTES:-180}" \
    -e LLM_SCAN_INTERVAL_MINUTES="${LLM_SCAN_INTERVAL_MINUTES:-2}" \
    -e REMOTE_API_ENABLED="${REMOTE_API_ENABLED:-true}" \
    -e REMOTE_API_PORT="${REMOTE_API_PORT:-3000}" \
    -e REMOTE_API_TOKEN="${REMOTE_API_TOKEN:-kroken-secure-token-change-me-12345}" \
    -e REMOTE_API_AUTH="${REMOTE_API_AUTH:-true}" \
    -e REMOTE_TRAINING_MIN_MESSAGES="${REMOTE_TRAINING_MIN_MESSAGES:-1000}" \
    -e REMOTE_TRAINING_INTERVAL_HOURS="${REMOTE_TRAINING_INTERVAL_HOURS:-12}" \
    -e REMOTE_TRAINING_MAX_CLIENTS="${REMOTE_TRAINING_MAX_CLIENTS:-1}" \
    -e REMOTE_TRAINING_DEFAULT_EPOCHS="${REMOTE_TRAINING_DEFAULT_EPOCHS:-10}" \
    -e REMOTE_TRAINING_MODEL_NAME="${REMOTE_TRAINING_MODEL_NAME:-krokenheimer}" \
    -e REMOTE_TRAINING_RECOMMEND_GPU="${REMOTE_TRAINING_RECOMMEND_GPU:-true}" \
    -e REMOTE_TRAINING_PREFERRED_GPU="${REMOTE_TRAINING_PREFERRED_GPU:-rocm}" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/chroma_data:/app/chroma_data" \
    krokenheimer-bot

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container started successfully!"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs:      docker logs -f krokenheimer-bot"
    echo "   Stop:           docker stop krokenheimer-bot"
    echo "   Restart:        docker restart krokenheimer-bot"
    echo "   Shell access:   docker exec -it krokenheimer-bot bash"
    echo ""
    echo "Showing logs (Ctrl+C to exit, container keeps running)..."
    sleep 2
    docker logs -f krokenheimer-bot
else
    echo "❌ Failed to start container!"
    exit 1
fi
