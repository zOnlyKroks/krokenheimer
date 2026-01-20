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
