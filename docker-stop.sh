#!/bin/bash

echo "🛑 Stopping krokenheimer-bot container..."
if docker stop krokenheimer-bot 2>/dev/null; then
    echo "✅ Container stopped successfully"
else
    echo "ℹ️  Container was not running"
fi

echo "🗑️  Removing container..."
if docker rm krokenheimer-bot 2>/dev/null; then
    echo "✅ Container removed successfully"
else
    echo "ℹ️  Container was already removed"
fi

echo "✅ Cleanup complete"
