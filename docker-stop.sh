#!/bin/bash

echo "🛑 Stopping krokenheimer-bot container..."
docker stop krokenheimer-bot

echo "🗑️  Removing container..."
docker rm krokenheimer-bot

echo "✅ Container stopped and removed"
