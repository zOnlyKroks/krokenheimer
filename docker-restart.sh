#!/bin/bash

echo "🔄 Restarting krokenheimer-bot..."

# Stop and remove existing container
docker stop krokenheimer-bot 2>/dev/null
docker rm krokenheimer-bot 2>/dev/null

# Run the container again
./docker-run.sh
