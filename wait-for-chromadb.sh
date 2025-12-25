#!/bin/bash
# /usr/local/bin/wait-for-chromadb.sh

set -e

# Host and port
HOST="localhost"
PORT=8000

echo "Waiting for ChromaDB to be ready at $HOST:$PORT..."

# Loop until connection succeeds
until nc -z $HOST $PORT; do
  sleep 1
done

echo "ChromaDB is up. Starting Discord bot..."
exec node /app/dist/index.js
