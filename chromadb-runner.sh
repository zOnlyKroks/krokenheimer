#!/bin/bash
set -e

# Activate virtual environment
source /opt/chromadb-venv/bin/activate

# Run ChromaDB
exec python -m chromadb.cli.cli run --host 0.0.0.0 --port 8000 --path /app/chroma_data