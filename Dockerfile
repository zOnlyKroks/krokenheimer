FROM node:20-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    make \
    g++ \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    pkg-config \
    curl \
    sqlite3 \
    supervisor \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install ChromaDB in a virtual environment with explicit version
RUN python3 -m venv /opt/chromadb-venv && \
    /opt/chromadb-venv/bin/pip install --no-cache-dir chromadb==0.4.22 && \
    /opt/chromadb-venv/bin/pip install --no-cache-dir 'pydantic<2.0.0' && \
    /opt/chromadb-venv/bin/pip install --no-cache-dir sentence-transformers

# Create app directory
WORKDIR /app

# Copy package files and install ALL dependencies (needed for build)
COPY package*.json ./
RUN npm install

# Copy application code
COPY . .

# Build TypeScript
RUN npm run build

# Remove dev dependencies to reduce image size
RUN npm prune --production

# Create data directories
RUN mkdir -p /app/data /app/chroma_data

# Copy supervisor config
COPY docker-supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy Docker entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create a simple test script for ChromaDB
RUN echo '#!/bin/bash\n/opt/chromadb-venv/bin/python3 -c "import chromadb; print(\"ChromaDB imported successfully\")"' > /test_chromadb.sh && \
    chmod +x /test_chromadb.sh

# Expose ports
EXPOSE 11434 8000

# Use entrypoint script
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]