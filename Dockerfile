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

# Install ChromaDB in a virtual environment
RUN python3 -m venv /opt/chromadb-venv && \
    /opt/chromadb-venv/bin/pip install --no-cache-dir chromadb

# Test ChromaDB installation
RUN /opt/chromadb-venv/bin/python -c "import chromadb; print('ChromaDB import successful')" && \
    echo '#!/bin/bash' > /usr/local/bin/run-chromadb && \
    echo '/opt/chromadb-venv/bin/python -m chromadb.cli.cli run --host 0.0.0.0 --port 8000 --path \$1' >> /usr/local/bin/run-chromadb && \
    chmod +x /usr/local/bin/run-chromadb

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

# Expose ports
EXPOSE 11434 8000

# Use entrypoint script
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]