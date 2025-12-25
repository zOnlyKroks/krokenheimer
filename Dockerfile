FROM node:20-bookworm

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

# Install ChromaDB
RUN python3 -m venv /opt/chromadb-venv && \
    /opt/chromadb-venv/bin/pip install --no-cache-dir chromadb

# ChromaDB runner
RUN /opt/chromadb-venv/bin/python -c "import chromadb; print('ChromaDB import successful')" && \
    echo '#!/bin/bash' > /usr/local/bin/run-chromadb && \
    echo 'exec /opt/chromadb-venv/bin/chroma run --host 0.0.0.0 --port 8000 --path "$1"' >> /usr/local/bin/run-chromadb && \
    chmod +x /usr/local/bin/run-chromadb

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build
RUN npm prune --production

RUN mkdir -p /app/data /app/chroma_data /var/log/supervisor

COPY docker-supervisord.conf /etc/supervisor/supervisord.conf

EXPOSE 11434 8000

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
