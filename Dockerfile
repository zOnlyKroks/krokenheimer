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
    netcat-openbsd \
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

# Install CPU-compatible training environment (no Unsloth - requires GPU)
# Using standard HuggingFace Transformers + PEFT for LoRA training
# Install CPU-compatible training environment (no Unsloth - requires GPU)
# Using standard HuggingFace Transformers + PEFT for LoRA training
RUN python3 -m venv /opt/training-venv && \
    /opt/training-venv/bin/pip install --upgrade pip && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    torchvision==0.15.2 && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    transformers \
    trl \
    datasets \
    accelerate \
    peft && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    scipy \
    sentencepiece \
    protobuf \
    ninja \
    packaging

# Verify installation works
RUN /opt/training-venv/bin/python -c "import torch; print(f'Torch version: {torch.__version__}')" && \
    /opt/training-venv/bin/python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" && \
    /opt/training-venv/bin/python -c "import peft; print(f'PEFT version: {peft.__version__}')" && \
    echo "✅ CPU training environment ready" \

WORKDIR /app

# Symlink training venv to app directory for bot access
RUN ln -s /opt/training-venv /app/venv

COPY package*.json ./
RUN npm install

COPY . .

# Show TypeScript build output
RUN echo "🔨 Building TypeScript..." && \
    npm run build && \
    echo "📦 Build complete. Checking output..." && \
    ls -la /app/dist && \
    test -f /app/dist/index.js && echo "✅ index.js exists" || echo "❌ index.js missing!"

RUN npm prune --production

RUN mkdir -p /app/data /app/chroma_data /app/data/models /app/data/checkpoints /var/log/supervisor

# Make training scripts executable
RUN chmod +x /app/scripts/*.py /app/scripts/*.sh || true

COPY wait-for-chromadb.sh /usr/local/bin/wait-for-chromadb.sh
RUN chmod +x /usr/local/bin/wait-for-chromadb.sh

COPY docker-supervisord.conf /etc/supervisor/supervisord.conf

EXPOSE 11434 8000

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
