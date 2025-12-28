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

# Install ChromaDB
RUN python3 -m venv /opt/chromadb-venv && \
    /opt/chromadb-venv/bin/pip install --no-cache-dir chromadb

# ChromaDB runner
RUN /opt/chromadb-venv/bin/python -c "import chromadb; print('ChromaDB import successful')" && \
    echo '#!/bin/bash' > /usr/local/bin/run-chromadb && \
    echo 'exec /opt/chromadb-venv/bin/chroma run --host 0.0.0.0 --port 8000 --path "$1"' >> /usr/local/bin/run-chromadb && \
    chmod +x /usr/local/bin/run-chromadb

# Create /app directory BEFORE trying to symlink to it
RUN mkdir -p /app

# Install CPU-compatible training environment for from-scratch training
RUN python3 -m venv /opt/training-venv && \
    /opt/training-venv/bin/pip install --upgrade pip && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    "numpy<2" && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    transformers==4.36.0 \
    tokenizers==0.15.0 \
    datasets==2.16.0 \
    accelerate==0.25.0 && \
    /opt/training-venv/bin/pip install --no-cache-dir \
    scipy \
    protobuf \
    ninja \
    matplotlib \
    packaging \
    psutil

# Verify installation works (with better error reporting)
RUN echo "Testing PyTorch installation..." && \
    /opt/training-venv/bin/python -c "import torch; print(f'✓ PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" && \
    echo "Testing NumPy installation..." && \
    /opt/training-venv/bin/python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" && \
    echo "Testing Transformers installation..." && \
    /opt/training-venv/bin/python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" && \
    echo "Testing Tokenizers installation..." && \
    /opt/training-venv/bin/python -c "import tokenizers; print(f'✓ Tokenizers {tokenizers.__version__}')" && \
    echo "Testing psutil installation..." && \
    /opt/training-venv/bin/python -c "import psutil; print(f'✓ psutil {psutil.__version__}')" && \
    echo "✅ 100% from-scratch training environment ready (custom tokenizer + random weights)"

WORKDIR /app

# NOW this will work since /app exists
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

EXPOSE 8000

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
