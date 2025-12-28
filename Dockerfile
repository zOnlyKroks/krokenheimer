FROM node:20-bookworm

# Install system dependencies including Rust
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
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# Verify Rust installation
RUN rustc --version && cargo --version

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

WORKDIR /app

# Copy package files for Node.js dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Build Rust ML module FIRST (before TypeScript)
WORKDIR /app/rust-ml
RUN echo "🦀 Building Rust ML module..." && \
    cargo build --release && \
    echo "✅ Rust ML module compiled successfully" && \
    ls -la target/release/ | grep -E "\.(so|dylib|dll)$" || echo "⚠️  No shared library found"

# Build TypeScript
WORKDIR /app
RUN echo "🔨 Building TypeScript..." && \
    npm run build && \
    echo "📦 Build complete. Checking output..." && \
    ls -la /app/dist && \
    test -f /app/dist/index.js && echo "✅ index.js exists" || echo "❌ index.js missing!"

# Clean up dev dependencies
RUN npm prune --production

# Create necessary directories
RUN mkdir -p /app/data /app/chroma_data /app/data/models /app/data/checkpoints /var/log/supervisor

# Make scripts executable
RUN chmod +x /app/scripts/*.py /app/scripts/*.sh || true

# Copy configuration files
COPY wait-for-chromadb.sh /usr/local/bin/wait-for-chromadb.sh
RUN chmod +x /usr/local/bin/wait-for-chromadb.sh
COPY docker-supervisord.conf /etc/supervisor/supervisord.conf

# Verify Rust ML module integration
RUN echo "🧪 Testing Rust ML module..." && \
    cd /app && \
    node -e "try { const rustML = require('./rust-ml/index.node'); console.log('✅ Rust ML module loaded successfully'); console.log('📊 Available functions:', Object.keys(rustML)); } catch (error) { console.log('⚠️  Rust ML module not available:', error.message); console.log('🔄 Will run in fallback mode'); }" || echo "⚠️  Node.js test failed - will run in fallback mode"

EXPOSE 8000

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]