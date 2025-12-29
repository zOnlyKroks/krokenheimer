FROM node:20-bookworm

# Install system dependencies including Rust and native module build tools
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
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install node-gyp globally for native module compilation
RUN npm install -g node-gyp

# Set Python path for node-gyp (fixes common build issues)
ENV PYTHON=/usr/bin/python3

# Set Node.js native module build environment variables
ENV npm_config_build_from_source=true
ENV npm_config_cache=/tmp/.npm

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

# ChromaDB runner - create script where supervisor expects it
RUN /opt/chromadb-venv/bin/python -c "import chromadb; print('ChromaDB import successful')" && \
    mkdir -p /app && \
    echo '#!/bin/bash' > /app/chromadb-runner.sh && \
    echo 'exec /opt/chromadb-venv/bin/chroma run --host 0.0.0.0 --port 8000 --path /app/chroma_data' >> /app/chromadb-runner.sh && \
    chmod +x /app/chromadb-runner.sh && \
    cp /app/chromadb-runner.sh /usr/local/bin/run-chromadb

# /app directory already created above

WORKDIR /app

# Copy package files for Node.js dependencies
COPY package*.json ./
RUN npm install

# Copy ONLY Rust module files first (for better caching)
COPY rust-ml/ ./rust-ml/

# Build Rust ML module FIRST (before TypeScript)
WORKDIR /app/rust-ml
RUN echo "🦀 Building Rust ML module with Neon..." && \
    npm run build && \
    echo "✅ Rust ML module build completed" && \
    echo "🔍 Checking for compiled native module..." && \
    ls -la target/release/ && \
    find . -name "*.node" -o -name "*krokenheimer*.so" -o -name "*krokenheimer*.dll" -o -name "*krokenheimer*.dylib" && \
    echo "🔗 Copying native module to expected location..." && \
    # Find and copy the native module to the expected location
    NATIVE_MODULE=$(find target/release -name "*.node" -o -name "*krokenheimer*.so" -o -name "*krokenheimer*.dll" -o -name "*krokenheimer*.dylib" | head -1) && \
    if [ -n "$NATIVE_MODULE" ]; then \
        cp "$NATIVE_MODULE" ./index.node && \
        echo "✅ Native module copied: $NATIVE_MODULE -> ./index.node" && \
        ls -la index.node; \
    else \
        echo "⚠️  No native module found - will run in fallback mode" && \
        find target/release -type f -name "*krokenheimer*"; \
    fi

# Now copy the rest of the source code (TypeScript, etc.)
WORKDIR /app
COPY src/ ./src/
COPY *.json *.js *.ts *.md ./
COPY docker-supervisord.conf wait-for-chromadb.sh ./

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

# Make scripts executable and copy to final locations
RUN chmod +x /app/wait-for-chromadb.sh && \
    cp /app/wait-for-chromadb.sh /usr/local/bin/wait-for-chromadb.sh && \
    cp /app/docker-supervisord.conf /etc/supervisor/supervisord.conf

# Verify Rust ML module integration
RUN echo "🧪 Testing Rust ML module..." && \
    cd /app && \
    node -e "try { const rustML = require('./rust-ml/index.node'); console.log('✅ Rust ML module loaded successfully'); console.log('📊 Available functions:', Object.keys(rustML)); } catch (error) { console.log('⚠️  Rust ML module not available:', error.message); console.log('🔄 Will run in fallback mode'); }" || echo "⚠️  Node.js test failed - will run in fallback mode"

EXPOSE 8000

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]