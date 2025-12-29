FROM node:20-bookworm

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Rust toolchain (INSTALL ONCE)
# ------------------------------------------------------------
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable && \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME

# 🔑 Make cargo globally visible (fixes ENOENT)
RUN ln -s /usr/local/cargo/bin/cargo /usr/bin/cargo

RUN rustc --version && cargo --version

# ------------------------------------------------------------
# ChromaDB (Python venv)
# ------------------------------------------------------------
RUN python3 -m venv /opt/chromadb-venv && \
    /opt/chromadb-venv/bin/pip install --no-cache-dir chromadb

RUN /opt/chromadb-venv/bin/python - <<EOF
import chromadb
print("ChromaDB import successful")
EOF

RUN mkdir -p /app && \
    echo '#!/bin/bash' > /app/chromadb-runner.sh && \
    echo 'exec /opt/chromadb-venv/bin/chroma run --host 0.0.0.0 --port 8000 --path /app/chroma_data' >> /app/chromadb-runner.sh && \
    chmod +x /app/chromadb-runner.sh && \
    cp /app/chromadb-runner.sh /usr/local/bin/run-chromadb

# ------------------------------------------------------------
# Node dependencies (root)
# ------------------------------------------------------------
WORKDIR /app
COPY package*.json ./
RUN npm install

# ------------------------------------------------------------
# Rust ML module (Neon build happens HERE)
# ------------------------------------------------------------
COPY rust-ml/ ./rust-ml/
WORKDIR /app/rust-ml

RUN echo "🦀 Building Rust ML module..." && \
    npm run build && \
    echo "✅ Rust build completed" && \
    find . -name "*.node" -o -path "*target/release*" -name "*krokenheimer*"

# ------------------------------------------------------------
# App source
# ------------------------------------------------------------
WORKDIR /app
COPY src/ ./src/
COPY *.json *.js *.ts *.md ./
COPY docker-supervisord.conf wait-for-chromadb.sh ./

# ------------------------------------------------------------
# Build TypeScript
# ------------------------------------------------------------
RUN npm run build && test -f /app/dist/index.js

# ------------------------------------------------------------
# Production cleanup
# ------------------------------------------------------------
RUN npm prune --production

# ------------------------------------------------------------
# Runtime dirs
# ------------------------------------------------------------
RUN mkdir -p \
    /app/data \
    /app/chroma_data \
    /app/data/models \
    /app/data/checkpoints \
    /var/log/supervisor

RUN chmod +x /app/wait-for-chromadb.sh && \
    cp /app/wait-for-chromadb.sh /usr/local/bin/wait-for-chromadb.sh && \
    cp /app/docker-supervisord.conf /etc/supervisor/supervisord.conf

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------
RUN node - <<'EOF'
try {
  const rustML = require('./rust-ml');
  console.log('✅ Rust ML module loaded');
  console.log('📊 Exports:', Object.keys(rustML));
} catch (e) {
  console.error('❌ Failed to load Rust ML module');
  console.error(e);
  process.exit(1);
}
EOF

# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------
EXPOSE 8000
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
