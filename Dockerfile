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

# Install Ollama for LLM inference
RUN curl -fsSL https://ollama.ai/install.sh | sh

WORKDIR /app

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

RUN mkdir -p /app/data /app/data/models /app/data/checkpoints /var/log/supervisor

# Make training scripts executable
RUN chmod +x /app/scripts/*.py /app/scripts/*.sh || true

# Pull the base LLM model during build
# This runs ollama in the background, pulls the model, then stops it
RUN ollama serve & \
    sleep 5 && \
    ollama pull phi3:mini && \
    pkill ollama || true

COPY docker-supervisord.conf /etc/supervisor/supervisord.conf

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
