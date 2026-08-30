FROM node:20-slim

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

WORKDIR /app

# Install Node dependencies
COPY package*.json ./
RUN npm install

# Copy application
COPY . .

# Build TypeScript
RUN npm run build && npm prune --production

# Create data directories
RUN mkdir -p /app/data /var/log/supervisor

# Pull base model
RUN ollama serve & \
    sleep 5 && \
    ollama pull gemma2:2b && \
    pkill ollama || true

# Supervisor config
COPY docker-supervisord.conf /etc/supervisor/supervisord.conf

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]