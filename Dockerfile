FROM node:20-slim

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

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

# Supervisor config
COPY docker-supervisord.conf /etc/supervisor/supervisord.conf

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]