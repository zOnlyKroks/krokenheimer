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


WORKDIR /app
COPY package*.json ./
RUN npm install

WORKDIR /app
COPY src/ ./src/
COPY *.json *.js *.ts *.md ./

RUN npm run build && test -f /app/dist/index.js

RUN npm prune --production

EXPOSE 8000

CMD ["node -e /app/dist/index.js"]
