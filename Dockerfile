FROM node:22-alpine

RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    cairo-dev \
    pango-dev \
    giflib-dev \
    jpeg-dev \
    libpng-dev \
    pkgconfig \
    bash

WORKDIR /app

COPY package*.json ./

RUN npm install --production

COPY . .

RUN npm run build

CMD ["node", "dist/index.js"]
