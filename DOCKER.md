# Docker Setup Guide

Run the Krokenheimer Discord bot with full LLM features in a Docker container.

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Add your BOT_TOKEN

# 2. Build and start
./docker-run.sh

# 3. View logs (Ctrl+C to exit, container keeps running)
docker logs -f krokenheimer-bot
```

That's it! The container will automatically:
- Install and run Ollama
- Download the LLM model (llama3.2:3b by default)
- Start ChromaDB
- Start the Discord bot with LLM features

## What's Included

The Docker container runs all required services:
- **Ollama** - Local LLM inference engine
- **ChromaDB** - Vector database for message embeddings
- **Discord Bot** - Your bot with all plugins including LLM

All services are managed by `supervisord` and start automatically.

## Configuration

### Environment Variables

Edit `.env` file:

```env
# Required
BOT_TOKEN=your_discord_bot_token
BOT_OWNERS=your_discord_user_id

# LLM Settings (optional)
LLM_ENABLED=true
LLM_MIN_INTERVAL_MINUTES=60
LLM_MAX_INTERVAL_MINUTES=180
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TEMPERATURE=0.9
```

### Data Persistence

Data is automatically persisted in Docker volumes:
- `./data/` - SQLite message database
- `./chroma_data/` - ChromaDB vector embeddings
- `ollama_models` (named volume) - Downloaded LLM models

This means your bot's learned data survives container restarts.

## Docker Commands

### Basic Operations

```bash
# Start
./docker-run.sh

# Stop
./docker-stop.sh

# Restart
./docker-restart.sh

# View logs
docker logs -f krokenheimer-bot

# Check status
docker ps | grep krokenheimer-bot
```

### Build and Rebuild

```bash
# Rebuild from scratch
docker build --no-cache -t krokenheimer-bot .
./docker-restart.sh

# Or use the restart script (rebuilds automatically)
./docker-restart.sh
```

### Maintenance

```bash
# Execute command in running container
docker exec -it krokenheimer-bot bash

# Check Ollama status inside container
docker exec krokenheimer-bot ollama list

# Check ChromaDB inside container
docker exec krokenheimer-bot curl http://localhost:8000/api/v1/heartbeat

# View supervisor status
docker exec krokenheimer-bot supervisorctl status

# Pull a model manually
docker exec krokenheimer-bot ollama pull llama3.2:3b
```

### Data Management

```bash
# Clear learned data (stop first)
./docker-stop.sh
rm -rf ./data ./chroma_data
./docker-run.sh

# Backup data
tar -czf backup-$(date +%Y%m%d).tar.gz data/ chroma_data/

# Restore data
tar -xzf backup-20231225.tar.gz
docker restart krokenheimer-bot
```

## Troubleshooting

### Check Service Status

```bash
# View all service logs
docker logs -f krokenheimer-bot

# Check individual services via supervisor
docker exec krokenheimer-bot supervisorctl status
```

Expected output:
```
chromadb                         RUNNING   pid 123, uptime 0:05:23
discord-bot                      RUNNING   pid 456, uptime 0:05:20
ollama                           RUNNING   pid 789, uptime 0:05:25
```

### Restart Individual Service

```bash
# Inside container
docker exec krokenheimer-bot supervisorctl restart discord-bot
docker exec krokenheimer-bot supervisorctl restart ollama
docker exec krokenheimer-bot supervisorctl restart chromadb
```

### View Service Logs

```bash
# All services
docker logs -f krokenheimer-bot

# Ollama only
docker exec krokenheimer-bot supervisorctl tail -f ollama

# ChromaDB only
docker exec krokenheimer-bot supervisorctl tail -f chromadb

# Bot only
docker exec krokenheimer-bot supervisorctl tail -f discord-bot
```

### Common Issues

**Bot won't start:**
- Check BOT_TOKEN is set: `docker exec krokenheimer-bot env | grep BOT_TOKEN`
- View bot logs: `docker logs krokenheimer-bot`

**Out of memory:**
- Reduce model size in `.env`: `OLLAMA_MODEL=llama3.2:3b`
- Check container resources: `docker stats krokenheimer-bot`

**Model download failed:**
- Check internet connection
- Restart container: `docker restart krokenheimer-bot`
- Manually pull model: `docker exec krokenheimer-bot ollama pull llama3.2:3b`

**Services not starting:**
- Check supervisor status: `docker exec krokenheimer-bot supervisorctl status`
- View supervisor logs: `docker logs krokenheimer-bot | grep supervisor`

## Resource Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 10GB
- **Model**: llama3.2:3b

### Recommended
- **CPU**: 12+ cores
- **RAM**: 32GB
- **Disk**: 20GB
- **Model**: llama3.2:8b or mistral:7b

### Docker Compose Resources

You can limit resources in `docker-compose.yml`:

```yaml
services:
  krokenheimer:
    # ... other config ...
    deploy:
      resources:
        limits:
          cpus: '12'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 8G
```

## Advanced Configuration

### Using Different Models

Edit `.env`:
```env
# Smaller, faster (3GB model)
OLLAMA_MODEL=llama3.2:3b

# Better quality (8GB model)
OLLAMA_MODEL=llama3.2:8b

# Alternative
OLLAMA_MODEL=mistral:7b
```

After changing, rebuild:
```bash
docker-compose down
docker-compose up -d
```

The new model will be downloaded automatically on first start.

### Custom Docker Build

For advanced users, build with custom options:

```bash
# Build with specific Node version
docker build --build-arg NODE_VERSION=20 -t krokenheimer .

# Build for different architecture
docker buildx build --platform linux/amd64 -t krokenheimer .
```

### Development Mode

Run with live code reload:

```yaml
# docker-compose.override.yml
services:
  krokenheimer:
    volumes:
      - ./src:/app/src
      - ./node_modules:/app/node_modules
    command: npm run dev:watch
```

## Production Deployment

### Using Docker Compose

```bash
# Production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Using Docker Swarm

```bash
docker stack deploy -c docker-compose.yml krokenheimer
```

### Using Kubernetes

Convert compose file:
```bash
kompose convert -f docker-compose.yml
kubectl apply -f .
```

## Security Considerations

- **Never commit `.env`** - Contains sensitive tokens
- **Limit container resources** - Prevent resource exhaustion
- **Use read-only volumes** where possible
- **Run as non-root user** (TODO: add USER directive)
- **Network isolation** - Container only needs outbound access

## Monitoring

### Health Checks

Add to `docker-compose.yml`:

```yaml
services:
  krokenheimer:
    healthcheck:
      test: ["CMD", "supervisorctl", "status"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### View Statistics

```bash
# Real-time resource usage
docker stats krokenheimer-bot

# Detailed inspect
docker inspect krokenheimer-bot
```

## Migration from Host Installation

If you were running the bot directly on your host:

1. **Backup existing data:**
```bash
cp -r data/ data.backup/
cp -r chroma_data/ chroma_data.backup/
```

2. **Stop host services:**
```bash
./stop-llm-services.sh
# Kill any running bot processes
```

3. **Start with Docker:**
```bash
docker-compose up -d
```

Your existing `data/` and `chroma_data/` directories will be mounted and used by the container.

## FAQ

**Q: Can I run without Docker?**
A: Yes! See [LLM_SETUP.md](LLM_SETUP.md) for host installation.

**Q: How do I update the bot?**
A:
```bash
git pull
docker-compose build
docker-compose up -d
```

**Q: Where are the models stored?**
A: Inside the `ollama_models` Docker volume. List with: `docker volume ls`

**Q: Can I use GPU acceleration?**
A: Yes, but requires NVIDIA Docker runtime and Dockerfile modifications. The current setup is CPU-only.

**Q: How much disk space do I need?**
A:
- Base image: ~2GB
- llama3.2:3b model: ~2GB
- llama3.2:8b model: ~4.7GB
- Dependencies: ~1GB
- Runtime data: ~30MB per 10k messages

**Q: Can I run multiple bots?**
A: Yes, duplicate the folder and change container names in `docker-compose.yml`.

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Check supervisor: `docker-compose exec krokenheimer supervisorctl status`
3. Restart container: `docker-compose restart`
4. Rebuild if needed: `docker-compose up -d --build`
