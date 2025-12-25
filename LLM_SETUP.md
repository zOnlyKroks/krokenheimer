# LLM Plugin Setup Guide

The LLM Plugin enables your Discord bot to automatically learn from server messages and generate contextual responses using a local LLM (no cloud/API costs).

## Quick Start (TL;DR)

```bash
# 1. Install prerequisites
brew install ollama screen  # macOS
pip install chromadb

# 2. Install dependencies
npm install

# 3. Start everything
./start-all.sh

# To stop everything later
# Press Ctrl+C to stop bot, then:
./stop-llm-services.sh
```

That's it! The bot will automatically learn from messages and start posting after collecting enough data.

## Features

- **Automatic Message Collection**: Silently collects all non-command messages from your Discord server
- **Local LLM Generation**: Uses Ollama to generate Discord-style messages based on learned patterns
- **Vector Search**: ChromaDB stores message embeddings for semantic similarity search
- **Scheduled Posting**: Automatically posts messages at randomized intervals
- **Channel-Aware**: Learns and generates messages specific to each channel's context
- **Zero Manual Work**: Fully automated after initial setup

## Prerequisites

### 0. Install screen (for background services)

Optional but recommended for running services in the background:

```bash
# macOS
brew install screen

# Linux
sudo apt install screen

# Check installation
screen --version
```

If you prefer not to use screen, you can run services manually in separate terminals (see Manual Start section).

### 1. Install Ollama

Ollama runs the LLM locally on your CPU (no GPU required).

**Installation:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

**Download the model:**
```bash
ollama pull llama3.2:3b
```

The 3B model is perfect for CPU-only systems with 32GB RAM. For better quality, you can use:
- `llama3.2:8b` (recommended if you have enough RAM)
- `mistral:7b` (alternative)

**Start Ollama server:**
```bash
ollama serve
```

### 2. Install ChromaDB

ChromaDB runs locally and requires Python.

```bash
pip install chromadb
chroma run --path ./chroma_data
```

Or run in a separate terminal:
```bash
python -m chromadb.cli run --path ./chroma_data
```

## Installation

1. **Install dependencies:**
```bash
npm install
```

2. **Configure environment variables** (optional - defaults work out of the box):

Create or edit `.env`:
```env
# Existing Discord config
BOT_TOKEN=your_discord_token
BOT_OWNERS=your_user_id

# LLM Configuration (all optional)
LLM_ENABLED=true
LLM_MIN_INTERVAL_MINUTES=60
LLM_MAX_INTERVAL_MINUTES=180

# Ollama settings (optional)
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TEMPERATURE=0.9
OLLAMA_MAX_TOKENS=200

# Channel filtering (optional)
# LLM_CHANNEL_IDS=123456789,987654321  # Only these channels
# LLM_EXCLUDE_CHANNEL_IDS=111111111    # Exclude these channels
```

## Running the Bot

### Quick Start (Recommended)

Use the provided scripts to manage all services:

```bash
# Start Ollama and ChromaDB in background screen sessions
./start-llm-services.sh

# Start the Discord bot
npm run dev
```

**Managing background services:**
```bash
# View service logs
screen -r ollama_server   # View Ollama logs
screen -r chroma_server   # View ChromaDB logs
# Press Ctrl+A, then D to detach

# Stop all services
./stop-llm-services.sh
```

### Manual Start (Alternative)

If you prefer to run services manually:

1. **Start Ollama** (in a separate terminal):
```bash
ollama serve
```

2. **Start ChromaDB** (in another terminal):
```bash
chroma run --path ./chroma_data
```

3. **Start the bot**:
```bash
npm run dev
```

## How It Works

### Automatic Message Collection
- The bot listens to all messages in your Discord server
- Messages are stored in two places:
  - **SQLite** (`./data/messages.db`) - for structured storage
  - **ChromaDB** - for semantic vector search
- Bot messages and commands (starting with `!`) are ignored

### Automatic Message Generation
- Every minute, the bot checks if it should generate a message
- Generation happens at random intervals between `LLM_MIN_INTERVAL_MINUTES` and `LLM_MAX_INTERVAL_MINUTES`
- The bot:
  1. Picks a random active channel (respecting channel filters)
  2. Retrieves the last 50 messages for context
  3. Generates a contextual message using the LLM
  4. Posts the message to the channel

### Learning Process
- The bot needs at least 10 messages per channel before generating
- More messages = better quality generation
- The LLM learns:
  - Writing style and tone
  - Common topics and phrases
  - Conversation patterns
  - Channel-specific context

## Commands

### `!llmstats`
Shows LLM learning statistics:
- Total messages collected
- Number of vector embeddings
- Active channels and message counts
- LLM configuration
- Auto-generation status

Example output:
```
🤖 LLM Statistics

Message Collection:
• Total messages stored: 1,543
• Vector embeddings: 1,543
• Active channels: 5

Top Channels:
• #general: 832 messages
• #random: 401 messages
• #tech-talk: 310 messages

LLM Configuration:
• Model: llama3.2:3b
• Temperature: 0.9
• Max tokens: 200

Auto-generation:
• Status: ✅ Enabled
• Interval: 60-180 minutes
• Last generation: 45 minutes ago
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENABLED` | `true` | Enable/disable automatic message generation |
| `LLM_MIN_INTERVAL_MINUTES` | `60` | Minimum time between generated messages |
| `LLM_MAX_INTERVAL_MINUTES` | `180` | Maximum time between generated messages |
| `LLM_CHANNEL_IDS` | - | Comma-separated list of channel IDs to post in (optional) |
| `LLM_EXCLUDE_CHANNEL_IDS` | - | Comma-separated list of channel IDs to exclude (optional) |
| `OLLAMA_MODEL` | `llama3.2:3b` | Ollama model to use |
| `OLLAMA_TEMPERATURE` | `0.9` | Creativity (0.0-1.0, higher = more creative) |
| `OLLAMA_MAX_TOKENS` | `200` | Maximum message length in tokens |
| `OLLAMA_CONTEXT_WINDOW` | `2048` | Context window size |

### Channel Filtering

**Post only in specific channels:**
```env
LLM_CHANNEL_IDS=123456789,987654321
```

**Exclude specific channels:**
```env
LLM_EXCLUDE_CHANNEL_IDS=111111111,222222222
```

### Disable Auto-Generation
```env
LLM_ENABLED=false
```

The bot will still collect messages for learning, but won't generate any.

## Troubleshooting

### "Ollama is not running"
- Make sure you ran `ollama serve` in a separate terminal
- Check if Ollama is running: `curl http://localhost:11434/api/tags`

### "Required model not found"
- Run: `ollama pull llama3.2:3b`
- Verify: `ollama list`

### "Failed to initialize ChromaDB"
- Make sure ChromaDB is running: `chroma run --path ./chroma_data`
- Check if port 8000 is available

### Messages not generating
- Check `!llmstats` - you need at least 10 messages per channel
- Verify `LLM_ENABLED=true` in `.env`
- Check Ollama logs for errors

### Out of memory
- Use a smaller model: `OLLAMA_MODEL=llama3.2:3b`
- Reduce context window: `OLLAMA_CONTEXT_WINDOW=1024`

## Architecture

```
Discord Bot
    ↓
LLMPlugin (src/plugins/LLMPlugin.ts)
    ├── Message Collector (on every message)
    │   ├── MessageStorageService (SQLite)
    │   └── VectorStoreService (ChromaDB)
    │
    └── Message Generator (scheduled)
        ├── OllamaService (LLM)
        └── Post to Discord
```

**Services:**
- `MessageStorageService.ts` - SQLite storage for structured queries
- `VectorStoreService.ts` - ChromaDB for semantic search
- `OllamaService.ts` - Ollama LLM integration

**Types:**
- `llm.ts` - TypeScript interfaces for LLM functionality

## Performance

**With 32GB RAM and 12-core CPU:**
- Llama 3.2 3B: ~1-2 seconds per message
- Llama 3.2 8B: ~3-5 seconds per message
- Mistral 7B: ~2-4 seconds per message

**Storage:**
- SQLite database: ~1KB per message
- ChromaDB vectors: ~2KB per message
- 10,000 messages ≈ 30MB total

## Privacy & Data

- All data is stored locally (no cloud)
- Messages are stored in `./data/messages.db` and `./chroma_data/`
- To clear all learned data:
  ```bash
  rm -rf ./data/ ./chroma_data/
  ```

## Script Reference

| Script | Description |
|--------|-------------|
| `./start-all.sh` | Start all services (Ollama, ChromaDB, bot) in one command |
| `./start-llm-services.sh` | Start only Ollama and ChromaDB in background screen sessions |
| `./stop-llm-services.sh` | Stop all background services |
| `screen -r ollama_server` | Attach to Ollama logs (Ctrl+A D to detach) |
| `screen -r chroma_server` | Attach to ChromaDB logs (Ctrl+A D to detach) |
| `screen -list` | List all running screen sessions |

## Tips

1. **Let it learn**: Wait for 50-100 messages per channel before expecting quality output
2. **Adjust temperature**: Lower (0.5-0.7) for more conservative, higher (0.9-1.0) for creative
3. **Channel filtering**: Start with one channel to test, then expand
4. **Monitor with `!llmstats`**: Check learning progress regularly

## Next Steps

After setup, the bot will:
1. ✅ Start collecting messages immediately
2. ✅ Begin generating messages after 10+ messages per channel
3. ✅ Post at random intervals (1-3 hours by default)

No further action required - it's fully automated!
