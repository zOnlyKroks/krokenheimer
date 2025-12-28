# Krokenheimer Discord Bot

An extensible Discord bot with plugin architecture featuring automated LLM-powered message generation.

## Features

### Core Plugins
- **Core** - Help, ping, and plugin management
- **GIF Creator** - Create GIFs from images with customizable settings
- **Bioinformatics** - DNA sequence analysis and BLAST integration
- **Calculator** - Advanced mathematical calculations
- **ASCII Art** - Generate ASCII art from text
- **Down Checker** - Website monitoring and uptime tracking

### LLM Plugin (New!)
- **Automatic Learning** - Collects and learns from all Discord messages
- **Context-Aware Generation** - Generates messages that fit each channel's style
- **Local Processing** - No cloud APIs, runs entirely on your hardware
- **Scheduled Posting** - Automatically posts at random intervals
- **Zero Manual Work** - Fully automated after setup

## Quick Start

### Docker (Recommended - Includes Everything)

**Easiest way to run with full LLM features:**

```bash
# 1. Configure environment
cp .env.example .env
nano .env  # Add your BOT_TOKEN

# 2. Build and run
./docker-run.sh

# 3. Management commands
docker logs -f krokenheimer-bot  # View logs
docker restart krokenheimer-bot  # Restart
./docker-stop.sh                 # Stop
```

See [DOCKER.md](DOCKER.md) for detailed Docker instructions.

### Basic Setup (Without LLM)

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env and add your BOT_TOKEN

# 3. Run the bot
npm run dev
```

### Full Setup (With LLM Features)

**macOS:**
```bash
# 1. Install prerequisites
brew install ollama screen
pip install chromadb

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env
# Edit .env and add your BOT_TOKEN

# 4. Start everything
./start-all.sh
```

**Linux:**
```bash
# 1. Install prerequisites
curl -fsSL https://ollama.com/install.sh | sh
sudo apt install screen pipx
pipx install chromadb
pipx ensurepath  # Then log out and back in

# 2. Install dependencies
npm install

# 3. Configure environment
cp .env.example .env
# Edit .env and add your BOT_TOKEN

# 4. Start everything
./start-all.sh
```

**That's it!** The bot will automatically:
- Learn from Discord messages
- Build contextual understanding per channel
- Generate and post messages (after 10+ messages per channel)

## Commands

### Core Commands
- `!help [command]` - Show available commands or detailed command info
- `!ping` - Check bot latency
- `!plugins` - List loaded plugins

### GIF Commands
- `!gif` - Create a GIF from attached images
- `!gifhelp` - Show detailed GIF creation options

### Bioinformatics Commands
- `!biostats` - Analyze DNA sequences
- `!alias` - View species name aliases

### Calculator Commands
- `!calc <expression>` - Advanced mathematical calculations

### ASCII Art Commands
- `!ascii <text>` - Generate ASCII art

### Website Monitoring Commands
- `!monitor <url>` - Start monitoring a website
- `!unmonitor <url>` - Stop monitoring a website
- `!status <url>` - Check website status
- `!monitored` - List all monitored websites

### LLM Commands
- `!llmstats` - View learning statistics and generation status

## LLM Configuration

The LLM plugin is fully configurable via environment variables:

```env
# Enable/disable auto-generation
LLM_ENABLED=true

# Generation interval (1-3 hours by default)
LLM_MIN_INTERVAL_MINUTES=60
LLM_MAX_INTERVAL_MINUTES=180

# Channel filtering (optional)
LLM_CHANNEL_IDS=123,456,789          # Only these channels
LLM_EXCLUDE_CHANNEL_IDS=111,222,333  # Exclude these channels

# Model settings
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TEMPERATURE=0.9
```

See [LLM_SETUP.md](LLM_SETUP.md) for detailed configuration and troubleshooting.

## Scripts Reference

| Script | Description |
|--------|-------------|
| `npm run dev` | Build and run the bot (development) |
| `npm run build` | Compile TypeScript to JavaScript |
| `npm start` | Run the bot from compiled files |
| `./start-all.sh` | Start all services + bot in one command |
| `./start-llm-services.sh` | Start Ollama + ChromaDB in background |
| `./stop-llm-services.sh` | Stop background services |

## Architecture

### Plugin System
The bot uses a modular plugin architecture where each plugin can:
- Register commands
- Handle Discord events
- Initialize with the bot client
- Clean up on shutdown

### LLM Architecture
```
Discord Messages
    ↓
Message Collector (automatic)
    ↓
├─ SQLite DB (structured storage)
└─ ChromaDB (vector embeddings)
    ↓
Scheduled Generator (random intervals)
    ↓
Ollama LLM (local inference)
    ↓
Post to Discord
```

## Requirements

### Basic Bot
- Node.js 18+
- npm

### LLM Features
- **CPU**: 12+ cores recommended
- **RAM**: 32GB minimum for llama3.2:3b (16GB for minimal mode)
- **Storage**: ~100MB for dependencies + ~30MB per 10k messages
- **OS**: macOS, Linux (Windows with WSL)

**Additional Software**:
- [Ollama](https://ollama.com) - Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) - Vector database
- `screen` - Background process management (optional)

## Development

```bash
# Watch mode (auto-rebuild on changes)
npm run dev:watch

# In another terminal
npm start
```

## Project Structure

```
krokenheimer/
├── src/
│   ├── core/
│   │   ├── Bot.ts              # Main bot class
│   │   └── util/logger.ts      # Logging utility
│   ├── plugins/
│   │   ├── CorePlugin.ts       # Core commands
│   │   ├── LLMPlugin.ts        # LLM features
│   │   └── ...                 # Other plugins
│   ├── services/
│   │   ├── MessageStorageService.ts  # SQLite storage
│   │   ├── VectorStoreService.ts     # ChromaDB integration
│   │   ├── OllamaService.ts          # LLM interface
│   │   └── ...                       # Other services
│   ├── types/
│   │   ├── index.ts            # Bot type definitions
│   │   └── llm.ts              # LLM type definitions
│   └── index.ts                # Entry point
├── data/                       # SQLite database (auto-created)
├── chroma_data/                # ChromaDB storage (auto-created)
├── .env.example                # Environment template
├── LLM_SETUP.md               # Detailed LLM guide
└── package.json
```

## Troubleshooting

### Bot won't start
- Check `BOT_TOKEN` in `.env`
- Verify Discord bot has proper intents enabled (Message Content Intent)

### LLM features not working
- Ensure Ollama is running: `curl http://localhost:11434/api/tags`
- Ensure ChromaDB is running: `curl http://localhost:8000/api/v1/heartbeat`
- Check model is downloaded: `ollama list`
- View detailed logs: `screen -r ollama_server` or `screen -r chroma_server`

### Out of memory
- Use smaller model: `OLLAMA_MODEL=llama3.2:3b`
- Reduce context: `OLLAMA_CONTEXT_WINDOW=1024`

See [LLM_SETUP.md](LLM_SETUP.md) for more troubleshooting.

## Privacy

- All message data stays on your local machine
- No cloud APIs or external services (except Discord)
- Data stored in `./data/` and `./chroma_data/`
- Delete these directories to clear all learned data

## License

Private project

## Contributing

This is a private Discord bot. For issues or questions, contact the maintainer.
