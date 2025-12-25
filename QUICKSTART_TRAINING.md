# 🚀 Quick Start: TRUE Fine-Tuning

Get your bot doing **real machine learning** in 5 minutes.

## Prerequisites

- Bot already running and collecting messages
- Python 3.8+ installed
- 16GB+ RAM
- 50GB+ free disk space

## Setup (One-Time)

```bash
# 1. Run setup script
./scripts/setup_training.sh

# This will:
# - Create Python virtual environment
# - Install Unsloth and dependencies
# - Takes ~10-15 minutes
```

That's it! Setup is done.

## How It Works Now

### Automatic Training

Bot automatically trains every 50 messages (configurable):

```
You: *writes message #50*
Bot: 🎯 Training threshold reached! Starting background training...
      🔥 Training krokenheimer-v1 with Unsloth LoRA...
      ⏳ This will take 3-7 days on CPU.

[Bot continues working normally]

[3-7 days later]
Bot: ✅ Training complete!
     📦 New model: krokenheimer-v1
     🔄 Switching to new model: krokenheimer-v1
     ✅ Now using model: krokenheimer-v1

[Bot now has your server's personality baked into its weights]
```

### Manual Training

Want to trigger training manually?

```
!llmtrain now
```

Check progress:

```
!llmtrain status
```

## What's Happening?

1. **RAG (Real-time)** - Immediate learning
   - Every message → stored in vector DB
   - Bot can access ALL history instantly
   - No training needed

2. **LoRA Fine-Tuning (Long-term)** - Permanent learning
   - Every 50 messages → triggers training
   - 3-7 days on CPU to complete
   - Creates custom model with your server's personality
   - Model switches automatically when done

## Verify It's Working

### Check if training is set up:
```bash
# Should show Python + libraries
./venv/bin/python3 -c "import unsloth; print('✅ Unsloth ready')"
```

### Monitor training:
```
# In Discord
!llmstats

# Shows:
# Training: 🔄 In progress... (or ✅ Idle)
# Messages since last train: 25/50
# Progress: 50%
```

### Watch live training output:
```bash
# If training is running, you'll see:
tail -f nohup.out  # or check console where bot runs
```

## Configuration

Want to change training frequency?

Edit `src/services/FineTuningService.ts`:
```typescript
private trainingThreshold = 50;  // ← Change this number
```

- `50` = Trains often (active servers)
- `500` = Trains weekly (normal servers)
- `1000` = Trains monthly (quiet servers)

## Common Questions

**Q: Can I use GPU instead of CPU?**
A: Yes! If you have NVIDIA GPU, training will auto-detect and use it (20x faster).

**Q: Will training crash my server?**
A: No. Training runs at low priority and bot continues working.

**Q: What if I don't have 7 days?**
A: Use cloud GPU (RunPod/Vast.ai) for $5-20 per training run. Takes 6-12 hours.

**Q: Can I see what it learned?**
A: Yes! The model's behavior will naturally reflect your server's style after training.

**Q: Can I stop/cancel training?**
A: Yes: `pkill -f train_unsloth` (training can be restarted later)

## Troubleshooting

### "Python module 'unsloth' not found"
```bash
./scripts/setup_training.sh
```

### "Out of memory"
Lower batch size in `scripts/train_unsloth.py`:
```python
per_device_train_batch_size=1  # Already lowest
gradient_accumulation_steps=2   # Lower from 4 to 2
```

### Training seems stuck
Check if it's actually running:
```bash
ps aux | grep train_unsloth
```

Should show Python process using CPU.

## Next Steps

**You're done!** The bot will now:
- ✅ Learn from every message (RAG)
- ✅ Auto-train every 50 messages
- ✅ Continuously evolve its personality
- ✅ Never stop learning

Just use your Discord server normally. The bot handles everything automatically.

---

**For detailed info:** See [TRAINING.md](./TRAINING.md)
