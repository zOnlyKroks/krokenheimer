# 🔥 TRUE Fine-Tuning with Unsloth

This bot uses **TRUE LoRA fine-tuning** with Unsloth to actually train model weights on your server's messages.

## What is LoRA Fine-Tuning?

**Not fine-tuning (what other bots do):**
- Just changing system prompts
- Model forgets everything after restart
- No real learning

**TRUE fine-tuning (what this does):**
- **Updates model weights** with LoRA adapters
- Model **permanently learns** your server's style
- Works offline after training
- Creates a custom model version

## How It Works

### 1. Automatic Training
- Bot counts every message
- When threshold reached (default: 50 messages) → auto-trains
- Training runs in background (3-7 days on CPU)
- Bot keeps working normally during training

### 2. What Gets Trained
- **Conversation patterns** from all channels
- **Writing style** and tone
- **Inside jokes** and references
- **Topic preferences** and common subjects
- **Response patterns** to different contexts

### 3. Training Process
```
Message #50 arrives
  ↓
Export all messages as training data (JSONL)
  ↓
Run Unsloth LoRA training (3-7 days on CPU)
  ↓
Export trained model as GGUF
  ↓
Import into Ollama as krokenheimer-v1
  ↓
Bot automatically switches to new model
  ↓
Continues counting for next training cycle
```

## Setup

### First-Time Setup

1. **Install Python dependencies:**
```bash
cd /path/to/krokenheimer
./scripts/setup_training.sh
```

This installs:
- Unsloth (CPU-optimized)
- PyTorch
- Transformers
- TRL (training library)
- All required dependencies

### Configuration

Training threshold in `src/services/FineTuningService.ts`:
```typescript
private trainingThreshold = 50; // Retrain every X new messages
```

Adjust based on your server activity:
- **50**: Very active servers (retrains often)
- **500**: Normal servers (weekly retraining)
- **1000**: Slower servers (monthly retraining)

## Commands

### Check Training Status
```
!llmtrain status
```
Shows:
- Current model version
- Messages until next training
- Training progress %

### Manual Training
```
!llmtrain now
```
Immediately starts training (if >= 500 messages)

### View Overall Stats
```
!llmstats
```
Shows everything including training status

## Training Details

### What Happens During Training

1. **Data Export** (instant)
   - Extracts all messages from SQLite
   - Formats as conversation pairs
   - Saves as JSONL

2. **LoRA Training** (3-7 days on CPU)
   - Downloads base model from HuggingFace
   - Loads with 4-bit quantization (saves RAM)
   - Trains LoRA adapters (rank 16)
   - Saves checkpoints every 50 steps

3. **Export & Import** (30-60 min)
   - Exports to GGUF format
   - Imports into Ollama
   - Creates `krokenheimer-v1`, `v2`, etc.

4. **Auto-Switch** (instant)
   - Bot switches to new model
   - Resets message counter
   - Continues normal operation

### System Requirements

**Minimum:**
- 16GB RAM
- 50GB free disk space
- Multi-core CPU (more cores = faster)

**Recommended:**
- 32GB RAM
- 100GB free disk space
- 8+ core CPU
- SSD for faster I/O

**With GPU (optional):**
- 8GB+ VRAM → 10-20x faster training
- 24GB+ VRAM → Can train 7B models

### CPU vs GPU Training

| Model Size | CPU Time | GPU Time (RTX 4090) |
|-----------|----------|---------------------|
| 1B params | 2-3 days | 2-4 hours |
| 3B params | 5-7 days | 6-12 hours |
| 7B params | 2-3 weeks | 1-2 days |

## Monitoring Training

### Live Output
Training prints to console:
```
🔥 Training krokenheimer-v1 with Unsloth LoRA...
   Base model: unsloth/Llama-3.2-3B-Instruct
   Training data: ./data/training_data.jsonl

📥 Loading base model...
🔧 Adding LoRA adapters...
⚙️  Setting up training...

🏋️  Starting training...
Step 10/200 | Loss: 2.341
Step 20/200 | Loss: 1.892
...
```

### Check Progress
```bash
# View training logs
tail -f nohup.out  # if running in background

# Check disk usage (models are large)
du -sh data/models/*

# Check if training process is running
ps aux | grep train_unsloth
```

## Model Versioning

Models are named: `krokenheimer-v1`, `krokenheimer-v2`, etc.

Each version represents:
- A snapshot of training at that time
- Learned patterns from all messages up to that point
- Can roll back to older versions if needed

### Rollback to Previous Version
```bash
# List available models
ollama list | grep krokenheimer

# Switch back
ollama run krokenheimer-v1  # or v2, v3, etc.
```

Then update the bot config to use that version.

## Troubleshooting

### Training Fails to Start
```
Error: Python module 'unsloth' not found
```
**Fix:** Run `./scripts/setup_training.sh` again

### Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fix:**
- Reduce batch size in `train_unsloth.py`
- Enable 4-bit quantization (already default)
- Close other applications

### Training is Slow
**This is normal on CPU!**
- 3-7 days is expected for 3B models
- Bot continues working during training
- Consider using cloud GPU if faster needed

### Model Import Fails
```
Error: GGUF file not found
```
**Fix:** Training didn't complete. Check logs for errors.

## Advanced: Cloud Training

Want faster training? Use a cloud GPU:

1. **Export training data** from your server
2. **Upload** to RunPod/Vast.ai/Lambda Labs
3. **Run training** on GPU (6-12 hours)
4. **Download** GGUF file
5. **Import** to Ollama on your server

Cost: $5-20 per training run

## Technical Details

### LoRA Parameters
```python
r=16              # Rank (higher = more capacity)
lora_alpha=16     # Scaling factor
target_modules=[  # Which layers to train
  "q_proj", "k_proj", "v_proj", "o_proj",
  "gate_proj", "up_proj", "down_proj"
]
```

### Training Hyperparameters
```python
batch_size=1              # Per device
gradient_accumulation=4   # Effective batch = 4
learning_rate=2e-4        # Standard for LoRA
max_steps=200             # Adjust based on data size
```

### Model Mappings
Bot auto-converts Ollama → HuggingFace:
- `llama3.2:3b` → `unsloth/Llama-3.2-3B-Instruct`
- `llama3.2:1b` → `unsloth/Llama-3.2-1B-Instruct`
- `llama3.1:8b` → `unsloth/Meta-Llama-3.1-8B-Instruct`

## What Makes This Different

**Other bots:**
- Use RAG only (vector search)
- Model never actually learns
- Forget everything when restarted

**This bot:**
- **RAG** for immediate access to new messages
- **LoRA fine-tuning** for permanent learning
- **Hybrid approach** = best of both worlds

The model literally learns your server's personality and bakes it into its weights forever.

---

**Questions?** Check the training logs or run `!llmtrain status` in Discord.
