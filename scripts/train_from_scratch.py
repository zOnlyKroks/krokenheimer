#!/usr/bin/env python3
"""
From-Scratch Training for Discord Bot
Trains a small GPT-2 model from scratch using ONLY your Discord messages.

WARNING: This approach requires 50k-100k+ messages for coherent output.
Current message count will produce repetitive/incoherent text initially.

IMPORTANT: For long-running training, use tmux or screen:
    tmux new -s training
    python3 scripts/train_from_scratch.py [training_data.jsonl] [output_name]
    # Detach: Ctrl+B, then D
    # Reattach: tmux attach -t training
"""

import torch
import os
import sys
import json
import argparse
from typing import Dict, List
from pathlib import Path
import time
import warnings

# Suppress tokenizer warnings
warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

# ==================== CPU OPTIMIZATION ====================
def setup_cpu_environment():
    """Optimize for CPU training"""
    # Set process priority to prevent SSH disconnects
    try:
        import psutil
        p = psutil.Process()
        p.nice(10)
        print("✓ Process priority set to nice=10 (prevents SSH disconnects)")
    except ImportError:
        print("⚠️  psutil not installed - cannot set CPU priority")
    except Exception as e:
        print(f"⚠️  Could not set process priority: {e}")

    # Set thread pools
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['OPENBLAS_NUM_THREADS'] = '8'

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)

    print("=" * 70)
    print("CPU Training from Scratch - Discord Bot")
    print("=" * 70)
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Device: CPU")
    print("=" * 70)

def load_training_data(jsonl_path: str) -> List[Dict]:
    """Load JSONL training data"""
    data = []
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"✓ Loaded {len(data)} training examples")
        return data
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        sys.exit(1)

def format_conversation(example: Dict) -> str:
    """Format conversation for training"""
    messages = example.get('messages', [])
    formatted = ""

    for msg in messages:
        role = msg['role']
        content = msg['content']

        if role == 'system':
            formatted += f"<|system|>\n{content}\n"
        elif role == 'user':
            formatted += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            formatted += f"<|assistant|>\n{content}\n"

    formatted += "<|endoftext|>"
    return formatted.strip()

def main():
    parser = argparse.ArgumentParser(description='Train GPT-2 from scratch on Discord messages')
    parser.add_argument('training_data', type=str, help='Path to training data JSONL')
    parser.add_argument('output_name', type=str, help='Output model name')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from', default=None)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=10)

    args = parser.parse_args()

    # Setup environment
    setup_cpu_environment()

    print("\n🚀 Starting From-Scratch Training")
    print(f"   Training data: {args.training_data}")
    print(f"   Output name: {args.output_name}")
    print(f"   Epochs: {args.epochs}")
    if args.resume:
        print(f"   Resuming from: {args.resume}")
    print("-" * 60)

    # Import here to avoid issues if not installed
    try:
        from transformers import (
            GPT2LMHeadModel,
            GPT2Tokenizer,
            GPT2Config,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install with: pip install transformers datasets")
        sys.exit(1)

    # Load or create model
    print("📥 Initializing model and tokenizer...")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Add special tokens for our Discord format
        special_tokens = {
            'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>'],
            'pad_token': '<|pad|>'
        }
        tokenizer.add_special_tokens(special_tokens)

        if args.resume:
            # Resume from checkpoint
            print(f"📂 Loading model from checkpoint: {args.resume}")
            model = GPT2LMHeadModel.from_pretrained(args.resume)
        else:
            # Create NEW model with GPT-2 Small architecture (124M params)
            print("🆕 Creating NEW GPT-2 Small model (124M parameters)")
            config = GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=512,  # Context length
                n_embd=768,       # Embedding size
                n_layer=12,       # Number of layers
                n_head=12,        # Attention heads
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
            model = GPT2LMHeadModel(config)
            print("✓ Model initialized with RANDOM weights (from scratch)")

        # Resize token embeddings for new special tokens
        model.resize_token_embeddings(len(tokenizer))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,} (100%)")

    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        sys.exit(1)

    # Load and prepare training data
    print("📊 Preparing training data...")
    raw_data = load_training_data(args.training_data)

    # Format conversations
    formatted_texts = [format_conversation(ex) for ex in raw_data]

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512
        )

    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Split train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(eval_dataset)}")

    # Training arguments optimized for CPU
    print("🎯 Setting up training...")

    output_dir = f"./data/models/{args.output_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,  # Save checkpoints frequently for resume
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,  # Lower LR for from-scratch training
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        disable_tqdm=False,
        save_total_limit=3,  # Keep only last 3 checkpoints to save space
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train!
    print("\n" + "=" * 70)
    print("STARTING TRAINING FROM SCRATCH")
    print("⚠️  WARNING: With limited data, expect poor quality initially")
    print("=" * 70)

    start_time = time.time()

    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume)

        # Save final model
        print("\n💾 Saving final model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save training metrics
        metrics = train_result.metrics
        metrics["train_runtime"] = time.time() - start_time
        metrics["samples_per_second"] = len(train_dataset) / metrics["train_runtime"]

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Training time: {metrics['train_runtime']:.0f}s")
        print(f"Samples/second: {metrics['samples_per_second']:.2f}")
        print(f"Final loss: {metrics['train_loss']:.4f}")
        print(f"Model saved to: {output_dir}")
        print("=" * 70)

        # Save metrics
        metrics_path = f"{output_dir}/training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Model checkpoint saved. Resume with: --resume <checkpoint_path>")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
