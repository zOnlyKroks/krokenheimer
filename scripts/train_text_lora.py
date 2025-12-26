#!/usr/bin/env python3
"""
CPU-Optimized LoRA Fine-Tuning for Text Models
For Xeon 8-core/16-thread with 32GB RAM
"""

import torch
import torch.nn as nn
import os
import sys
import json
import argparse
from typing import Dict, List
from pathlib import Path
import numpy as np
import time

# ==================== XEON CPU OPTIMIZATION ====================
def setup_xeon_environment():
    """Optimize for Xeon 8-core/16-thread CPU"""
    # Set thread pools
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['OPENBLAS_NUM_THREADS'] = '8'

    # PyTorch settings
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = True
    torch.backends.mkldnn.allow_tf32 = True

    print("=" * 70)
    print("Xeon CPU Optimization Active")
    print("=" * 70)
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"Device: CPU (CUDA: {torch.cuda.is_available()})")
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
            formatted += f"<|system|>\n{content}</s>\n"
        elif role == 'user':
            formatted += f"<|user|>\n{content}</s>\n"
        elif role == 'assistant':
            formatted += f"<|assistant|>\n{content}</s>\n"

    return formatted.strip()

def main():
    parser = argparse.ArgumentParser(description='CPU LoRA Training for Text Models')
    parser.add_argument('base_model', type=str, help='Hugging Face model name (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)')
    parser.add_argument('training_data', type=str, help='Path to training data JSONL')
    parser.add_argument('output_name', type=str, help='Output model name')

    args = parser.parse_args()

    # Setup environment
    setup_xeon_environment()

    print("\n🚀 Starting CPU LoRA Training")
    print(f"   Base model: {args.base_model}")
    print(f"   Training data: {args.training_data}")
    print(f"   Output name: {args.output_name}")
    print("-" * 60)

    # Import here to avoid issues if not installed
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        import transformers
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Install with: pip install transformers peft datasets")
        sys.exit(1)

    # Load tokenizer and model
    print("📥 Loading model and tokenizer...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        print(f"✓ Model loaded: {model.config.model_type}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Apply LoRA configuration
    print("⚙️  Applying LoRA configuration...")

    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # For Llama architecture
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
            padding="max_length",
            max_length=512  # Adjust based on your context
        )

    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Split train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(eval_dataset)}")

    # Training arguments optimized for CPU
    print("🎯 Setting up training...")

    training_args = TrainingArguments(
        output_dir=f"./data/models/{args.output_name}",
        num_train_epochs=3,  # Start small
        per_device_train_batch_size=1,  # Small batch for CPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Simulate batch size of 8
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=False,  # False for CPU
        bf16=False,   # False for CPU
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        push_to_hub=False,
        gradient_checkpointing=False,  # Too slow on CPU
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked
    )

    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train!
    print("\n" + "=" * 70)
    print("STARTING TRAINING (CPU - This will take a while...)")
    print("=" * 70)

    start_time = time.time()

    try:
        train_result = trainer.train()

        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(f"./data/models/{args.output_name}")

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
        print(f"Model saved to: ./data/models/{args.output_name}")
        print("=" * 70)

        # Save metrics
        metrics_path = f"./data/models/{args.output_name}/training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create a simple Modelfile for Ollama
        modelfile_content = f"""FROM ./data/models/{args.output_name}

PARAMETER temperature 0.7
PARAMETER num_ctx 2048
PARAMETER num_predict 512

SYSTEM You are a helpful AI assistant fine-tuned on Discord conversations.
"""

        modelfile_path = f"./data/models/{args.output_name}/Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        print(f"📄 Modelfile created: {modelfile_path}")
        print("\nTo import into Ollama, run:")
        print(f"  ollama create {args.output_name} -f {modelfile_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Model checkpoint saved.")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()