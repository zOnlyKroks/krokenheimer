#!/usr/bin/env python3
"""
Unsloth LoRA Fine-tuning Script for Krokenheimer Bot
Trains on CPU with quantization for memory efficiency
"""

import json
import sys
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Config
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # Use 4-bit quantization to save RAM

def load_training_data(jsonl_path):
    """Load JSONL training data"""
    print(f"📂 Loading training data from {jsonl_path}")
    dataset = load_dataset('json', data_files=jsonl_path, split='train')
    print(f"✅ Loaded {len(dataset)} training examples")
    return dataset

def format_prompts(examples):
    """Format data for training"""
    texts = []
    for messages in examples['messages']:
        # Convert chat format to single string
        text = ""
        for msg in messages:
            if msg['role'] == 'system':
                text += f"### System:\n{msg['content']}\n\n"
            elif msg['role'] == 'user':
                text += f"### User:\n{msg['content']}\n\n"
            elif msg['role'] == 'assistant':
                text += f"### Assistant:\n{msg['content']}"
        texts.append(text)
    return {"text": texts}

def main():
    if len(sys.argv) < 4:
        print("Usage: train_unsloth.py <base_model> <training_data.jsonl> <output_model_name>")
        sys.exit(1)

    base_model = sys.argv[1]
    training_data_path = sys.argv[2]
    output_model_name = sys.argv[3]

    print("🚀 Starting Unsloth LoRA Fine-tuning")
    print(f"   Base model: {base_model}")
    print(f"   Output: {output_model_name}")
    print(f"   Device: {'CPU (slow but works)' if not torch.cuda.is_available() else 'GPU'}")

    # Load model with LoRA
    print("\n📥 Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect best dtype
        load_in_4bit=LOAD_IN_4BIT,
    )

    # Add LoRA adapters
    print("🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank (higher = more capacity, slower)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,  # Optimized for small datasets
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=42,
    )

    # Load and format dataset
    dataset = load_training_data(training_data_path)
    dataset = dataset.map(format_prompts, batched=True)

    # Training arguments optimized for CPU
    print("\n⚙️  Setting up training...")
    training_args = TrainingArguments(
        output_dir=f"./data/checkpoints/{output_model_name}",
        per_device_train_batch_size=1,  # Small batch for CPU
        gradient_accumulation_steps=4,   # Effective batch size = 4
        warmup_steps=10,
        max_steps=200,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=False,  # CPU doesn't support fp16
        bf16=False,  # Most CPUs don't support bf16
        logging_steps=10,
        optim="adamw_8bit",  # Memory efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,  # Keep only 2 checkpoints
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # Train
    print("\n🏋️  Starting training...")
    print("⏳ This will take 3-7 days on CPU. Grab a coffee (or 100)...")
    trainer.train()

    # Save LoRA adapters
    print("\n💾 Saving LoRA adapters...")
    model.save_pretrained(f"./data/models/{output_model_name}")
    tokenizer.save_pretrained(f"./data/models/{output_model_name}")

    # Export to GGUF for Ollama
    print("\n📦 Exporting to GGUF for Ollama...")
    model.save_pretrained_gguf(
        f"./data/models/{output_model_name}",
        tokenizer,
        quantization_method="q4_k_m"  # Good balance of size/quality
    )

    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: ./data/models/{output_model_name}")
    print(f"📦 GGUF file ready for Ollama import")

if __name__ == "__main__":
    main()
