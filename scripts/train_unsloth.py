#!/usr/bin/env python3
"""
CPU-Compatible LoRA Fine-tuning Script for Krokenheimer Bot
Uses HuggingFace Transformers + PEFT (no Unsloth - that requires GPU)
"""

import json
import sys
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# Config
MAX_SEQ_LENGTH = 2048

def load_training_data(jsonl_path):
    """Load JSONL training data"""
    print(f"📂 Loading training data from {jsonl_path}")
    dataset = load_dataset('json', data_files=jsonl_path, split='train')
    print(f"✅ Loaded {len(dataset)} training examples")
    return dataset

def format_prompts(examples, tokenizer):
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

    # Tokenize
    return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")

def main():
    if len(sys.argv) < 4:
        print("Usage: train_cpu.py <base_model> <training_data.jsonl> <output_model_name>")
        sys.exit(1)

    base_model = sys.argv[1]
    training_data_path = sys.argv[2]
    output_model_name = sys.argv[3]

    # Set nice priority to not kill the server
    os.nice(19)  # Lowest CPU priority

    # Limit threads to prevent CPU overload
    torch.set_num_threads(2)  # Use only 2 CPU threads

    print("🚀 Starting CPU LoRA Fine-tuning (HuggingFace + PEFT)")
    print(f"   Base model: {base_model}")
    print(f"   Output: {output_model_name}")
    print(f"   Device: CPU (expect 3-7 days)")
    print(f"   ⚙️  CPU threads limited to 2 (low priority)")

    # Load model and tokenizer
    print("\n📥 Loading base model...")
    print("   This will download ~2GB on first run (5-15 minutes)")
    print("   Loading model into RAM...")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Tokenizer loaded")

    print("   📦 Loading model weights (this takes a few minutes on CPU)...")
    sys.stdout.flush()  # Force flush to see progress immediately

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,  # CPU needs float32 (was torch_dtype, now deprecated)
        low_cpu_mem_usage=True
    )
    print("   ✅ Model loaded into memory")
    sys.stdout.flush()

    # Add LoRA adapters
    print("🔧 Adding LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    dataset = load_training_data(training_data_path)
    dataset = dataset.map(
        lambda x: format_prompts(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    # Training arguments optimized for CPU
    print("\n⚙️  Setting up training...")
    training_args = TrainingArguments(
        output_dir=f"./data/checkpoints/{output_model_name}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,
        learning_rate=2e-4,
        fp16=False,  # CPU doesn't support fp16
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataloader_num_workers=0,  # Important for CPU
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
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n🏋️  Starting training...")
    print("⏳ This will take 3-7 days on CPU. Go touch grass...")
    trainer.train()

    # Save LoRA adapters
    print("\n💾 Saving LoRA adapters...")
    model.save_pretrained(f"./data/models/{output_model_name}")
    tokenizer.save_pretrained(f"./data/models/{output_model_name}")

    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: ./data/models/{output_model_name}")
    print(f"💡 Import to Ollama with the saved model files")

if __name__ == "__main__":
    main()
