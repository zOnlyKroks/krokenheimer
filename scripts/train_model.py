#!/usr/bin/env python3
"""Real fine-tuning using LoRA"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict
import sys
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nInstall required packages:")
    print("pip install torch transformers trl peft datasets accelerate bitsandbytes")
    sys.exit(1)


class DiscordModelTrainer:
    def __init__(self, db_path: str = "./data/messages.db", model_name: str = "google/gemma-2-2b"):
        self.db_path = db_path
        self.model_name = model_name
        self.output_dir = "./models/trained"

    def load_messages(self) -> List[Dict]:
        """Load messages from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT messageId, guildId, channelId, userId, username, content, timestamp
            FROM messages
            ORDER BY timestamp ASC
        """)

        messages = []
        for row in cursor.fetchall():
            messages.append({
                'messageId': row[0],
                'guildId': row[1],
                'channelId': row[2],
                'userId': row[3],
                'username': row[4],
                'content': row[5],
                'timestamp': row[6]
            })

        conn.close()
        return messages

    def prepare_training_data(self, messages: List[Dict]) -> List[str]:
        """Create conversation windows from messages"""
        training_examples = []
        window_size = 5

        quality_messages = [
            m for m in messages
            if len(m['content']) > 3
            and not (m['content'].startswith('!') and len(m['content']) < 20)
            and not (m['content'].startswith('http') and ' ' not in m['content'])
        ]

        print(f"Filtered: {len(quality_messages)}/{len(messages)} messages")

        for i in range(len(quality_messages) - window_size):
            window = quality_messages[i:i + window_size]
            conversation = [f"{msg['username']}: {msg['content']}" for msg in window]
            training_examples.append("\n".join(conversation))

        print(f"Created {len(training_examples)} training examples")
        return training_examples

    def train(self, max_steps: int = 500, learning_rate: float = 2e-4):
        """Train the model with LoRA"""
        print(f"\n{'='*50}")
        print(f"Starting REAL fine-tuning")
        print(f"Model: {self.model_name}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*50}\n")

        # Load messages
        print("Loading messages from database...")
        messages = self.load_messages()
        if len(messages) < 100:
            print(f"Error: Not enough messages. Found {len(messages)}, need at least 100")
            return False

        # Prepare training data
        print("Preparing training data...")
        training_texts = self.prepare_training_data(messages)

        if len(training_texts) < 50:
            print(f"Error: Not enough training examples. Created {len(training_texts)}, need at least 50")
            return False

        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})

        # Load model and tokenizer
        print(f"Loading model: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # CPU-friendly loading
        if torch.cuda.is_available():
            print("GPU detected - using 8-bit quantization")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )
        else:
            print("No GPU detected - using CPU (this will be slower)")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        # Prepare for LoRA training
        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)

        # LoRA config - efficient fine-tuning
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments - CPU/GPU specific
        use_gpu = torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2 if not use_gpu else 4,  # Smaller batch for CPU
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_steps=10,
            save_steps=100,
            warmup_steps=50,
            fp16=use_gpu,  # Only use fp16 on GPU
            optim="paged_adamw_8bit" if use_gpu else "adamw_torch",
            save_total_limit=3,
            report_to="none",
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            max_seq_length=512,
            dataset_text_field="text",
        )

        # Train!
        print("\n🔥 Starting training... This will take a while!")
        print(f"Training on {len(training_texts)} examples")
        print(f"This is REAL training with gradient descent\n")

        trainer.train()

        # Save the model
        print("\n💾 Saving trained model...")
        model.save_pretrained(f"{self.output_dir}/final")
        tokenizer.save_pretrained(f"{self.output_dir}/final")

        print(f"\n✅ Training complete!")
        print(f"Model saved to: {self.output_dir}/final")
        print(f"\nTo use with Ollama:")
        print(f"1. Create a Modelfile")
        print(f"2. Run: ollama create discord-bot-trained -f Modelfile")

        return True


if __name__ == "__main__":
    trainer = DiscordModelTrainer()

    # Check if database exists
    if not Path("./data/messages.db").exists():
        print("Error: Database not found at ./data/messages.db")
        print("Make sure to run the bot and scan messages first!")
        sys.exit(1)

    # Run training
    success = trainer.train(max_steps=500)

    if success:
        print("\n🎉 Training successful!")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)