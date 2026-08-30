#!/usr/bin/env python3
"""Local training script for AMD GPUs (ROCm)"""

import json
import sys
from pathlib import Path
from typing import List, Dict

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
except ImportError as e:
    print(f"Missing required package: {e}")
    print("\nInstall required packages:")
    print("pip install torch transformers trl peft datasets accelerate")
    sys.exit(1)


class LocalDiscordTrainer:
    def __init__(self, messages_file: str = "messages_export.json", model_name: str = "google/gemma-2-2b"):
        self.messages_file = messages_file
        self.model_name = model_name
        self.output_dir = "./trained_model"

    def load_messages(self) -> List[Dict]:
        """Load messages from JSON export"""
        if not Path(self.messages_file).exists():
            print(f"Error: Messages file not found at {self.messages_file}")
            print("Run !scan on your Discord bot to export messages first")
            sys.exit(1)

        with open(self.messages_file, 'r', encoding='utf-8') as f:
            messages = json.load(f)

        print(f"Loaded {len(messages)} messages from {self.messages_file}")
        return messages

    def prepare_training_data(self, messages: List[Dict]) -> List[str]:
        """Create conversation windows from messages"""
        training_examples = []
        window_size = 5

        # Filter quality messages
        quality_messages = [
            m for m in messages
            if len(m['content']) > 3
            and not (m['content'].startswith('!') and len(m['content']) < 20)
            and not (m['content'].startswith('http') and ' ' not in m['content'])
        ]

        print(f"Filtered: {len(quality_messages)}/{len(messages)} messages")

        # Create conversation windows
        for i in range(len(quality_messages) - window_size):
            window = quality_messages[i:i + window_size]
            conversation = [f"{msg['username']}: {msg['content']}" for msg in window]
            training_examples.append("\n".join(conversation))

        print(f"Created {len(training_examples)} training examples")
        return training_examples

    def train(self, max_steps: int = 500, learning_rate: float = 2e-4):
        """Train the model with LoRA on AMD GPU"""
        print(f"\n{'='*50}")
        print(f"Starting local fine-tuning on AMD GPU")
        print(f"Model: {self.model_name}")
        print(f"Max steps: {max_steps}")
        print(f"{'='*50}\n")

        # Load messages
        print("Loading messages from JSON export...")
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

        # Check device - DirectML for AMD GPU on Windows
        use_gpu = False
        device = "cpu"

        try:
            import torch_directml
            dml = torch_directml.device()
            device = dml
            use_gpu = True
            print("AMD GPU detected - using DirectML")
            print("Your RX 6900 XT will be used for training")

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model = model.to(device)
        except ImportError:
            print("DirectML not installed - falling back to CPU")
            print("Training will be slower without GPU")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        # LoRA config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2 if use_gpu else 1,  # Smaller batch for DirectML
            gradient_accumulation_steps=8,  # Accumulate more to compensate
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_steps=10,
            save_steps=100,
            warmup_steps=50,
            fp16=False,  # DirectML uses float32
            optim="adamw_torch",
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

        # Train
        print("\n🔥 Starting training...")
        print(f"Training on {len(training_texts)} examples")
        print(f"This is REAL training with gradient descent\n")

        trainer.train()

        # Save the model
        print("\n💾 Saving trained model...")
        model.save_pretrained(f"{self.output_dir}/final")
        tokenizer.save_pretrained(f"{self.output_dir}/final")

        print(f"\n✅ Training complete!")
        print(f"Model saved to: {self.output_dir}/final")
        print(f"\nNext steps:")
        print(f"1. Use the upload script to deploy the model to your server")
        print(f"2. Create a Modelfile for Ollama")
        print(f"3. Run: ollama create discord-bot-trained -f Modelfile")

        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Discord bot model locally")
    parser.add_argument("--messages", default="messages_export.json", help="Path to messages export JSON file")
    parser.add_argument("--model", default="google/gemma-2-2b", help="Base model to fine-tune")
    parser.add_argument("--steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    trainer = LocalDiscordTrainer(messages_file=args.messages, model_name=args.model)
    success = trainer.train(max_steps=args.steps, learning_rate=args.lr)

    if success:
        print("\n🎉 Training successful!")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)