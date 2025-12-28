#!/usr/bin/env python3
"""
Windows Training Script optimized for RX 5700 XT GPU
Based on the original train_from_scratch.py but adapted for Windows and AMD GPU
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_windows_environment(gpu_type: str = "rocm"):
    """Setup Windows environment for training."""
    logger.info(f"🪟 Setting up Windows environment with {gpu_type.upper()} GPU...")

    if gpu_type.lower() == "rocm":
        # ROCm setup for RX 5700 XT
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.1.0'  # RX 5700 XT is gfx1010
        os.environ['ROCM_PATH'] = r'C:\Program Files\AMD\ROCm\5.7'
        os.environ['HIP_VISIBLE_DEVICES'] = '0'

        # Additional ROCm optimizations
        os.environ['HCC_AMDGPU_TARGET'] = 'gfx1010'
        os.environ['HSA_ENABLE_SDMA'] = '0'  # Disable SDMA for stability

        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"🔥 ROCm detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"🔥 CUDA devices available: {torch.cuda.device_count()}")
            else:
                logger.warning("⚠️ ROCm not detected, falling back to CPU")
                gpu_type = "cpu"
        except ImportError:
            logger.warning("⚠️ PyTorch not available, falling back to CPU")
            gpu_type = "cpu"

    elif gpu_type.lower() == "directml":
        # DirectML setup for Windows
        os.environ['TORCH_DIRECTML_DEVICE'] = '0'

        try:
            import torch_directml
            logger.info("🪟 DirectML device available")
        except ImportError:
            logger.warning("⚠️ DirectML not available, falling back to CPU")
            gpu_type = "cpu"

    else:
        logger.info("💻 Using CPU training")
        gpu_type = "cpu"

    return gpu_type

def load_training_data(file_path: str):
    """Load training data from JSONL file."""
    logger.info(f"📖 Loading training data from {file_path}...")

    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'messages' in data:
                    conversations.append(data['messages'])
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")

    logger.info(f"✅ Loaded {len(conversations)} conversations")
    return conversations

def calculate_dynamic_params(num_conversations: int, gpu_type: str):
    """Calculate training parameters based on dataset size and hardware."""
    logger.info(f"🧮 Calculating parameters for {num_conversations} conversations on {gpu_type}")

    # Base parameters
    params = {
        'learning_rate': 2e-5,
        'weight_decay': 0.05,
        'num_train_epochs': 10,
        'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 4,
        'warmup_steps': 100,
        'save_steps': 500,
        'logging_steps': 50,
        'max_grad_norm': 1.0,
        'dataloader_num_workers': 0,  # Windows can be finicky with multiprocessing
    }

    # Adjust for dataset size
    if num_conversations < 1000:
        params.update({
            'learning_rate': 1e-5,
            'weight_decay': 0.1,
            'num_train_epochs': 5,
            'warmup_steps': 50,
        })
        logger.info("📊 Small dataset: Conservative parameters")

    elif num_conversations > 5000:
        params.update({
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'num_train_epochs': 15,
            'warmup_steps': 500,
        })
        logger.info("📊 Large dataset: Aggressive parameters")

    # Adjust for hardware
    if gpu_type == "cpu":
        params.update({
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 8,
            'dataloader_num_workers': 0,
            'dataloader_pin_memory': False,
        })
        logger.info("💻 CPU mode: Reduced batch size")

    elif gpu_type in ["rocm", "directml"]:
        # RX 5700 XT has 8GB VRAM, be conservative
        params.update({
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 2,
            'dataloader_pin_memory': True,
            'fp16': False,  # AMD GPUs can be finicky with FP16
        })
        logger.info("🎮 GPU mode: Optimized for RX 5700 XT")

    return params

def create_tokenizer(conversations, output_dir: Path):
    """Create and train a custom tokenizer."""
    logger.info("📝 Creating custom tokenizer...")

    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

        # Collect all text for tokenizer training
        texts = []
        for conv in conversations:
            for message in conv:
                if 'content' in message:
                    texts.append(message['content'])

        logger.info(f"Training tokenizer on {len(texts)} messages...")

        # Create BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        # Train tokenizer
        trainer = trainers.BpeTrainer(
            vocab_size=8000,
            min_frequency=2,
            special_tokens=["<|endoftext|>", "<|pad|>", "<unk>"]
        )

        tokenizer.train_from_iterator(texts, trainer)

        # Add post processor
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Save tokenizer
        tokenizer_path = output_dir / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))

        logger.info(f"✅ Tokenizer saved to {tokenizer_path}")
        return str(tokenizer_path)

    except Exception as e:
        logger.error(f"❌ Failed to create tokenizer: {e}")
        raise

def setup_model_and_trainer(conversations, tokenizer_path: str, output_dir: Path, params: dict, gpu_type: str):
    """Setup model and trainer for training."""
    logger.info("🧠 Setting up model and trainer...")

    try:
        from transformers import (
            GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast,
            TrainingArguments, Trainer, DataCollatorForLanguageModeling
        )
        from datasets import Dataset
        import torch

        # Load tokenizer
        tokenizer = GPT2TokenizerFast(tokenizer_file=tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token

        # Create model config - smaller for RX 5700 XT
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=512,  # Reduced context length for memory
            n_embd=512,       # Reduced embedding size
            n_layer=8,        # Reduced layers
            n_head=8,         # Reduced attention heads
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        logger.info(f"🏗️ Model config: {config.n_layer} layers, {config.n_embd} embedding size")

        # Initialize model
        model = GPT2LMHeadModel(config)

        # Move to appropriate device
        device = "cpu"
        if gpu_type == "rocm":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif gpu_type == "directml":
            import torch_directml
            device = torch_directml.device()

        model = model.to(device)
        logger.info(f"🎯 Model moved to device: {device}")

        # Prepare dataset
        def tokenize_function(examples):
            # Concatenate conversation messages
            texts = []
            for conv in examples['conversations']:
                text = ""
                for message in conv:
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    text += f"{role}: {content}\n"
                text += "<|endoftext|>"
                texts.append(text)

            return tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

        # Create dataset
        dataset_dict = {"conversations": conversations}
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=params['num_train_epochs'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            gradient_accumulation_steps=params['gradient_accumulation_steps'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            warmup_steps=params['warmup_steps'],
            logging_steps=params['logging_steps'],
            save_steps=params['save_steps'],
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_num_workers=params['dataloader_num_workers'],
            dataloader_pin_memory=params.get('dataloader_pin_memory', False),
            max_grad_norm=params['max_grad_norm'],
            fp16=params.get('fp16', False),
            logging_dir=str(output_dir / "logs"),
            report_to=[],  # Disable wandb/tensorboard
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        return trainer, model, tokenizer

    except Exception as e:
        logger.error(f"❌ Failed to setup model and trainer: {e}")
        raise

def train_model(trainer, model, tokenizer, output_dir: Path):
    """Train the model."""
    logger.info("🚀 Starting training...")

    try:
        # Start training
        start_time = time.time()
        train_result = trainer.train()
        end_time = time.time()

        # Log training results
        training_time = end_time - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        logger.info(f"📊 Final training loss: {train_result.training_loss:.4f}")

        # Save model and tokenizer
        logger.info("💾 Saving model and tokenizer...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save training metrics
        metrics = {
            "training_loss": train_result.training_loss,
            "training_time_seconds": training_time,
            "global_step": train_result.global_step,
            "num_train_samples": len(trainer.train_dataset),
            "timestamp": time.time()
        }

        with open(output_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Cleanup checkpoints to save space
        checkpoints_dir = output_dir / "checkpoints"
        if checkpoints_dir.exists():
            import shutil
            shutil.rmtree(checkpoints_dir)
            logger.info("🗑️ Cleaned up checkpoint files")

        logger.info(f"🎉 Model successfully saved to {output_dir}")
        return True

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Windows Training Script for RX 5700 XT")
    parser.add_argument("training_data", help="Path to training data JSONL file")
    parser.add_argument("output_dir", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpu-type", choices=["rocm", "directml", "cpu"], default="rocm",
                        help="GPU type to use")

    args = parser.parse_args()

    logger.info("🏁 Starting Windows training script...")
    logger.info(f"📂 Training data: {args.training_data}")
    logger.info(f"📂 Output directory: {args.output_dir}")
    logger.info(f"🔢 Epochs: {args.epochs}")
    logger.info(f"🎮 GPU type: {args.gpu_type}")

    # Setup environment
    gpu_type = setup_windows_environment(args.gpu_type)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load training data
        conversations = load_training_data(args.training_data)
        if not conversations:
            logger.error("❌ No valid conversations found in training data")
            sys.exit(1)

        # Calculate training parameters
        params = calculate_dynamic_params(len(conversations), gpu_type)
        params['num_train_epochs'] = args.epochs

        # Create tokenizer
        tokenizer_path = create_tokenizer(conversations, output_dir)

        # Setup model and trainer
        trainer, model, tokenizer = setup_model_and_trainer(
            conversations, tokenizer_path, output_dir, params, gpu_type
        )

        # Train model
        success = train_model(trainer, model, tokenizer, output_dir)

        if success:
            logger.info("🎉 Training completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Training failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()