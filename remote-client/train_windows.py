#!/usr/bin/env python3
"""
Windows Training Script optimized for RX 5700 XT GPU
Based on the original train_from_scratch.py but adapted for Windows and AMD GPU
"""

import json
import sys
import time
import argparse
from pathlib import Path
import logging
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_windows_environment():
    """Setup Windows environment for MAXIMUM SPEED CPU training."""
    import os

    logger.info("[SPEED] Setting up for ULTRA-FAST CPU training...")

    # Use ALL available cores
    cpu_count = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)

    # Disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    logger.info(f"[SPEED] Using ALL {cpu_count} CPU cores")
    return "cpu"

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

    logger.info(f"[SUCCESS] Loaded {len(conversations)} conversations")
    return conversations

def calculate_dynamic_params(num_conversations: int):
    """Calculate training parameters for FAST 10-20 minute CPU training."""
    logger.info(f"[FAST] Calculating parameters for {num_conversations} conversations")

    # OPTIMIZED FOR SPEED - 10-20 minute training
    params = {
        'learning_rate': 5e-4,      # Higher for faster convergence
        'weight_decay': 0.01,       # Light regularization
        'num_train_epochs': 3,      # FEW epochs (main speed factor)
        'per_device_train_batch_size': 8,    # Larger batches = faster
        'gradient_accumulation_steps': 2,    # Small for speed
        'warmup_steps': 10,         # Minimal warmup
        'save_steps': 50,           # Less frequent saves
        'logging_steps': 5,         # Minimal logging
        'max_grad_norm': 1.0,       # Less restrictive
        'dataloader_num_workers': 8,  # ALL available cores
        'lr_scheduler_type': 'constant',  # Simple = faster
        'eval_steps': None,         # NO evaluation during training
        'save_total_limit': 1,      # Keep only latest checkpoint
        'load_best_model_at_end': False,  # Skip best model selection
        'dataloader_pin_memory': False,
        'fp16': False,
        'bf16': False,
        'tf32': False,
        'gradient_checkpointing': False,  # OFF for speed
        'optim': 'adamw_torch_fused',     # Faster optimizer
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-8,
        'max_steps': 100,           # MAXIMUM 100 steps for speed
    }

    logger.info(f"[SPEED] Target: 10-20 minute training")
    logger.info(f"[SPEED] Using MAX steps: {params['max_steps']} (early stopping)")

    # Adjust based on dataset size
    if num_conversations < 500:
        params.update({
            'num_train_epochs': 2,
            'max_steps': 50,
        })
        logger.info("[TINY] Very small dataset: ultra-fast training")

    elif num_conversations > 2000:
        params.update({
            'per_device_train_batch_size': 4,  # Reduce for memory
            'max_steps': 150,  # Slightly more for larger dataset
        })
        logger.info("[MEDIUM] Medium dataset: adjusted for size")

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

def setup_model_and_trainer(conversations, tokenizer_path: str, output_dir: Path, params: dict):
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

        # LARGE MODEL CONFIG - maximum quality for CPU training
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=1024,  # Longer context for better understanding
            n_embd=768,        # Larger embedding (GPT-2 medium size)
            n_layer=16,        # More layers for better learning
            n_head=12,         # More attention heads
            bos_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            dropout=0.1,       # Some dropout for regularization
            resid_pdrop=0.1,   # Residual dropout
            embd_pdrop=0.1,    # Embedding dropout
        )

        logger.info(f"[LARGE] Model config: {config.n_layer} layers, {config.n_embd} embedding size, {config.n_positions} context")
        logger.info(f"[LARGE] Model parameters: ~{(config.n_layer * config.n_embd * config.n_embd * 4 + config.vocab_size * config.n_embd) // 1000000}M parameters")

        # Initialize model
        model = GPT2LMHeadModel(config)

        # FORCE CPU DEVICE - maximum CPU optimization
        device = torch.device("cpu")
        model = model.to(device)

        # Set CPU optimization flags
        torch.set_num_threads(min(multiprocessing.cpu_count(), 16))  # Use all cores
        torch.set_num_interop_threads(min(multiprocessing.cpu_count(), 4))

        logger.info(f"[CPU] Model forced to CPU device: {device}")
        logger.info(f"[CPU] PyTorch threads: {torch.get_num_threads()}")
        logger.info(f"[CPU] Interop threads: {torch.get_num_interop_threads()}")

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

        # Training arguments with proper device handling
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
            no_cuda=False if str(device).startswith(('cuda', 'privateuseone')) else True,  # Force GPU usage
            max_steps=params.get('max_steps', -1),  # Add max_steps support
        )

        # Log device information for debugging
        logger.info(f"[DEVICE] Training device: {device}")
        logger.info(f"[DEVICE] CUDA available: {torch.cuda.is_available()}")
        logger.info(f"[DEVICE] no_cuda setting: {training_args.no_cuda}")

        # Create trainer with device placement
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )

        # Force model to GPU if DirectML
        if str(device).startswith('privateuseone'):
            trainer.model = trainer.model.to(device)
            logger.info(f"[FORCE] Forced model back to DirectML device: {device}")

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
    parser = argparse.ArgumentParser(description="MAXIMUM QUALITY CPU Training Script")
    parser.add_argument("training_data", help="Path to training data JSONL file")
    parser.add_argument("output_dir", help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (CPU optimized)")

    args = parser.parse_args()

    logger.info("[START] Starting MAXIMUM QUALITY CPU training...")
    logger.info(f"[DATA] Training data: {args.training_data}")
    logger.info(f"[OUTPUT] Output directory: {args.output_dir}")
    logger.info(f"[EPOCHS] Epochs: {args.epochs}")

    # Setup CPU-optimized environment
    setup_windows_environment()

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
        params = calculate_dynamic_params(len(conversations))
        params['num_train_epochs'] = args.epochs

        # Create tokenizer
        tokenizer_path = create_tokenizer(conversations, output_dir)

        # Setup model and trainer
        trainer, model, tokenizer = setup_model_and_trainer(
            conversations, tokenizer_path, output_dir, params
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