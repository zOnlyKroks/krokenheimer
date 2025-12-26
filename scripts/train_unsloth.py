#!/usr/bin/env python3
"""
Xeon-Optimized Training Script
For CPU: 8 cores, 16 threads, 32GB RAM
Optimized for maximum performance on CPU-only systems
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import time
import psutil
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration for Xeon 8-core/16-thread with 32GB RAM"""

    # CPU Optimization
    PHYSICAL_CORES = 8
    LOGICAL_CORES = 16

    # DataLoader optimization
    NUM_WORKERS = 12  # Logical cores - 4 for system
    PREFETCH_FACTOR = 2

    # Training parameters
    DEFAULT_BATCH_SIZE = 32  # Start with this, adjust based on model
    ACCUMULATION_STEPS = 2   # Gradient accumulation
    DEFAULT_EPOCHS = 100
    LEARNING_RATE = 0.001

    # Memory limits (32GB total, leave 4GB for system)
    MAX_RAM_GB = 28
    WARN_RAM_GB = 24

    # Mixed precision (use bfloat16 for CPU)
    MIXED_PRECISION = True
    AMP_DTYPE = torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float32

    # Checkpointing
    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR = Path("logs")

    # Learning rate schedule
    LR_SCHEDULE = 'cosine'  # 'step', 'cosine', 'plateau'
    WARMUP_EPOCHS = 5

# ==================== SYSTEM OPTIMIZATION ====================
def optimize_system_settings():
    """Set optimal environment variables for Xeon CPU"""

    # Set thread pools for BLAS libraries
    os.environ['OMP_NUM_THREADS'] = str(Config.PHYSICAL_CORES)
    os.environ['MKL_NUM_THREADS'] = str(Config.PHYSICAL_CORES)
    os.environ['OPENBLAS_NUM_THREADS'] = str(Config.PHYSICAL_CORES)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(Config.PHYSICAL_CORES)
    os.environ['NUMEXPR_NUM_THREADS'] = str(Config.PHYSICAL_CORES)

    # PyTorch settings
    torch.set_num_threads(Config.PHYSICAL_CORES)  # Use physical cores for ops
    torch.set_num_interop_threads(1)  # Better for many small operations

    # Enable CPU optimizations
    torch.backends.mkldnn.enabled = True
    torch.backends.mkldnn.allow_tf32 = True
    torch.backends.mkl.enabled = True

    # Reduce TensorFlow logging if present
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    print("=" * 70)
    print("SYSTEM CONFIGURATION")
    print("=" * 70)
    print(f"CPU: {Config.PHYSICAL_CORES} cores, {Config.LOGICAL_CORES} threads")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"DataLoader workers: {Config.NUM_WORKERS}")

    # Check RAM
    ram = psutil.virtual_memory()
    print(f"RAM: Total={ram.total/1e9:.1f}GB, Available={ram.available/1e9:.1f}GB")

    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Mixed precision: {Config.MIXED_PRECISION}")
    print("=" * 70)

# ==================== MEMORY MANAGER ====================
class MemoryManager:
    """Monitor and manage memory usage"""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.history = []

    def check_memory(self, stage=""):
        """Check current memory usage and warn if high"""
        current = self.process.memory_info().rss / 1e9  # GB

        # Update peak
        if current > self.peak_memory:
            self.peak_memory = current

        # Record
        self.history.append({
            'time': time.time(),
            'memory_gb': current,
            'stage': stage
        })

        # Warn if approaching limit
        if current > Config.WARN_RAM_GB:
            print(f"⚠️  WARNING: High memory usage: {current:.1f}GB at {stage}")

        return current

    def clear_cache(self):
        """Clear memory caches"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_summary(self):
        """Get memory usage summary"""
        return {
            'peak_memory_gb': self.peak_memory,
            'avg_memory_gb': np.mean([h['memory_gb'] for h in self.history]) if self.history else 0,
            'samples': len(self.history)
        }

# ==================== DATASET HANDLER ====================
class ImageDataset(Dataset):
    """Generic image dataset with caching"""

    def __init__(self, image_paths, labels, transform=None, cache_images=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache_images = cache_images

        # Cache images in RAM if small enough
        if cache_images:
            self._cache = []
            print(f"Caching {len(image_paths)} images in RAM...")
            for img_path in image_paths:
                img = self._load_image(img_path)
                self._cache.append(img)
            print("Caching complete!")
        else:
            self._cache = None

    def _load_image(self, path):
        """Load image with PIL"""
        from PIL import Image
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return blank image as fallback
            return Image.new('RGB', (224, 224), color='white')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image
        if self._cache is not None:
            image = self._cache[idx]
        else:
            image = self._load_image(self.image_paths[idx])

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.labels[idx]

        return image, label

def create_data_loaders(data_dir, batch_size, val_split=0.2, cache_train=False):
    """
    Create optimized DataLoaders for image classification

    Args:
        data_dir: Directory with class subdirectories
        batch_size: Batch size per iteration
        val_split: Validation split ratio
        cache_train: Cache training images in RAM
    """

    # Default transforms (customize as needed)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Find all images
    image_paths = []
    labels = []
    class_names = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes: {class_names}")

    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_paths.append(str(img_path))
                labels.append(class_to_idx[class_name])

    print(f"Total images: {len(image_paths)}")

    # Split into train/val
    indices = np.arange(len(image_paths))
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create datasets
    train_dataset = ImageDataset(
        [image_paths[i] for i in train_indices],
        [labels[i] for i in train_indices],
        transform=train_transform,
        cache_images=cache_train
    )

    val_dataset = ImageDataset(
        [image_paths[i] for i in val_indices],
        [labels[i] for i in val_indices],
        transform=val_transform,
        cache_images=False  # Don't cache validation
    )

    # Custom collate function for efficiency
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

    # Create optimized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False,
        persistent_workers=True if Config.NUM_WORKERS > 0 else False,
        prefetch_factor=Config.PREFETCH_FACTOR,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=Config.NUM_WORKERS // 2,  # Fewer workers for validation
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader, class_names

# ==================== MODEL OPTIMIZATION ====================
def create_model(model_name='resnet18', num_classes=10, pretrained=False):
    """Create and optimize model for CPU training"""

    model_dict = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'efficientnet_b0': models.efficientnet_b0,
        'mobilenet_v2': models.mobilenet_v2,
        'simple_cnn': None  # Custom simple model
    }

    if model_name == 'simple_cnn':
        # Simple CNN for quick testing
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),

                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(128 * 28 * 28, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        model = SimpleCNN(num_classes)

    elif model_name in model_dict:
        # Use torchvision model
        model_fn = model_dict[model_name]
        if pretrained:
            model = model_fn(weights='IMAGENET1K_V1')
        else:
            model = model_fn()

        # Adjust final layer for number of classes
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # Apply optimizations
    model = optimize_model_for_cpu(model)

    return model

def optimize_model_for_cpu(model):
    """Apply CPU-specific optimizations to model"""

    print("Optimizing model for CPU...")

    # 1. Convert to evaluation mode for optimizations
    model.eval()

    # 2. Fuse Conv+BN layers if available (reduces operations)
    if hasattr(torch, 'jit'):
        try:
            model = torch.jit.script(model)
            print("  ✓ JIT compilation applied")
        except:
            pass

    # 3. Set all parameters to require gradient
    for param in model.parameters():
        param.requires_grad = True

    # 4. Print model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model

# ==================== TRAINING ENGINE ====================
class XeonTrainer:
    """Main training engine optimized for Xeon CPUs"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cpu')
        self.memory_manager = MemoryManager()

        # Move model to device
        self.model = self.model.to(self.device)

        # Setup directories
        Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
        Config.LOG_DIR.mkdir(exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': [], 'batch_times': []
        }

        print(f"Training device: {self.device}")

    def setup_training(self, train_loader, learning_rate):
        """Setup loss, optimizer, and scheduler"""

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.05
        )

        # Learning rate scheduler
        if self.config.LR_SCHEDULE == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=learning_rate * 0.01
            )
        elif self.config.LR_SCHEDULE == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:  # ReduceLROnPlateau
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )

        # Warmup scheduler (first few epochs)
        self.warmup_scheduler = None
        if self.config.WARMUP_EPOCHS > 0:
            warmup_steps = len(train_loader) * self.config.WARMUP_EPOCHS
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )

        # Mixed precision scaler (CPU version)
        self.scaler = torch.cpu.amp.GradScaler() if hasattr(torch.cpu.amp, 'GradScaler') else None

    def train_epoch(self, train_loader, epoch, total_epochs):
        """Train for one epoch"""

        self.model.train()
        mem_manager = self.memory_manager

        total_loss = 0
        correct = 0
        total = 0
        batch_times = []

        # Gradient accumulation
        self.optimizer.zero_grad()

        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print("-" * 60)

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start = time.time()

            # Memory check
            if batch_idx % 10 == 0:
                mem_used = mem_manager.check_memory(f"train_batch_{batch_idx}")

            # Move to device
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass with mixed precision
            if self.config.MIXED_PRECISION and hasattr(torch, 'cpu') and hasattr(torch.cpu, 'amp'):
                with torch.cpu.amp.autocast(dtype=self.config.AMP_DTYPE):
                    output = self.model(data)
                    loss = self.criterion(output, target) / self.config.ACCUMULATION_STEPS
            else:
                output = self.model(data)
                loss = self.criterion(output, target) / self.config.ACCUMULATION_STEPS

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update warmup scheduler if exists
                if self.warmup_scheduler and epoch < self.config.WARMUP_EPOCHS:
                    self.warmup_scheduler.step()

            # Statistics
            total_loss += loss.item() * self.config.ACCUMULATION_STEPS
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Timing
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Progress bar
            if batch_idx % max(1, len(train_loader) // 20) == 0 or batch_idx == len(train_loader) - 1:
                avg_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else batch_time
                progress = (batch_idx + 1) / len(train_loader)
                eta = avg_time * (len(train_loader) - batch_idx - 1)

                # Format ETA
                if eta > 3600:
                    eta_str = f"{eta/3600:.1f}h"
                elif eta > 60:
                    eta_str = f"{eta/60:.1f}m"
                else:
                    eta_str = f"{eta:.0f}s"

                acc = 100. * correct / total
                print(f'\r  [{progress*100:3.0f}%] Loss: {loss.item()*self.config.ACCUMULATION_STEPS:.4f} | '
                      f'Acc: {acc:.1f}% | Time: {avg_time:.1f}s/it | ETA: {eta_str}', end='')

        print()  # New line

        # Update learning rate scheduler (not for plateau)
        if self.config.LR_SCHEDULE != 'plateau':
            self.scheduler.step()

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        avg_batch_time = np.mean(batch_times)

        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        self.history['batch_times'].append(avg_batch_time)
        self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

        return epoch_loss, epoch_acc, avg_batch_time

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model"""

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in val_loader:
            # Memory check
            self.memory_manager.check_memory("validation")

            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        val_loss = total_loss / len(val_loader)
        val_acc = 100. * correct / total

        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

        # Update plateau scheduler if using
        if self.config.LR_SCHEDULE == 'plateau':
            self.scheduler.step(val_acc)

        return val_loss, val_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history,
            'config': vars(self.config) if hasattr(self.config, '__dict__') else self.config
        }

        # Regular checkpoint
        checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Best model
        if is_best:
            best_path = Config.CHECKPOINT_DIR / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model with accuracy: {val_acc:.2f}%")

        # Latest model
        latest_path = Config.CHECKPOINT_DIR / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

        return checkpoint_path

    def train(self, train_loader, val_loader, epochs, learning_rate):
        """Main training loop"""

        self.setup_training(train_loader, learning_rate)

        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)

        start_time = time.time()

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                epoch_start = time.time()

                # Train
                train_loss, train_acc, avg_batch_time = self.train_epoch(
                    train_loader, epoch, epochs
                )

                # Validate
                val_loss, val_acc = self.validate(val_loader)

                # Epoch time
                epoch_time = time.time() - epoch_start

                # Print epoch summary
                print(f"\n  Epoch Summary:")
                print(f"    Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"    Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"    Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
                print(f"    Avg Batch Time: {avg_batch_time:.2f}s")
                print(f"    Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

                # Save checkpoint
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc

                if epoch % 5 == 0 or epoch == epochs - 1 or is_best:
                    self.save_checkpoint(epoch, val_acc, is_best)

                # Early stopping check
                if len(self.history['val_acc']) > 20:
                    recent_acc = self.history['val_acc'][-10:]
                    if max(recent_acc) - min(recent_acc) < 0.5:  # Plateau
                        print("  ⚠️  Early stopping: Validation accuracy plateau")
                        break

                # Clear cache
                self.memory_manager.clear_cache()

                print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        except Exception as e:
            print(f"\n\nError during training: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Final summary
            total_time = time.time() - start_time
            memory_summary = self.memory_manager.get_summary()

            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Total time: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m")
            print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
            print(f"Peak memory usage: {memory_summary['peak_memory_gb']:.1f}GB")
            print(f"Average batch time: {np.mean(self.history['batch_times']):.2f}s")

            # Save final plots
            self.plot_training_history()

            return self.history

    def plot_training_history(self):
        """Plot training history"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Batch time plot
        axes[1, 0].plot(self.history['batch_times'])
        axes[1, 0].set_title('Batch Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate plot
        axes[1, 1].plot(self.history['learning_rates'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = Config.LOG_DIR / 'training_history.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Training plots saved to: {plot_path}")

# ==================== COMMAND LINE INTERFACE ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Xeon-Optimized Training Script')

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with class subdirectories')
    parser.add_argument('--cache-data', action='store_true',
                        help='Cache training data in RAM (if dataset fits)')

    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'efficientnet_b0', 'mobilenet_v2', 'simple_cnn'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (auto-detected from data if not specified)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=Config.DEFAULT_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=Config.DEFAULT_BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--accumulation-steps', type=int, default=Config.ACCUMULATION_STEPS,
                        help='Gradient accumulation steps')

    # Optimization arguments
    parser.add_argument('--workers', type=int, default=Config.NUM_WORKERS,
                        help='Number of DataLoader workers')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')

    # Experimental arguments
    parser.add_argument('--test-mode', action='store_true',
                        help='Run quick test with small dataset')

    return parser.parse_args()

# ==================== MAIN FUNCTION ====================
def main():
    """Main entry point"""

    # Parse arguments
    args = parse_args()

    # Update config with args
    Config.NUM_WORKERS = args.workers
    Config.ACCUMULATION_STEPS = args.accumulation_steps
    Config.MIXED_PRECISION = not args.no_mixed_precision

    # Optimize system
    optimize_system_settings()

    # Test mode: create synthetic data
    if args.test_mode:
        print("\nRunning in TEST MODE with synthetic data...")

        # Create synthetic dataset
        train_data = torch.randn(100, 3, 224, 224)
        train_labels = torch.randint(0, 10, (100,))
        val_data = torch.randn(20, 3, 224, 224)
        val_labels = torch.randint(0, 10, (20,))

        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)

        # Create simple dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(4, Config.NUM_WORKERS)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=min(2, Config.NUM_WORKERS)
        )

        num_classes = 10
        class_names = [f"class_{i}" for i in range(num_classes)]

    else:
        # Real data
        print(f"\nLoading data from: {args.data_dir}")

        # Auto-detect number of classes
        if args.num_classes is None:
            class_dirs = [d for d in Path(args.data_dir).iterdir() if d.is_dir()]
            num_classes = len(class_dirs)
            print(f"Auto-detected {num_classes} classes")
        else:
            num_classes = args.num_classes

        # Create data loaders
        train_loader, val_loader, class_names = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=0.2,
            cache_train=args.cache_data
        )

    print(f"\nData loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")

    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained
    )

    # Create trainer
    trainer = XeonTrainer(model, Config)

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.lr
    )

    # Save final results
    results = {
        'config': vars(args),
        'history': history,
        'best_val_acc': trainer.best_val_acc,
        'peak_memory_gb': trainer.memory_manager.get_summary()['peak_memory_gb'],
        'timestamp': datetime.now().isoformat()
    }

    results_path = Config.LOG_DIR / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Checkpoints saved to: {Config.CHECKPOINT_DIR}")
    print("\nTraining complete! 🎉")

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    main()