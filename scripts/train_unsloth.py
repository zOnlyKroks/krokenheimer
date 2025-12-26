import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import psutil
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

# ==================== XEON-SPECIFIC OPTIMIZATION ====================
# Your CPU: 8 cores, 16 threads (Xeon with hyperthreading)
NUM_PHYSICAL_CORES = 8
NUM_LOGICAL_CORES = 16

# Optimal settings for Xeon with hyperthreading
os.environ['OMP_NUM_THREADS'] = '8'  # Use physical cores for OpenMP
os.environ['MKL_NUM_THREADS'] = '8'  # Use physical cores for MKL
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

# PyTorch configuration
torch.set_num_threads(8)  # Use physical cores for PyTorch ops
torch.set_num_interop_threads(1)  # Better for many small operations

# Enable all CPU optimizations
torch.backends.mkldnn.enabled = True
torch.backends.mkl.enabled = True

# DataLoader workers: use logical cores for I/O
NUM_WORKERS = 12  # Leave 4 threads for system/other processes

print(f"=== Xeon CPU Configuration ===")
print(f"Physical Cores: {NUM_PHYSICAL_CORES}")
print(f"Logical Cores: {NUM_LOGICAL_CORES}")
print(f"PyTorch Threads: {torch.get_num_threads()}")
print(f"DataLoader Workers: {NUM_WORKERS}")
print(f"RAM: 32GB Total, {psutil.virtual_memory().available / 1e9:.1f}GB Available")
print("=" * 50)

# ==================== MEMORY-EFFICIENT DATALOADER ====================
class OptimizedDataLoader:
    @staticmethod
    def create_dataloader(dataset, batch_size, shuffle=True, collate_fn=None):
        """Optimized DataLoader for 32GB RAM"""

        # Dynamically adjust batch size based on available memory
        available_ram = psutil.virtual_memory().available / 1e9
        if available_ram < 4:  # Less than 4GB free
            batch_size = max(1, batch_size // 2)
            print(f"Warning: Low RAM ({available_ram:.1f}GB free). Reducing batch size to {batch_size}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=False,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            prefetch_factor=2,
            drop_last=True,  # Better for optimization
            multiprocessing_context='spawn' if NUM_WORKERS > 0 else None,
            collate_fn=collate_fn
        )

# ==================== XEON-OPTIMIZED TRAINING ====================
def train_epoch_xeon(model, train_loader, criterion, optimizer, device,
                     epoch, total_epochs, accumulation_steps=2):
    """Training optimized for Xeon CPUs with 32GB RAM"""

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Monitoring
    mem_monitor = psutil.Process()
    batch_times = []

    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start = time.time()

        # Move to device
        data, target = data.to(device), target.to(device)

        # Use mixed precision if available (PyTorch 1.10+)
        if hasattr(torch, 'cpu') and hasattr(torch.cpu, 'amp'):
            with torch.cpu.amp.autocast():
                output = model(data)
                loss = criterion(output, target) / accumulation_steps
        else:
            # Fallback to regular precision
            output = model(data)
            loss = criterion(output, target) / accumulation_steps

        # Backward
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Statistics
        total_loss += loss.item() * accumulation_steps
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Timing
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Memory usage
        mem_used = mem_monitor.memory_info().rss / 1e9  # GB

        # Progress update every 5 batches
        if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1:
            avg_time = np.mean(batch_times[-5:]) if len(batch_times) >= 5 else batch_time
            eta = avg_time * (len(train_loader) - batch_idx - 1) / 60  # minutes

            print(f'\rEpoch {epoch+1}/{total_epochs} | '
                  f'Batch {batch_idx+1}/{len(train_loader)} | '
                  f'Loss: {loss.item()*accumulation_steps:.4f} | '
                  f'Acc: {100.*correct/total:.1f}% | '
                  f'Time: {avg_time:.1f}s/it | '
                  f'ETA: {eta:.0f}m | '
                  f'RAM: {mem_used:.1f}GB', end='')

    print()  # New line after epoch
    return total_loss / len(train_loader), 100. * correct / total

# ==================== OPTIMIZED VALIDATION ====================
@torch.no_grad()
def validate_xeon(model, val_loader, criterion, device):
    """Validation without gradient overhead"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Disable gradient for the entire validation
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss / len(val_loader), 100. * correct / total

# ==================== MEMORY-EFFICIENT BATCH COLLATOR ====================
def efficient_collate_fn(batch):
    """Custom collate function to reduce memory overhead"""
    if isinstance(batch[0], tuple):
        # Separate data and targets
        data = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Stack with minimal copying
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data, dim=0)
        else:
            data = torch.tensor(np.array(data))

        if isinstance(targets[0], torch.Tensor):
            targets = torch.stack(targets, dim=0)
        else:
            targets = torch.tensor(targets)

        return data, targets
    return torch.utils.data.default_collate(batch)

# ==================== MAIN TRAINING FUNCTION ====================
def train_xeon_optimized(model, train_dataset, val_dataset,
                         num_epochs=100, batch_size=32, learning_rate=0.001):
    """
    Main training function optimized for Xeon + 32GB RAM
    
    Recommendations for your setup:
    - batch_size: 16-32 (depends on model size)
    - accumulation_steps: 2-4 (simulates larger batch)
    """

    device = torch.device('cpu')
    print(f"\nTraining on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimized loaders with custom collate
    train_loader = OptimizedDataLoader.create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=efficient_collate_fn
    )

    val_loader = OptimizedDataLoader.create_dataloader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        collate_fn=efficient_collate_fn
    )

    # Model to device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Use AdamW with carefully tuned parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,
        eta_min=learning_rate * 0.01
    )

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("\n" + "="*70)
    print("Starting Xeon-optimized training...")
    print("="*70)

    total_start = time.time()
    best_val_acc = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch_xeon(
            model, train_loader, criterion, optimizer, device,
            epoch, num_epochs, accumulation_steps=2
        )

        # Validate
        val_loss, val_acc = validate_xeon(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'best_model_xeon_epoch{epoch+1}_acc{val_acc:.1f}.pth')
            print(f"✓ Saved best model with val_acc: {val_acc:.1f}%")

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        print(f"  Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)

        # Early stopping check (optional)
        if epoch >= 20 and val_loss > np.mean(history['val_loss'][-5:]):
            print("Early stopping triggered (validation loss plateau)")
            break

    total_time = time.time() - total_start
    print(f"\n" + "="*70)
    print(f"Training completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m")
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print("="*70)

    return model, history

# ==================== MEMORY OPTIMIZATION UTILITIES ====================
def optimize_model_memory(model):
    """Apply memory optimizations to model"""
    model.eval()

    # Convert to half precision where possible
    try:
        model.half()
        print("Model converted to half precision (float16)")
    except:
        print("Could not convert to half precision, using float32")

    # Freeze early layers if applicable
    for param in model.parameters():
        param.requires_grad = True  # Keep all trainable for now

    return model

def clear_memory():
    """Clear PyTorch and system memory"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()