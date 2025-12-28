# Remote Training Setup Guide

This guide explains how to configure the Krokenheimer Discord bot to train models on a remote Windows machine with AMD GPU acceleration using ROCm.

## Overview

The remote training feature allows the bot to:
- Train models on a powerful Windows machine with AMD GPU
- Use ROCm for AMD GPU acceleration (much faster than CPU)
- Automatically fall back to local CPU training if remote fails
- Handle all file transfers and synchronization automatically

## Prerequisites

### Remote Windows Machine Requirements

- **Windows 10/11** with OpenSSH Server installed
- **AMD GPU** with ROCm support (RX 5000 series or newer recommended)
- **Python 3.8+** installed and accessible via PATH
- **Minimum 16GB RAM** (32GB recommended for large models)
- **50GB+ free disk space** for training data and models
- **Stable internet connection** for SSH and file transfers

### Supported AMD GPUs (ROCm Compatible)

- AMD Radeon RX 5000 series and newer
- AMD Radeon RX 6000 series (recommended)
- AMD Radeon RX 7000 series (recommended)
- AMD Instinct series (data center GPUs)

## Step 1: Setup Remote Windows Machine

### 1.1 Install OpenSSH Server

```powershell
# Open PowerShell as Administrator
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start and enable SSH service
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'

# Configure Windows Firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

### 1.2 Install Python and pip

1. Download Python from [python.org](https://www.python.org/downloads/windows/)
2. Install with "Add Python to PATH" checked
3. Verify installation:

```cmd
python --version
pip --version
```

### 1.3 Install ROCm for Windows

1. Download ROCm from [AMD's official page](https://rocm.docs.amd.com/en/latest/deploy/windows/quick_start.html)
2. Follow the installation instructions for your GPU
3. Reboot after installation
4. Verify ROCm installation:

```cmd
rocm-smi
```

### 1.4 Setup SSH Authentication

#### Option A: SSH Key (Recommended)

On your bot machine:
```bash
# Generate SSH key pair
ssh-keygen -t rsa -b 4096 -f ~/.ssh/krokenheimer_remote

# Copy public key to Windows machine
ssh-copy-id -i ~/.ssh/krokenheimer_remote.pub username@your-windows-ip
```

#### Option B: Password Authentication

Enable password authentication in SSH config:
```powershell
# Edit SSH config (run as Administrator)
notepad C:\ProgramData\ssh\sshd_config

# Add or modify these lines:
PasswordAuthentication yes
PubkeyAuthentication yes

# Restart SSH service
Restart-Service sshd
```

## Step 2: Configure Bot Environment

### 2.1 Install Dependencies

Add SSH dependencies to your bot:
```bash
npm install ssh2 @types/ssh2
```

### 2.2 Configure Environment Variables

Copy and modify `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your remote machine settings:

```env
# Enable remote training
REMOTE_TRAINING_ENABLED=true

# Your Windows machine details
REMOTE_TRAINING_HOST=192.168.1.100
REMOTE_TRAINING_PORT=22
REMOTE_TRAINING_USERNAME=your_username

# SSH key authentication (recommended)
REMOTE_TRAINING_PRIVATE_KEY_PATH=/home/user/.ssh/krokenheimer_remote
# Alternative: password authentication
# REMOTE_TRAINING_PASSWORD=your_password

# Remote paths (Windows-style paths)
REMOTE_TRAINING_WORKING_DIR=C:/temp/krokenheimer
REMOTE_TRAINING_PYTHON_PATH=python

# AMD GPU settings
REMOTE_TRAINING_USE_GPU=true
REMOTE_TRAINING_GPU_TYPE=rocm

# Enable fallback to local training if remote fails
REMOTE_TRAINING_FALLBACK_LOCAL=true

# Connection timeouts
REMOTE_TRAINING_CONNECTION_TIMEOUT=30000
REMOTE_TRAINING_TRANSFER_TIMEOUT=300000
```

## Step 3: Test Remote Connection

### 3.1 Manual SSH Test

Test SSH connection manually:
```bash
ssh -i ~/.ssh/krokenheimer_remote username@your-windows-ip
```

### 3.2 Bot Connection Test

Use Discord command to test from the bot:
```
!llmremote test
```

Expected output:
```
✅ Remote connection test successful
   Python: Python 3.11.5
   ROCm: Available
   GPU: AMD Radeon RX 6800 XT
```

## Step 4: Start Remote Training

### 4.1 Trigger Training

Use the normal training command:
```
!llmtrain now
```

The bot will automatically:
1. Detect remote training is configured
2. Test the remote connection
3. Upload training data to remote machine
4. Install PyTorch with ROCm support if needed
5. Execute training with GPU acceleration
6. Download the trained model back to local machine
7. Clean up remote temporary files

### 4.2 Monitor Training Progress

Check training status:
```
!llmtrain status
```

Training will show:
```
🌐 Training will use remote Windows machine with AMD GPU
⏳ Remote training typically completes faster than local CPU training.
🚀 Executing training on remote machine...
```

## Troubleshooting

### Connection Issues

**Problem**: SSH connection timeout
```
❌ Remote connection failed: connect ETIMEDOUT
```

**Solutions**:
- Check firewall settings on Windows machine
- Verify SSH service is running: `Get-Service sshd`
- Test network connectivity: `ping your-windows-ip`

### Authentication Issues

**Problem**: Authentication failed
```
❌ Remote connection failed: All configured authentication methods failed
```

**Solutions**:
- Verify SSH key permissions: `chmod 600 ~/.ssh/krokenheimer_remote`
- Check username is correct
- Try password authentication temporarily
- Verify public key is in `~/.ssh/authorized_keys` on Windows

### ROCm Issues

**Problem**: ROCm not detected
```
⚠️ ROCm not available, falling back to CPU
```

**Solutions**:
- Verify ROCm installation: `rocm-smi`
- Check GPU compatibility with ROCm
- Try DirectML as alternative: `REMOTE_TRAINING_GPU_TYPE=directml`
- Update AMD GPU drivers

### Python Environment Issues

**Problem**: Missing dependencies
```
❌ Training dependencies not installed
```

**Solutions**:
- The bot will automatically install dependencies
- If it fails, manually install: `pip install torch --index-url https://download.pytorch.org/whl/rocm5.7`
- Check Python PATH is correct in configuration

### Disk Space Issues

**Problem**: Insufficient disk space
```
❌ Training failed: No space left on device
```

**Solutions**:
- Free up disk space on Windows machine
- Change working directory: `REMOTE_TRAINING_WORKING_DIR=D:/temp/krokenheimer`
- Clean up old training files

## Performance Comparison

Training speed comparison (GPT-2 Small, 10k messages):

| Setup | Time | Speed Improvement |
|-------|------|------------------|
| Local CPU (8 cores) | 4-6 hours | Baseline |
| Remote AMD RX 6800 XT | 45-90 minutes | **3-5x faster** |
| Remote AMD RX 7900 XTX | 30-60 minutes | **4-8x faster** |

## Security Considerations

1. **Use SSH keys** instead of passwords
2. **Limit SSH access** to specific IP addresses if possible
3. **Keep ROCm and drivers updated**
4. **Monitor remote machine** for unusual activity
5. **Use strong passwords** for Windows user accounts

## Discord Commands

| Command | Description |
|---------|-------------|
| `!llmremote test` | Test remote connection and show system info |
| `!llmremote status` | Show remote training configuration status |
| `!llmtrain now` | Start training (will use remote if configured) |
| `!llmtrain status` | Show current training progress and location |
| `!llmconfig` | Show complete bot configuration including remote settings |

## Advanced Configuration

### Multiple Remote Machines

Currently supports one remote machine. For multiple machines, you can:
1. Change configuration in `.env` to switch between machines
2. Restart the bot to pick up new configuration

### Custom GPU Memory Settings

For large models or limited GPU memory:
```env
# Use smaller batch sizes for limited VRAM
REMOTE_TRAINING_GPU_TYPE=rocm
# The training script will automatically adjust batch sizes
```

### Network Optimization

For slow connections, increase timeouts:
```env
REMOTE_TRAINING_CONNECTION_TIMEOUT=60000   # 1 minute
REMOTE_TRAINING_TRANSFER_TIMEOUT=1800000   # 30 minutes
```

## Support

If you encounter issues:

1. Check bot logs for detailed error messages
2. Test SSH connection manually
3. Verify ROCm installation with `rocm-smi`
4. Check Discord for help in bot support channels

The remote training feature includes automatic fallback to local CPU training if any issues occur with the remote machine.