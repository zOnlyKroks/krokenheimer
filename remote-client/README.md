# Remote Training Client for Windows 11 + RX 5700 XT

This directory contains the remote training client that runs on the Windows 11 machine with RX 5700 XT GPU. The client connects to the main Discord bot, fetches training data, and performs model training locally.

## 🚀 **QUICK START**

1. **Setup**: Run `setup_windows.bat`
2. **Configure**: Edit `remote_config.json` with your bot's IP and token
3. **Start**: Double-click `START_HERE.bat`

That's it! The terminal will stay open showing training progress.

## 📁 **Files Overview**

### **Main Scripts**
- `START_HERE.bat` - **Click this to start training** (keeps terminal open)
- `setup_windows.bat` - One-time setup and installation
- `remote_config.json` - Configuration (edit with bot IP/token)

### **Advanced Scripts**
- `run_trainer.bat` - Manual training control
- `run_scheduler.bat` - Scheduled training management
- `run_service.bat` - Windows service management
- `test_setup.bat` - Verify installation

### **Python Files**
- `remote_trainer.py` - Main training client
- `train_windows.py` - RX 5700 XT optimized training
- `scheduler.py` - Training scheduler
- `windows_service.py` - Windows service wrapper

## 🔧 **For Always-On Training**

**Recommended**: Use `START_HERE.bat` - keeps a terminal visible so you can monitor training progress, model uploads, and any issues.

## 📋 **Commands**

```cmd
# Test connection to Discord bot
run_trainer.bat --test-connection

# Start training daemon (terminal stays open)
START_HERE.bat

# Start training daemon manually
run_trainer.bat --daemon

# Force training now (ignore thresholds)
run_trainer.bat --force-train

# Install Windows Task Scheduler
run_scheduler.bat --install-task

# Install Windows Service
run_service.bat install
```

## ⚡ **Performance with RX 5700 XT**

- **DirectML**: 1-2 hours per training session
- **ROCm** (if installed): 45-90 minutes per training session
- **CPU fallback**: 4-6 hours per training session

Training automatically triggers when:
- New Discord messages ≥ 1000 (configurable)
- Time since last training ≥ 12 hours (configurable)