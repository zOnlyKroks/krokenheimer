@echo off
REM Setup script for Windows Remote Training Client

echo ========================================
echo Windows Remote Training Client Setup
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found:
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip

REM Install ALL dependencies in one go
echo.
echo Installing all dependencies...
venv\Scripts\python.exe -m pip install requests schedule torch torchvision torchaudio transformers tokenizers datasets numpy pandas tqdm pywin32

REM Verify requests is installed
echo.
echo Verifying requests installation...
venv\Scripts\python.exe -c "import requests; print('requests installed successfully')"
if errorlevel 1 (
    echo ERROR: requests failed to install
    echo Trying alternative installation...
    venv\Scripts\pip.exe install requests
)

REM Try DirectML as optional
echo.
echo Attempting DirectML installation (optional GPU support)...
venv\Scripts\python.exe -m pip install torch-directml
if errorlevel 1 (
    echo DirectML not available - will use CPU fallback
)

REM Final test
echo.
echo Testing all packages...
venv\Scripts\python.exe -c "import requests, torch, transformers; print('All packages installed successfully!')"
if errorlevel 1 (
    echo ERROR: Package verification failed
    pause
    exit /b 1
)

REM Create config template
echo.
echo Creating configuration template...
if not exist "remote_config.json" (
    echo {"bot_host": "192.168.1.100", "bot_port": 3000, "auth_token": "your_secure_token_here", "training_interval_hours": 12, "min_messages_threshold": 1000, "gpu_enabled": true, "gpu_type": "directml", "max_epochs": 10, "upload_trained_model": true} > remote_config.json
    echo Created remote_config.json
)

REM Create directories
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Edit remote_config.json with your bot's IP and token
echo 2. Test connection: run_trainer.bat --test-connection
echo 3. Start training: run_trainer.bat --daemon
echo.
echo Available commands:
echo   run_trainer.bat --test-connection    Test API connection
echo   run_trainer.bat --daemon            Run continuous training
echo   run_scheduler.bat --install-task    Install Windows task
echo   run_service.bat install             Install Windows service
echo.

pause