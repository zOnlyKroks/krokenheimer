@echo off
REM Wrapper to run remote trainer with correct Python

echo Starting remote training client...
python.exe remote_trainer.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Training client failed to start
    echo Make sure setup_windows.bat completed successfully
    pause
)