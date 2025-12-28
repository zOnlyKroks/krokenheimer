@echo off
REM Wrapper to run scheduler with correct Python

echo Starting training scheduler...
venv\Scripts\python.exe scheduler.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Scheduler failed to start
    pause
)