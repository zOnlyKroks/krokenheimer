@echo off
REM Wrapper to manage Windows service with correct Python

echo Managing Windows service...
venv\Scripts\python.exe windows_service.py %*

if errorlevel 1 (
    echo.
    echo ERROR: Service management failed
    pause
)