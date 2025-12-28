@echo off
REM Quick start script for remote training

title Krokenheimer Remote Training Client

echo ========================================
echo   KROKENHEIMER REMOTE TRAINING CLIENT
echo ========================================
echo.

echo Checking configuration...
if not exist "remote_config.json" (
    echo ERROR: Configuration file missing!
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

echo Testing connection to Discord bot...
python.exe remote_trainer.py --test-connection
set EXIT_CODE=%ERRORLEVEL%
echo Debug: Exit code was %EXIT_CODE%

if %EXIT_CODE% NEQ 0 (
    echo.
    echo ERROR: Cannot connect to Discord bot!
    echo Check your remote_config.json settings:
    echo - bot_host: IP address of your Discord bot
    echo - bot_port: API port (usually 3000^)
    echo - auth_token: Must match bot's REMOTE_API_TOKEN
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  CONNECTION SUCCESSFUL - STARTING DAEMON
echo ========================================
echo.
echo Training will start automatically when conditions are met:
echo - Enough new messages (1000+ by default)
echo - Enough time passed (12+ hours by default)
echo.
echo Keep this window open to monitor training progress
echo Press Ctrl+C to stop
echo.

REM Start the daemon
python.exe remote_trainer.py --daemon

pause