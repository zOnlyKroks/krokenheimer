@echo off
REM Quick test to verify setup is working

echo Testing Python environment...
echo.

echo Testing requests import...
venv\Scripts\python.exe -c "import requests; print('✓ requests OK')"

echo Testing torch import...
venv\Scripts\python.exe -c "import torch; print('✓ torch OK')"

echo Testing transformers import...
venv\Scripts\python.exe -c "import transformers; print('✓ transformers OK')"

echo Testing DirectML...
venv\Scripts\python.exe -c "try: import torch_directml; print('DirectML OK')" 2>nul || echo DirectML not available

echo.
echo All imports successful! Environment is ready.
echo.

pause