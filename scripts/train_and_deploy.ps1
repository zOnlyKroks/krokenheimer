# Discord Bot - Complete Training and Deployment Script
# Run this on your Windows PC with AMD RX 6900 XT

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerHost,

    [Parameter(Mandatory=$true)]
    [string]$ServerUser,

    [Parameter(Mandatory=$true)]
    [string]$ServerBotPath,

    [int]$TrainingSteps = 500
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Discord Bot Training & Deployment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Server: $ServerUser@$ServerHost" -ForegroundColor White
Write-Host "  Bot Path: $ServerBotPath" -ForegroundColor White
Write-Host "  Training Steps: $TrainingSteps" -ForegroundColor White
Write-Host ""

# Check if Python is installed
Write-Host "[1/8] Checking Python installation..." -ForegroundColor Green
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "  Found: $pythonVersion" -ForegroundColor Gray
} else {
    Write-Host "  Python not found! Please install Python from python.org" -ForegroundColor Red
    exit 1
}

# Check if ssh is available
Write-Host "[2/8] Checking SSH client..." -ForegroundColor Green
if (Get-Command ssh -ErrorAction SilentlyContinue) {
    Write-Host "  Using OpenSSH" -ForegroundColor Gray
} else {
    Write-Host "  No SSH client found! Install OpenSSH" -ForegroundColor Red
    exit 1
}

# Test SSH connection
Write-Host "  Testing SSH connection..." -ForegroundColor Gray
$testCmd = "ssh $ServerUser@$ServerHost 'echo Connected'"
$testResult = Invoke-Expression $testCmd 2>&1
if ($testResult -like "*Connected*") {
    Write-Host "  SSH connection successful" -ForegroundColor Gray
} else {
    Write-Host "  SSH connection failed. Check credentials and try again." -ForegroundColor Red
    exit 1
}

# Setup Python virtual environment
Write-Host "[3/8] Setting up Python environment..." -ForegroundColor Green
if (!(Test-Path "venv")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Gray
    python -m venv venv
} else {
    Write-Host "  Virtual environment exists" -ForegroundColor Gray
}

# Activate venv
Write-Host "  Activating virtual environment..." -ForegroundColor Gray
$venvActivate = Join-Path (Get-Location) "venv\Scripts\Activate.ps1"
& $venvActivate

# Install PyTorch for AMD GPU on Windows
Write-Host "[4/8] Installing PyTorch with AMD GPU support..." -ForegroundColor Green
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch-directml

# Install training dependencies
Write-Host "  Installing training libraries..." -ForegroundColor Gray
python -m pip install --quiet transformers trl peft datasets accelerate sentencepiece protobuf

Write-Host "  All dependencies installed" -ForegroundColor Gray
Write-Host "  AMD RX 6900 XT will be used for training via DirectML" -ForegroundColor Green

# Download messages from server
Write-Host "[5/8] Downloading messages from server..." -ForegroundColor Green
Write-Host "  You may be prompted for your password again..." -ForegroundColor Gray

# Normalize path (convert relative to absolute on server)
$normalizedPath = if ($ServerBotPath.StartsWith("./")) {
    $ServerBotPath.Substring(2)
} elseif ($ServerBotPath.StartsWith("~/")) {
    $ServerBotPath.Substring(2)
} else {
    $ServerBotPath
}

# Try to download with explicit path
$remotePath = "${ServerUser}@${ServerHost}:~/${normalizedPath}/data/messages_export.json"
$localPath = "..\messages_export.json"

Write-Host "  Downloading from: $remotePath" -ForegroundColor DarkGray

try {
    scp -o StrictHostKeyChecking=no $remotePath $localPath 2>&1 | Out-Null
    if (!(Test-Path $localPath)) {
        throw "File not found after download"
    }
} catch {
    Write-Host "  Failed to download messages!" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Make sure you ran !scan command on Discord first" -ForegroundColor Gray
    Write-Host "  2. Check that the bot path is correct: $ServerBotPath" -ForegroundColor Gray
    Write-Host "  3. Try with absolute path: -ServerBotPath '/home/root/krokenheimer'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  You can also manually download:" -ForegroundColor Yellow
    Write-Host "  scp ${ServerUser}@${ServerHost}:${ServerBotPath}/data/messages_export.json ." -ForegroundColor Gray
    exit 1
}

$messageCount = (Get-Content $localPath | Select-String '"messageId"').Count
Write-Host "  Downloaded ~$messageCount messages" -ForegroundColor Gray

# Train the model
Write-Host "[6/8] Starting model training..." -ForegroundColor Green
Write-Host "  This will take 30-60 minutes on AMD RX 6900 XT" -ForegroundColor Yellow
Write-Host "  Monitor GPU with Task Manager > Performance > GPU" -ForegroundColor Yellow
Write-Host ""

python scripts\local_train.py --messages $localPath --steps $TrainingSteps

if (!(Test-Path "trained_model\final")) {
    Write-Host ""
    Write-Host "  Training failed! Check errors above" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  Training complete!" -ForegroundColor Gray

# Upload trained model
Write-Host "[7/8] Uploading trained model to server..." -ForegroundColor Green

# Create remote directory first
ssh "${ServerUser}@${ServerHost}" "mkdir -p ${ServerBotPath}/models/trained"

Write-Host "  Uploading model files (this may take a few minutes)..." -ForegroundColor Gray

# Upload all files from trained_model/final to server
Get-ChildItem "trained_model\final\*" | ForEach-Object {
    $fileName = $_.Name
    Write-Host "  Uploading $fileName..." -ForegroundColor DarkGray
    scp $_.FullName "${ServerUser}@${ServerHost}:${ServerBotPath}/models/trained/"
}

Write-Host "  Upload complete!" -ForegroundColor Gray

# Deploy on server
Write-Host "[8/8] Deploying model on server..." -ForegroundColor Green

$deployScript = @"
cd $ServerBotPath
cat > Modelfile << 'EOFMARKER'
FROM ./models/trained
TEMPLATE \"\"\"{{ .Prompt }}\"\"\"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOFMARKER
ollama create discord-bot-trained -f Modelfile
if grep -q 'LLM_BASE_MODEL' .env 2>/dev/null; then
    sed -i 's/^LLM_BASE_MODEL=.*/LLM_BASE_MODEL=discord-bot-trained/' .env
else
    echo 'LLM_BASE_MODEL=discord-bot-trained' >> .env
fi
docker-compose restart 2>/dev/null || echo 'Please restart bot manually'
echo 'Deployment complete!'
"@

ssh "${ServerUser}@${ServerHost}" $deployScript

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "ALL DONE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Your bot is now using the trained model!" -ForegroundColor Yellow
Write-Host "Test it in Discord by mentioning the bot." -ForegroundColor Yellow
Write-Host ""