#!/bin/bash
# Upload trained model from local PC to server

set -e

echo "📤 Uploading trained model to server..."

# Configuration - Edit these variables for your setup
SERVER_USER="your_username"
SERVER_HOST="your_server_ip_or_hostname"
SERVER_BOT_PATH="/path/to/krokenheimer"
LOCAL_MODEL_PATH="./trained_model/final"

# Check if required variables are set
if [ "$SERVER_USER" = "your_username" ] || [ "$SERVER_HOST" = "your_server_ip_or_hostname" ]; then
    echo "❌ Error: Please edit this script and set SERVER_USER and SERVER_HOST"
    echo ""
    echo "Edit upload_model.sh and set:"
    echo "  SERVER_USER=\"your_username\""
    echo "  SERVER_HOST=\"your_server_ip\""
    echo "  SERVER_BOT_PATH=\"/path/to/krokenheimer\""
    exit 1
fi

# Check if trained model exists
if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "❌ Error: Trained model not found at $LOCAL_MODEL_PATH"
    echo ""
    echo "Make sure to run training first:"
    echo "  python3 local_train.py"
    exit 1
fi

echo "Model path: $LOCAL_MODEL_PATH"
echo "Destination: ${SERVER_USER}@${SERVER_HOST}:${SERVER_BOT_PATH}/models/"

# Create models directory on server if it doesn't exist
ssh "${SERVER_USER}@${SERVER_HOST}" "mkdir -p ${SERVER_BOT_PATH}/models"

# Upload using rsync (preserves permissions, faster for multiple files)
echo "Uploading model files..."
rsync -avz --progress "$LOCAL_MODEL_PATH/" "${SERVER_USER}@${SERVER_HOST}:${SERVER_BOT_PATH}/models/trained/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Upload complete!"
    echo ""
    echo "Next steps on the server:"
    echo "  1. SSH into your server:"
    echo "     ssh ${SERVER_USER}@${SERVER_HOST}"
    echo ""
    echo "  2. Create a Modelfile for Ollama:"
    echo "     cd ${SERVER_BOT_PATH}"
    echo "     cat > Modelfile << 'EOF'"
    echo "FROM ./models/trained"
    echo "TEMPLATE \"\"\"{{ .Prompt }}\"\"\""
    echo "PARAMETER temperature 0.7"
    echo "PARAMETER top_p 0.9"
    echo "EOF"
    echo ""
    echo "  3. Create the Ollama model:"
    echo "     ollama create discord-bot-trained -f Modelfile"
    echo ""
    echo "  4. Update your .env to use the trained model:"
    echo "     LLM_BASE_MODEL=discord-bot-trained"
    echo ""
    echo "  5. Restart the bot"
else
    echo "❌ Upload failed!"
    echo ""
    echo "Make sure:"
    echo "  - SSH access is configured"
    echo "  - rsync is installed"
    echo "  - Server path is correct"
    exit 1
fi