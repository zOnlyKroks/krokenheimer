#!/bin/bash
# Download messages export from server to local PC for training

set -e

echo "📥 Downloading messages from server..."

# Configuration - Edit these variables for your setup
SERVER_USER="your_username"
SERVER_HOST="your_server_ip_or_hostname"
SERVER_BOT_PATH="/path/to/krokenheimer"
LOCAL_PATH="./messages_export.json"

# Check if required variables are set
if [ "$SERVER_USER" = "your_username" ] || [ "$SERVER_HOST" = "your_server_ip_or_hostname" ]; then
    echo "❌ Error: Please edit this script and set SERVER_USER and SERVER_HOST"
    echo ""
    echo "Edit download_messages.sh and set:"
    echo "  SERVER_USER=\"your_username\""
    echo "  SERVER_HOST=\"your_server_ip\""
    echo "  SERVER_BOT_PATH=\"/path/to/krokenheimer\""
    exit 1
fi

# Download using scp
echo "Downloading from ${SERVER_USER}@${SERVER_HOST}:${SERVER_BOT_PATH}/data/messages_export.json"

scp "${SERVER_USER}@${SERVER_HOST}:${SERVER_BOT_PATH}/data/messages_export.json" "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    MESSAGE_COUNT=$(grep -o '"messageId"' "$LOCAL_PATH" | wc -l)
    echo ""
    echo "✅ Download complete!"
    echo "📁 Saved to: $LOCAL_PATH"
    echo "📊 Messages: ~$MESSAGE_COUNT"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python3 local_train.py --messages $LOCAL_PATH"
    echo "  2. Wait for training to complete"
    echo "  3. Run: ./upload_model.sh to deploy to server"
else
    echo "❌ Download failed!"
    echo ""
    echo "Make sure:"
    echo "  - SSH access is configured"
    echo "  - Server path is correct"
    echo "  - messages_export.json exists on server (run !scan first)"
    exit 1
fi