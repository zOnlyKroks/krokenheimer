#!/bin/bash

echo "🔧 Fixing .env file..."

if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    exit 1
fi

# Create backup
cp .env .env.backup
echo "✅ Backed up .env to .env.backup"

# Remove Windows line endings
sed -i 's/\r$//' .env
echo "✅ Removed Windows line endings"

# Remove quotes from values
sed -i 's/BOT_TOKEN="\(.*\)"/BOT_TOKEN=\1/' .env
sed -i "s/BOT_TOKEN='\(.*\)'/BOT_TOKEN=\1/" .env
echo "✅ Removed quotes from BOT_TOKEN"

# Remove trailing/leading whitespace from values
sed -i 's/^\s*BOT_TOKEN\s*=\s*/BOT_TOKEN=/' .env
sed -i 's/^\s*BOT_OWNERS\s*=\s*/BOT_OWNERS=/' .env
echo "✅ Removed whitespace"

# Show result
echo ""
echo "📋 Cleaned .env file:"
echo "---"
grep "BOT_TOKEN=" .env | sed 's/\(BOT_TOKEN=.\{10\}\).*\(.\{10\}\)/\1...\2/'
grep "BOT_OWNERS=" .env
echo "---"

echo ""
echo "✅ .env file fixed! Now restart the container:"
echo "   docker-compose down"
echo "   docker-compose up -d"
