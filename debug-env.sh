#!/bin/bash

echo "🔍 Environment Variable Debug Script"
echo "====================================="
echo ""

echo "1. Checking .env file location:"
if [ -f .env ]; then
    echo "   ✅ .env file exists in current directory"
    echo "   Path: $(pwd)/.env"
else
    echo "   ❌ .env file NOT found in current directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

echo ""
echo "2. Checking .env file content:"
echo "   File size: $(wc -c < .env) bytes"
echo "   Line count: $(wc -l < .env) lines"

echo ""
echo "3. Checking BOT_TOKEN in .env (showing first/last 10 chars only):"
if grep -q "BOT_TOKEN=" .env; then
    TOKEN=$(grep "BOT_TOKEN=" .env | cut -d'=' -f2)
    TOKEN_LENGTH=${#TOKEN}
    if [ $TOKEN_LENGTH -gt 20 ]; then
        FIRST_10="${TOKEN:0:10}"
        LAST_10="${TOKEN: -10}"
        echo "   ✅ BOT_TOKEN found"
        echo "   Length: $TOKEN_LENGTH characters"
        echo "   Starts with: $FIRST_10..."
        echo "   Ends with: ...$LAST_10"

        # Check for common issues
        if [[ "$TOKEN" == *" "* ]]; then
            echo "   ⚠️  WARNING: Token contains spaces!"
        fi
        if [[ "$TOKEN" == \"*\" ]] || [[ "$TOKEN" == \'*\' ]]; then
            echo "   ⚠️  WARNING: Token is wrapped in quotes (remove them!)"
        fi
        if [[ "$TOKEN" == *$'\r'* ]]; then
            echo "   ⚠️  WARNING: Token contains Windows line endings!"
        fi
    else
        echo "   ❌ BOT_TOKEN is too short ($TOKEN_LENGTH chars)"
    fi
else
    echo "   ❌ BOT_TOKEN not found in .env"
fi

echo ""
echo "4. Checking what Docker Compose sees:"
if command -v docker-compose &> /dev/null; then
    echo "   Running docker-compose config..."
    if docker-compose config | grep -A 2 "BOT_TOKEN" &> /dev/null; then
        echo "   ✅ BOT_TOKEN is in docker-compose config"
    else
        echo "   ❌ BOT_TOKEN not found in docker-compose config"
    fi
else
    echo "   ⚠️  docker-compose not found"
fi

echo ""
echo "5. Checking if container can see the variable:"
if docker ps | grep -q "krokenheimer-bot"; then
    echo "   Container is running, checking environment..."
    TOKEN_IN_CONTAINER=$(docker exec krokenheimer-bot env | grep BOT_TOKEN | cut -d'=' -f2)
    if [ -n "$TOKEN_IN_CONTAINER" ]; then
        TOKEN_LENGTH=${#TOKEN_IN_CONTAINER}
        echo "   ✅ BOT_TOKEN is set in container"
        echo "   Length: $TOKEN_LENGTH characters"
    else
        echo "   ❌ BOT_TOKEN is NOT set in container!"
    fi
else
    echo "   ⚠️  Container is not running"
fi

echo ""
echo "6. Suggested fixes:"
echo ""
echo "   If token has quotes or spaces:"
echo "   nano .env"
echo "   Make sure it looks like: BOT_TOKEN=your_token_here"
echo "   (No quotes, no spaces)"
echo ""
echo "   If container doesn't see the variable:"
echo "   docker-compose down"
echo "   docker-compose up -d"
echo ""
echo "   To fix Windows line endings:"
echo "   dos2unix .env"
echo "   # or"
echo "   sed -i 's/\r$//' .env"
