#!/bin/bash

# Quick Start Script for Smart Search
# This script sets up and tests the smart search feature

echo "=========================================="
echo "Smart Search Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"
echo ""

# Check if .env exists
echo "2. Checking environment configuration..."
if [ ! -f .env ]; then
    echo "   ⚠️  .env file not found!"
    echo "   Creating .env template..."
    cat > .env << 'EOF'
# OpenAI API Key (required for smart search)
# Get yours at: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...

# Google OAuth (existing)
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...

# Supabase (existing)
SUPABASE_URL=...
SUPABASE_KEY=...
EOF
    echo "   ✅ Created .env template. Please add your OPENAI_API_KEY."
    echo "   Get your API key at: https://platform.openai.com/api-keys"
    exit 1
else
    echo "   ✅ .env file found"
    if grep -q "OPENAI_API_KEY=sk-" .env; then
        echo "   ✅ OPENAI_API_KEY configured"
    else
        echo "   ⚠️  OPENAI_API_KEY not found in .env"
        echo "   Please add: OPENAI_API_KEY=sk-..."
        echo "   Get your API key at: https://platform.openai.com/api-keys"
        exit 1
    fi
fi
echo ""

# Install dependencies
echo "3. Installing dependencies..."
pip3 install -q python-dateutil
if [ $? -eq 0 ]; then
    echo "   ✅ Dependencies installed"
else
    echo "   ❌ Failed to install dependencies"
    exit 1
fi
echo ""

# Test the query parser
echo "4. Testing query parser..."
python3 test_query_parser.py > /tmp/smart_search_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Query parser working"
    echo ""
    echo "   Sample parsed query:"
    echo "   -------------------"
    grep -A 10 '"all updates from john in the last 3 days"' /tmp/smart_search_test.log | head -15
else
    echo "   ❌ Query parser test failed"
    echo "   Check /tmp/smart_search_test.log for details"
    exit 1
fi
echo ""

# Check if server is running
echo "5. Checking if backend server is running..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "   ✅ Backend server is running"
    echo ""
    echo "6. Testing smart search endpoint..."

    # Test endpoint
    response=$(curl -s -X POST http://localhost:8000/smart-search \
        -H "Content-Type: application/json" \
        -d '{"query": "emails from last week"}')

    if echo "$response" | grep -q "parsed_query"; then
        echo "   ✅ Smart search endpoint working!"
        echo ""
        echo "   Response preview:"
        echo "   ----------------"
        echo "$response" | python3 -m json.tool 2>/dev/null | head -20
    else
        echo "   ⚠️  Smart search endpoint returned unexpected response"
        echo "   Response: $response"
    fi
else
    echo "   ⚠️  Backend server not running"
    echo "   Start with: python3 start_server.py"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start backend: python3 start_server.py"
echo "2. Test endpoint: POST http://localhost:8000/smart-search"
echo "3. Read guide: cat SMART_SEARCH_GUIDE.md"
echo ""
echo "Example queries:"
echo "  - 'all updates from john in the last 3 days'"
echo "  - 'show me receipts from DoorDash this month'"
echo "  - 'emails about the contract negotiation'"
echo "  - 'what did alice say about the bug fix?'"
echo ""
