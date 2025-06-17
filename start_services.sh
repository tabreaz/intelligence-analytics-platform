#!/bin/bash

echo "🚀 Starting Intelligence Analytics Platform Services"
echo "=================================================="

# Check if Redis is running
echo -n "📍 Checking Redis... "
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Not running"
    echo "   Please start Redis with: redis-server"
    echo "   Or on macOS: brew services start redis"
    exit 1
else
    echo "✅ Running"
fi

# Check if ClickHouse is running (try both local and Docker)
echo -n "🗄️  Checking ClickHouse... "
if clickhouse-client --query="SELECT 1" > /dev/null 2>&1; then
    echo "✅ Running (local)"
elif docker exec clickhouse clickhouse-client --query="SELECT 1" > /dev/null 2>&1; then
    echo "✅ Running (Docker)"
elif curl -s http://localhost:8123/?query=SELECT%201 > /dev/null 2>&1; then
    echo "✅ Running (HTTP interface)"
else
    echo "❌ Not running"
    echo "   Please start ClickHouse with one of:"
    echo "   - Docker: docker run -d --name clickhouse -p 8123:8123 -p 9000:9000 clickhouse/clickhouse-server"
    echo "   - Local: clickhouse-server"
    exit 1
fi

# Check if PostgreSQL is running
echo -n "🐘 Checking PostgreSQL... "
if ! psql -U tabreaz -d sigint -c "SELECT 1" > /dev/null 2>&1; then
    echo "⚠️  Not running or not accessible"
    echo "   Query history will be disabled"
    echo "   To start PostgreSQL: pg_ctl start"
else
    echo "✅ Running"
fi

echo ""
echo "✅ All required services are running!"
echo ""
echo "Starting API server..."
echo "=================================================="

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Run the API server
python run_api.py