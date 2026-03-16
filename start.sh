#!/bin/bash
# Start local backend server for development/testing
# Usage: ./start.sh [port]

PORT=${1:-8080}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Koda Vocal Health backend on port $PORT..."
echo "  UI:       http://localhost:$PORT"
echo "  Test:     http://localhost:$PORT/test"
echo "  Debug:    http://localhost:$PORT/debug"
echo "  Health:   http://localhost:$PORT/health"
echo ""

cd "$SCRIPT_DIR"
python3 src/vocal_health_backend.py --port "$PORT"
