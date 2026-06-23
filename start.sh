#!/bin/bash
# 量化回测系统启动脚本
# 用法: bash start.sh [backend|frontend|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/Users/flybirp/.workbuddy/binaries/python/envs/default/bin/python"
NODE="/Users/flybirp/.workbuddy/binaries/node/versions/22.22.2/bin/node"
NPM="/Users/flybirp/.workbuddy/binaries/node/versions/22.22.2/bin/npm"

start_backend() {
  echo "🚀 Starting backend on http://localhost:8200 ..."
  cd "$SCRIPT_DIR/backend"
  $PYTHON main.py
}

start_frontend() {
  echo "🚀 Starting frontend on http://localhost:3000 ..."
  cd "$SCRIPT_DIR/frontend"
  NODE_PATH="$SCRIPT_DIR/frontend/node_modules" $NODE node_modules/.bin/vite --host
}

case "${1:-all}" in
  backend) start_backend ;;
  frontend) start_frontend ;;
  all)
    echo "Starting both backend and frontend..."
    start_backend &
    sleep 2
    start_frontend &
    wait
    ;;
  *)
    echo "Usage: bash start.sh [backend|frontend|all]"
    ;;
esac
