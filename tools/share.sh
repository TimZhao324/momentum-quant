#!/bin/bash
# =============================================================================
# share.sh  —  Expose local web dashboard via Cloudflare quick tunnel.
#
#     bash tools/share.sh          (Ctrl+C to stop)
# =============================================================================
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

# 1. Ensure cloudflared binary is available
if [ ! -x tools/cloudflared ]; then
    echo "Downloading cloudflared binary (~39 MB)..."
    curl -sSL -o tools/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x tools/cloudflared
fi
echo "cloudflared $(tools/cloudflared --version 2>&1 | head -1)"

# 2. Check build artifacts
if [ ! -f web/data/overview.json ]; then
    echo "web/data/overview.json is missing. Run: .venv/bin/python tools/pipeline.py" >&2
    exit 1
fi

# 3. Start local background server if not already active
STARTED_SERVER=0
if curl -sf -o /dev/null http://127.0.0.1:8000/; then
    echo "Server already running on port 8000."
else
    echo "Starting serve.py in background..."
    .venv/bin/python tools/serve.py > logs/serve.log 2>&1 &
    SERVER_PID=$!
    STARTED_SERVER=1
    for _ in $(seq 1 20); do
        curl -sf -o /dev/null http://127.0.0.1:8000/ && break
        sleep 0.5
    done
fi

cleanup() {
    if [ "$STARTED_SERVER" = "1" ] && [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        echo -e "\nStopped serve.py."
    fi
}
trap cleanup EXIT

# 4. Open Cloudflare quick tunnel
echo
echo "Opening Cloudflare tunnel..."
echo "-------------------------------------------------------------------"
tools/cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate 2>&1 \
  | grep --line-buffered -E 'trycloudflare\.com|ERR|failed'
