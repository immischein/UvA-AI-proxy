#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/.session_token"

echo "=== Step 1: handling session token ==="

if [[ -f "$TOKEN_FILE" ]]; then
    echo "Found existing session token."
    read -p "Do you want to use the stored token? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Generating a new session token..."
        python "$SCRIPT_DIR/extract.py" --output "$TOKEN_FILE"
    else
        echo "Using stored token."
    fi
else
    echo "No stored token found. Extracting a new session token..."
    python "$SCRIPT_DIR/extract.py" --output "$TOKEN_FILE"
fi

TOKEN="$(cat "$TOKEN_FILE")"
if [[ -z "$TOKEN" ]]; then
    echo "ERROR: token file is empty — aborting." >&2
    exit 1
fi

echo "=== Step 2: starting UvA proxy server ==="
SESSION_TOKEN="$TOKEN" python "$SCRIPT_DIR/uva_server.py"
