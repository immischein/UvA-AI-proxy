#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/.session_token"

echo "=== Step 1: handling session token ==="

if [[ -f "$TOKEN_FILE" ]]; then
    SAVED_TOKEN="$(cat "$TOKEN_FILE")"
    if [[ -n "$SAVED_TOKEN" ]]; then
        echo "Found existing session token. Validating…"
        if python "$SCRIPT_DIR/extract.py" --validate "$SAVED_TOKEN" 2>/dev/null; then
            echo "Saved token is valid — using it automatically."
        else
            echo "Saved token is invalid or expired. Extracting a new one…"
            python "$SCRIPT_DIR/extract.py" --output "$TOKEN_FILE"
        fi
    else
        echo "Token file is empty. Extracting a new session token…"
        python "$SCRIPT_DIR/extract.py" --output "$TOKEN_FILE"
    fi
else
    echo "No stored token found. Extracting a new session token…"
    python "$SCRIPT_DIR/extract.py" --output "$TOKEN_FILE"
fi

TOKEN="$(cat "$TOKEN_FILE")"
if [[ -z "$TOKEN" ]]; then
    echo "ERROR: token file is empty — aborting." >&2
    exit 1
fi

echo "=== Step 2: starting UvA proxy server ==="
SESSION_TOKEN="$TOKEN" python "$SCRIPT_DIR/uva_server.py"
