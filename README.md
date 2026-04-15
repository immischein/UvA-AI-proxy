# UvA AI Proxy

An OpenAI-compatible API proxy for [aichatacc.uva.nl](https://aichatacc.uva.nl) — the University of Amsterdam's AI chat platform. It lets you connect any tool that speaks the OpenAI API (OpenWebUI, LibreChat, Cursor, etc.) to UvA's AI models using your existing UvA session.

## Features

- **OpenAI-compatible endpoints** — `GET /v1/models` and `POST /v1/chat/completions`
- **Streaming & non-streaming** responses
- **Multi-turn conversation** support (history is replayed via the system prompt)
- **Multiple models** — GPT-4.1, GPT-4o, GPT-5.1, o3, o4-mini, Claude Sonnet/Haiku, and more
- **CORS enabled** — browser-based tools can connect directly

## Requirements

- Python 3.10+
- A valid UvA session token (see [Getting a Session Token](#getting-a-session-token))

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install fastapi uvicorn requests
```

## Getting a S** (install inside the same venv):ion Token

Use the included `extract.py` script to obtain your `__Secure-next-auth.session-token` from the UvA AI chat site.

**Requirements:**

```bash
pip install playwright
playwright install chromium
```

**Usage:**

```bash
# Opens a visible browser — log in manually, token is printed automatically
python extract.py

# Headless mode (only works if SSO auto-login succeeds)
python extract.py --headless

# Save the token to a file
python extract.py --output token.txt
```

## Running the Server

### Quick start (recommended)

`start.sh` handles everything in one go — it extracts a fresh session token and immediately starts the proxy:

```bash
bash start.sh
```

The token is saved to `.session_token` in the repo directory and reused by the server automatically.

### Manual start

If you already have a token, you can start the server directly:

```bash
SESSION_TOKEN=<your-token> python uva_server.py
```

The server starts on `http://0.0.0.0:8000` by default.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SESSION_TOKEN` | *(required)* | Your UvA NextAuth session token |
| `HOST` | `0.0.0.0` | Host to bind to |
| `PORT` | `8000` | Port to listen on |
| `UVA_BASE_URL` | `https://aichatacc.uva.nl` | UvA API base URL |
| `KNOWN_MODELS` | *(see below)* | Comma-separated list of models to expose |

**Default models:** `gpt-5.1, gpt-4.1, gpt-4o, o3, o4-mini, claude-sonnet-4.6, claude-opus-4.6, claude-haiku-4.5`

## Connecting a Client

Point your OpenAI-compatible client to:

```
Base URL:  http://<host>:8000/v1
API Key:   any non-empty string (e.g. "uva")
```

### Example: curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer uva" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Example: OpenWebUI / LibreChat

Set the **OpenAI base URL** to `http://<this-host>:8000/v1` and use any string as the API key.

## Notes

- `temperature` and `max_tokens` are accepted but ignored — UvA's API does not expose these parameters.
- Each request creates a new chat thread on the UvA platform.
- Multi-turn history is injected into the system prompt since UvA's API only accepts single user turns.

## License

MIT
