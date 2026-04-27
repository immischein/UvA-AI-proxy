# UvA AI Proxy

An OpenAI-compatible API proxy for [aichat.uva.nl](https://aichat.uva.nl) — the University of Amsterdam's AI chat platform. It lets you connect any tool that speaks the OpenAI API (OpenWebUI, LibreChat, OpenClaude, Cursor, etc.) to UvA's AI models using your existing UvA session. Primarily tested with OpenClaude.

## Features

- **OpenAI-compatible endpoints** — `GET /v1/models` and `POST /v1/chat/completions`
- **Streaming & non-streaming** responses
- **Persistent conversation threads** — follow-up messages continue in the same UvA chat instead of opening a new one
- **Model alias resolution** — short names like `sonnet`, `haiku`, `gpt`, `mistral`, `oss`, etc. are mapped to full UvA model IDs automatically (see [Model Aliases](#model-aliases))
- **Actual model reporting** — the response reflects the model UvA actually ran, not just what was requested
- **File uploads** — attach files with `@file.pdf` in OpenClaude; they are automatically uploaded to UvA and included in the conversation context
- **Artifact saving** — when UvA generates a code file or document, it is automatically saved to your working directory
- **Title-request interception** — title-generation requests from OpenClaude are answered locally so they don't create extra UvA chats
- **Multiple models** — All models that are available on the UvA AI Chat (as of now: GPT-5.1, Claude Haiku 4.5, Claude Sonnet 4.6, GPT-4.1, GPT-4o, GPT-5, GPT-5-mini, GPT-5-nano, GPT-OSS-120b, Mistral Large)
- **CORS enabled** — browser-based tools can connect directly

## Requirements

- Python 3.10+
- An UvA account (see [Getting a Session Token](#getting-a-session-token))

## Installation

```bash
# Clone the repository
git clone https://github.com/immischein/UvA-AI-proxy.git
cd UvA-AI-proxy/

# Create a Python virtual environment (venv)
python3 -m venv .venv

# Activate the virual environment
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
playwright install chromium
```
## Quickstart

Start the session token extractor (opens a browser window) and start the server. Make sure that you are inside your Python venv.

```bash
bash ./start.sh
```

## Getting a Session Token

Use the included `extract.py` script to obtain your `__Secure-next-auth.session-token` from the UvA AI chat site.

```bash
# Opens a visible browser — log in manually, token is printed automatically
python extract.py

# Headless mode (only works if SSO auto-login succeeds)
python extract.py --headless

# Save the token to a file
python extract.py --output .session_token
```

## Running the Server

```bash
SESSION_TOKEN=<your-token> python uva_server.py
```

Or if you saved the token to `.session_token`:

```bash
SESSION_TOKEN=$(cat .session_token) python uva_server.py
```

The server starts on `http://0.0.0.0:8000` by default.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SESSION_TOKEN` | *(required)* | Your UvA NextAuth session token |
| `HOST` | `0.0.0.0` | Host to bind to |
| `PORT` | `8000` | Port to listen on |
| `UVA_BASE_URL` | `https://aichat.uva.nl` | UvA API base URL |
| `KNOWN_MODELS` | *(see below)* | Comma-separated list of models to expose |
| `UPLOAD_DIR` | `uploads/` | Directory for locally stored uploaded files |
| `USAGE_FILE` | `usage_stats.json` | Path to the JSON file where token usage stats are persisted |

**Default models:** `gpt-5.1, claude-haiku-4.5, claude-sonnet-4.6, gpt-4.1, gpt-4o, gpt-5, gpt-5-mini, gpt-5-nano, gpt-oss-120b, mistral-large`

## Model Aliases

You can use short alias names instead of full model IDs:

| Alias | Resolves to |
|---|---|
| `sonnet`, `claude-sonnet` | `claude-sonnet-4.6` |
| `haiku`, `claude-haiku` | `claude-haiku-4.5` |
| `gpt` | `gpt-5.1` |
| `gpt-mini`, `mini` | `gpt-5-mini` |
| `gpt-nano`, `nano` | `gpt-5-nano` |
| `gpt-oss`, `oss` | `gpt-oss-120b` |
| `mistral` | `mistral-large` |

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
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Example: OpenClaude

Set the base URL to `http://localhost:8000` in your OpenClaude config. Then use it normally — file attachments with `@file.pdf` are uploaded automatically, and any code/document artifacts the model generates are saved directly to your working directory.

## File Uploads

### From OpenClaude CLI

Just reference a file with `@`:

```
@"/path/to/document.pdf"  summarise this
```

The proxy detects the path, uploads the file to UvA's document API, and includes it in the conversation context automatically.

### Inline image / document blocks

The proxy also handles base64-encoded content blocks in the OpenAI (`image_url` with a `data:` URI) and Anthropic (`image` / `document` with a `base64` source) message formats. These are uploaded to UvA automatically — no special syntax needed when using a client that sends them natively.

### Via the upload endpoint

```bash
# Upload a file
curl -X POST http://localhost:8000/upload -F "file=@report.pdf"

# List uploaded files
curl http://localhost:8000/files

# Download a file
curl http://localhost:8000/download/report.pdf -o report.pdf
```

Or use the included `file_client.py`:

```bash
python file_client.py upload report.pdf
python file_client.py list
python file_client.py download report.pdf
```

## Other Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok", "models": [...]}` — useful for liveness checks |
| `GET /savings` | Returns estimated cost savings vs. commercial API pricing (see [Money Counter](#money-counter)) |
| `GET /files` | Lists files in the upload directory |
| `GET /download/{filename}` | Downloads a previously uploaded file |

## Artifact Saving

When the model generates a file (e.g. you ask it to *"write a helloworld.py"*), the proxy intercepts the `create_artifact` event and saves the file directly to your current working directory. No manual copy-paste needed.

## Notes

- `temperature` and `max_tokens` are accepted but ignored.
- Conversation thread state is kept in memory — restarting the server starts fresh threads.
- Multi-turn history is injected into the system prompt since UvA's API only accepts single user turns.

## Money Counter

Track how much you would have spent if you'd used each model's commercial API instead of UvA's free portal.

Savings data is available as JSON at `http://localhost:8000/savings`. You can also surface it in the OpenClaude status line by adding the following snippet to your `statusline.sh`:

```bash
#!/usr/bin/env bash

uva_result=$(curl -sf --max-time 1 http://localhost:8000/savings 2>/dev/null)
saved=$(echo "$uva_result" | jq -r '.total_saved_eur // empty' 2>/dev/null)
if [ -n "$saved" ]; then
    echo "You saved €$(printf '%.2f' "$saved") using UvA AI"
fi
```

For instructions on adding entries to the status line, see the [Anthropic documentation — Manually configure a status line](https://code.claude.com/docs/en/statusline).

> **Note:** Savings are an estimate based on public API pricing and should be taken with a grain of salt.

## License

GPLv3
