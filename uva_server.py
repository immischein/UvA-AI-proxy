"""
UvA AI Chat — OpenAI-compatible API server
===========================================
Exposes:  GET  /v1/models
          POST /v1/chat/completions   (streaming + non-streaming)

Run:
    pip install fastapi uvicorn requests
    python uva_server.py          # default: 0.0.0.0:8000

    or pass env vars:
    SESSION_TOKEN=<token> HOST=0.0.0.0 PORT=8000 python uva_server.py

External tools (OpenWebUI, LibreChat, etc.) should point their
"OpenAI base URL" to  http://<this-host>:8000/v1
and use any non-empty string as the API key.
"""

import json
import os
import random
import string
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Config (all values from environment variables) ────────────────────────────

BASE_URL = os.environ.get("UVA_BASE_URL", "https://aichatacc.uva.nl")

SESSION_TOKEN = os.environ.get("SESSION_TOKEN")
if not SESSION_TOKEN:
    raise RuntimeError("SESSION_TOKEN environment variable is required")

_default_models = "gpt-5.1,gpt-4.1,gpt-4o,o3,o4-mini,claude-sonnet-4.6,claude-opus-4.6,claude-haiku-4.5"
KNOWN_MODELS = os.environ.get("KNOWN_MODELS", _default_models).split(",")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="UvA AI Chat – OpenAI proxy", version="1.0.0")

# Allow all origins so browser-based tools can reach this server directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────


def _content_to_str(content: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Normalise OpenAI message content to a plain string.
    Accepts both the legacy string form and the newer multipart block form:
        [{"type": "text", "text": "..."}, ...]
    Only "text" blocks are extracted; image/tool blocks are skipped.
    """
    if isinstance(content, str):
        return content
    parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    return "".join(parts)


class Message(BaseModel):
    role: str                                    # "system" | "user" | "assistant"
    content: Union[str, List[Dict[str, Any]]]    # string or multipart blocks

    @property
    def text(self) -> str:
        return _content_to_str(self.content)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = None   # accepted but ignored — UvA doesn't expose it
    max_tokens: Optional[int] = None      # accepted but ignored
    # Any extra OpenAI fields are silently ignored via model_config below.

    model_config = {"extra": "allow"}


# ── UvA helpers ───────────────────────────────────────────────────────────────


def _random_id(length: int = 16) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _uva_headers() -> dict:
    return {
        "Cookie": f"__Secure-next-auth.session-token={SESSION_TOKEN}",
        "Content-Type": "application/json",
    }


def _build_system_prompt(messages: List[Message]) -> tuple[Optional[str], str, List[Message]]:
    """
    Split messages into (system_prompt, prior_history_text, remaining_user_messages).

    Because UvA's API only accepts user turns (we cannot inject assistant text
    into the thread), we replay prior conversation turns as context inside the
    system prompt so the model has full history.

    Returns:
        system_prompt  – combined system text (original + history block), or None
        last_user_text – the final user message to actually send
        skipped        – list of messages we folded into the system prompt
    """
    system_parts: List[str] = []
    history: List[Message] = []
    last_user_text: Optional[str] = None

    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.text.strip())
        else:
            history.append(msg)

    # The last message must be a user turn (OpenAI contract).
    if not history or history[-1].role != "user":
        raise ValueError("Last message must be from the user.")

    last_user_text = history[-1].text
    prior = history[:-1]  # everything before the final user turn

    if prior:
        lines = ["=== Conversation history (continue from here) ==="]
        for m in prior:
            label = "User" if m.role == "user" else "Assistant"
            lines.append(f"{label}: {m.text}")
        lines.append("=== End of history ===")
        system_parts.append("\n".join(lines))

    combined_system = "\n\n".join(system_parts) if system_parts else None
    return combined_system, last_user_text


def _uva_payload(thread_id: str, text: str, *, is_new_chat: bool,
                 system_prompt: Optional[str], model: Optional[str]) -> dict:
    overrides: dict = {}
    if system_prompt:
        overrides["systemPrompt"] = system_prompt
    if model:
        overrides["model"] = model

    return {
        "id": thread_id,
        "message": {
            "parts": [{"type": "text", "text": text}],
            "id": _random_id(16),
            "role": "user",
        },
        "flags": {
            "studyMode": False,
            "enforceInternetSearch": False,
            "enforceArtifactCreation": False,
            "enforceImageGeneration": False,
            "regenerate": False,
            "continue": False,
            "isNewChat": is_new_chat,
        },
        "overrides": overrides,
        "requestTime": datetime.now(timezone.utc).isoformat(),
    }


def send_message(thread_id: str, text: str, system_prompt: Optional[str],
                 model: Optional[str]) -> requests.Response:
    """POST to UvA and return the full response — mirrors uva_chat.py exactly."""
    payload = _uva_payload(thread_id, text, is_new_chat=True,
                           system_prompt=system_prompt, model=model)
    resp = requests.post(
        f"{BASE_URL}/api/v1/chat",
        headers=_uva_headers(),
        json=payload,
    )
    if not resp.ok:
        raise HTTPException(status_code=502,
                            detail=f"UvA API error {resp.status_code}: {resp.text[:300]}")
    return resp


def extract_text_from_sse(response: requests.Response) -> List[str]:
    """
    Parse UvA's SSE body into a list of text-delta strings — mirrors uva_chat.py.
    Returns individual deltas (not joined) so the streaming path can emit them
    as separate OpenAI chunks.
    """
    deltas: List[str] = []
    for line in response.text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except Exception:
            continue
        if data.get("type") == "text-delta":
            deltas.append(data.get("delta", ""))
    return deltas


# ── SSE generators ────────────────────────────────────────────────────────────


def _openai_chunk(completion_id: str, model: str, delta_content: str,
                  finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _generate_stream(deltas: List[str], completion_id: str,
                     model: str) -> Generator[str, None, None]:
    """
    Emit OpenAI SSE chunks from a list of text-delta strings.
    The full UvA response is already collected (uva_chat.py style),
    so we replay the deltas as individual chunks.
    """
    # First chunk: role announcement (OpenAI convention)
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    for delta in deltas:
        if delta:
            yield _openai_chunk(completion_id, model, delta)

    yield _openai_chunk(completion_id, model, "", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── API endpoints ─────────────────────────────────────────────────────────────


@app.get("/v1/models")
@app.get("/models")
def list_models():
    now = int(time.time())
    data = [
        {
            "id": m,
            "object": "model",
            "created": now,
            "owned_by": "uva",
        }
        for m in KNOWN_MODELS
    ]
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id}")
@app.get("/models/{model_id}")
def get_model(model_id: str):
    if model_id not in KNOWN_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "uva"}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, http_req: Request):
    try:
        system_prompt, last_user_text = _build_system_prompt(req.messages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = req.model or "gpt-4o"
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    thread_id = _random_id(38)

    uva_resp = send_message(thread_id, last_user_text, system_prompt, model)
    deltas = extract_text_from_sse(uva_resp)

    # ── Streaming ──────────────────────────────────────────────────────────────
    if req.stream:
        return StreamingResponse(
            _generate_stream(deltas, completion_id, model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming ──────────────────────────────────────────────────────────
    full_text = "".join(deltas).strip()
    prompt_tokens = sum(len(m.text.split()) for m in req.messages)
    completion_tokens = len(full_text.split())

    return JSONResponse({
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "models": KNOWN_MODELS}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting UvA AI Chat OpenAI-compatible server on http://{HOST}:{PORT}")
    print(f"Base URL for external tools: http://{HOST}:{PORT}/v1")
    print(f"Available models: {', '.join(KNOWN_MODELS)}")
    uvicorn.run(app, host=HOST, port=PORT)
