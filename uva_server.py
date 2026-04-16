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

import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import string
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import requests
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Config (all values from environment variables) ────────────────────────────

BASE_URL = os.environ.get("UVA_BASE_URL", "https://aichat.uva.nl")

SESSION_TOKEN = os.environ.get("SESSION_TOKEN")
if not SESSION_TOKEN:
    raise RuntimeError("SESSION_TOKEN environment variable is required")

_default_models = "gpt-5.1,claude-haiku-4.5,claude-sonnet-4.6,gpt-4.1,gpt-4o,gpt-5,gpt-5-mini,gpt-5-nano,gpt-oss-120b,mistral-large"
KNOWN_MODELS = os.environ.get("KNOWN_MODELS", _default_models).split(",")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    Strips <system-reminder> blocks injected by Claude Code's runtime.
    """
    import re as _re
    _sysreminder_re = _re.compile(r"<system-reminder>.*?</system-reminder>\s*", _re.DOTALL)

    if isinstance(content, str):
        return _sysreminder_re.sub("", content).strip()
    parts: list[str] = []
    for block in content:
        if block.get("type") == "text":
            cleaned = _sysreminder_re.sub("", block.get("text", "")).strip()
            if cleaned:
                parts.append(cleaned)
    return "\n".join(parts)


# Short model aliases → full UvA model IDs
_MODEL_ALIASES: dict[str, str] = {
    "sonnet":       "claude-sonnet-4.6",
    "haiku":        "claude-haiku-4.5",
    "claude-sonnet": "claude-sonnet-4.6",
    "claude-haiku":  "claude-haiku-4.5",
    "gpt":           "gpt-5.1",
    "mistral":      "mistral-large",
    "gpt-oss":       "gpt-oss-120b",
    "oss":           "gpt-oss-120b",
    "gpt-mini":      "gpt-5-mini",
    "gpt-nano":      "gpt-5-nano",
    "mini":           "gpt-5-mini",
    "nano":           "gpt-5-nano",
}

def _resolve_model(model: Optional[str]) -> str:
    if not model:
        return "gpt-5.1"                      # default model if none specified; default in UvA's UI
    return _MODEL_ALIASES.get(model.lower(), model)


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
    thread_id: Optional[str] = None       # reuse a thread from a prior /upload call
    # Any extra OpenAI fields are silently ignored via model_config below.

    model_config = {"extra": "allow"}


# ── UvA helpers ───────────────────────────────────────────────────────────────


def _random_id(length: int = 16) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def _uva_headers() -> dict[str, str]:
    return {
        "Cookie": f"__Secure-next-auth.session-token={SESSION_TOKEN}",
        "Content-Type": "application/json",
    }


def _ext_for(media_type: str) -> str:
    return {
        "application/pdf": "pdf",
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "text/plain": "txt",
    }.get(media_type, "bin")


# Matches @"/path/to/file" or @/path/to/file in user text
_AT_FILE_RE = re.compile(r'@"([^"]+)"|@(\S+)')

_already_uploaded: set[tuple[str, str]] = set()  # (thread_id, filepath)

# Persistent thread registry: conversation_key → thread_id
_conversation_threads: dict[str, str] = {}


def _conversation_key(messages: List[Message]) -> str:
    """Stable key derived from the first user message only.
    The system prompt from Claude Code contains dynamic content (version numbers,
    dates, file listings) that changes between turns — so we ignore it."""
    first_user = next((m.text for m in messages if m.role == "user"), "")
    return hashlib.sha256(first_user.encode()).hexdigest()[:20]


def _get_thread_id(messages: List[Message], explicit: Optional[str]) -> tuple[str, bool]:
    """
    Returns (thread_id, is_new_chat).
    - If the conversation has prior assistant turns, reuse the stored thread.
    - Otherwise start a new thread.
    """
    if explicit:
        _conversation_threads.setdefault(_conversation_key(messages), explicit)
        return explicit, False

    key = _conversation_key(messages)
    has_history = any(m.role == "assistant" for m in messages)

    if has_history and key in _conversation_threads:
        return _conversation_threads[key], False

    thread_id = _random_id(38)
    _conversation_threads[key] = thread_id
    return thread_id, True


def _upload_file_to_uva(path: Path, thread_id: str, headers: dict[str, str]) -> None:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    raw = path.read_bytes()
    resp = requests.post(
        f"{BASE_URL}/api/document/upload/stream",
        headers=headers,
        files={"file": (path.name, raw, media_type)},
        data={"chatThreadId": thread_id, "chatState": "true"},
        timeout=60,
    )
    if resp.ok:
        print(f"[upload] {path.name} ({len(raw)} bytes) → thread {thread_id[:8]}…")
    else:
        print(f"[upload] warning: UvA returned {resp.status_code} for {path.name}")


def _upload_attachments(messages: List[Message], thread_id: str) -> None:
    """
    Scan the last user message for file attachments and upload them to UvA.

    Handles three formats:
    - @"/path/to/file" or @/path/to/file text references (Claude Code style)
    - OpenAI image_url blocks with data: URIs
    - Anthropic image/document blocks with base64 source
    """
    headers = {k: v for k, v in _uva_headers().items() if k.lower() != "content-type"}

    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if last_user is None:
        return

    content = last_user.content if isinstance(last_user.content, list) else []

    # ── 1. @"path" references in text blocks ──────────────────────────────────
    all_text = " ".join(
        b.get("text", "") for b in content if b.get("type") == "text"
    ) if content else (last_user.content if isinstance(last_user.content, str) else "")

    for m in _AT_FILE_RE.finditer(all_text):
        filepath = m.group(1) or m.group(2)
        path = Path(filepath)
        key = (thread_id, str(path))
        if path.is_file() and key not in _already_uploaded:
            _already_uploaded.add(key)
            try:
                _upload_file_to_uva(path, thread_id, headers)
            except Exception as exc:
                print(f"[upload] warning: {path.name}: {exc}")

    # ── 2. Base64 content blocks (OpenAI / Anthropic format) ──────────────────
    for block in content:
        btype = block.get("type")
        raw: Optional[bytes] = None
        media_type = "application/octet-stream"

        if btype == "image_url":
            url = block.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                header, _, data = url.partition(",")
                media_type = header.split(":")[1].split(";")[0]
                raw = base64.b64decode(data)

        elif btype in ("image", "document"):
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "application/octet-stream")
                raw = base64.b64decode(source.get("data", ""))

        if raw:
            filename = f"attachment.{_ext_for(media_type)}"
            try:
                resp = requests.post(
                    f"{BASE_URL}/api/document/upload/stream",
                    headers=headers,
                    files={"file": (filename, raw, media_type)},
                    data={"chatThreadId": thread_id, "chatState": "true"},
                    timeout=60,
                )
                if resp.ok:
                    print(f"[upload] {filename} ({len(raw)} bytes) → thread {thread_id[:8]}…")
                else:
                    print(f"[upload] warning: UvA returned {resp.status_code} for {filename}")
            except Exception as exc:
                print(f"[upload] warning: {filename}: {exc}")


_ARTIFACT_INSTRUCTION = (
    "Whenever you produce a file — code, scripts, documents, data, configurations, "
    "or any other written output meant to be saved — always use the artifact creation "
    "feature (create_artifact) so the file is automatically downloaded to the user's machine."
)


def _build_system_prompt(messages: List[Message]) -> tuple[Optional[str], str]:
    """
    Returns (system_prompt, last_user_text).
    Folds the system message and prior conversation history into a single system
    prompt, and extracts the final user message to send to UvA.
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

    system_parts.append(_ARTIFACT_INSTRUCTION)
    combined_system = "\n\n".join(system_parts) if system_parts else None
    return combined_system, last_user_text


def _uva_payload(thread_id: str, text: str, *, is_new_chat: bool,
                 system_prompt: Optional[str], model: Optional[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
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
                 model: Optional[str], *, is_new_chat: bool = True) -> requests.Response:
    """POST to UvA and return the full response."""
    payload = _uva_payload(thread_id, text, is_new_chat=is_new_chat,
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


def extract_text_from_sse(response: requests.Response,
                          messages: Optional[List[Message]] = None) -> tuple[List[str], Optional[str]]:
    """
    Parse UvA's SSE body.
    Returns (deltas, actual_model) where actual_model comes from the finish event.
    Also saves any artifact objects to the CLI's working directory.
    """
    deltas: List[str] = []
    actual_model: Optional[str] = None

    for line in response.text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except Exception:
            continue

        t = data.get("type")

        if t == "text-delta":
            deltas.append(data.get("delta", ""))

        elif t == "finish":
            actual_model = data.get("messageMetadata", {}).get("model")

        elif t == "tool-input-available" and data.get("toolName") == "create_artifact":
            inp = data.get("input", {})
            if "title" in inp and "content" in inp:
                _save_artifact(inp, messages or [])

    return deltas, actual_model


def _extract_cwd(messages: List[Message]) -> Path:
    """
    Extract the CLI's working directory from Claude Code's injected system-reminders.
    Looks for 'Primary working directory: /path' in any message content.
    Falls back to the proxy's own cwd.
    """
    _cwd_re = re.compile(r"Primary working directory:\s*(\S+)")
    for msg in messages:
        raw = msg.content if isinstance(msg.content, str) else " ".join(
            b.get("text", "") for b in msg.content if b.get("type") == "text"
        )
        m = _cwd_re.search(raw)
        if m:
            p = Path(m.group(1))
            if p.is_dir():
                return p
    return Path.cwd()


def _save_artifact(item: dict[str, Any], messages: List[Message]) -> None:
    """Save a UvA artifact object to the CLI's working directory."""
    title = item.get("title", "artifact")
    content = item.get("content", "")
    artifact_type = item.get("artifactType", "")

    filename = Path(title).name or f"artifact_{int(time.time())}"
    dest = _extract_cwd(messages) / filename
    dest.write_text(content, encoding="utf-8")
    print(f"[artifact] saved {artifact_type} → {dest}")


# ── SSE generators ────────────────────────────────────────────────────────────


def _openai_chunk(completion_id: str, model: str, delta_content: str,
                  finish_reason: Optional[str] = None) -> str:
    chunk: dict[str, Any] = {
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
    first: dict[str, Any] = {
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
def list_models() -> dict[str, Any]:
    now = int(time.time())
    data: list[dict[str, Any]] = [
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
def get_model(model_id: str) -> dict[str, str|int]:
    if model_id not in KNOWN_MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "uva"}


def _is_title_request(req: ChatCompletionRequest) -> bool:
    sys_msg = next((m for m in req.messages if m.role == "system"), None)
    if sys_msg is None:
        return False
    text = sys_msg.text
    return "sentence-case title" in text or "Generate a concise" in text


def _make_title(req: ChatCompletionRequest) -> str:
    first_user = next((m for m in req.messages if m.role == "user"), None)
    if first_user is None:
        return "New conversation"
    words = first_user.text.split()
    return " ".join(words[:6]).capitalize() or "New conversation"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, http_req: Request):
    if _is_title_request(req):
        title = _make_title(req)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = _resolve_model(req.model)
        if req.stream:
            return StreamingResponse(
                _generate_stream([title], completion_id, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return JSONResponse({
            "id": completion_id, "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": title},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(title.split()), "total_tokens": len(title.split())},
        })

    try:
        system_prompt, last_user_text = _build_system_prompt(req.messages)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = _resolve_model(req.model)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    thread_id, is_new_chat = _get_thread_id(req.messages, req.thread_id)

    # Auto-upload any file/image blocks attached to the last user message
    _upload_attachments(req.messages, thread_id)

    uva_resp = send_message(thread_id, last_user_text, system_prompt, model,
                            is_new_chat=is_new_chat)
    deltas, actual_model = extract_text_from_sse(uva_resp, req.messages)
    model = actual_model or model  # use what UvA actually ran

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


# ── File upload / download ────────────────────────────────────────────────────


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    chat_thread_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Upload a file to UvA's document API.
    Returns the thread_id (use it in subsequent /v1/chat/completions calls)
    and the raw ndjson lines from UvA so you can inspect the document reference.
    """
    thread_id = chat_thread_id or _random_id(38)
    content = await file.read()

    # Strip Content-Type so requests sets the correct multipart boundary itself
    headers = {k: v for k, v in _uva_headers().items() if k.lower() != "content-type"}

    resp = requests.post(
        f"{BASE_URL}/api/document/upload/stream",
        headers=headers,
        files={"file": (file.filename, content, file.content_type or "application/octet-stream")},
        data={"chatThreadId": thread_id, "chatState": "true"},
        timeout=60,
    )

    if not resp.ok:
        raise HTTPException(
            status_code=502,
            detail=f"UvA upload error {resp.status_code}: {resp.text[:300]}",
        )

    # Parse ndjson — each non-empty line is a JSON object
    lines: list[Any] = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            lines.append(json.loads(line))
        except Exception:
            lines.append({"raw": line})

    return {
        "thread_id": thread_id,
        "filename": file.filename,
        "size": len(content),
        "uva_response": lines,
    }


@app.get("/download/{filename}")
def download_file(filename: str):
    path = UPLOAD_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if not path.resolve().is_relative_to(UPLOAD_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return FileResponse(path, filename=filename)


@app.get("/files")
def list_files() -> dict[str, Any]:
    files: list[dict[str, Any]] = [
        {"filename": p.name, "size": p.stat().st_size, "download_url": f"/download/{p.name}"}
        for p in sorted(UPLOAD_DIR.iterdir())
        if p.is_file()
    ]
    return {"files": files}


# ── Health check ──────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str|list[str]]:
    return {"status": "ok", "models": KNOWN_MODELS}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting UvA AI Chat OpenAI-compatible server on http://{HOST}:{PORT}")
    print(f"Base URL for external tools: http://{HOST}:{PORT}/v1")
    print(f"Available models: {', '.join(KNOWN_MODELS)}")
    uvicorn.run(app, host=HOST, port=PORT)
