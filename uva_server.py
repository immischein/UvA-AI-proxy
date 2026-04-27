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

USAGE_FILE = Path(os.environ.get("USAGE_FILE", "usage_stats.json"))

# ── Pricing table (USD per 1M tokens) ────────────────────────────────────────
# Input / output prices from public API pricing pages.
# Used only for the /savings endpoint; UvA access is free to the user.
_PRICING: dict[str, tuple[float, float]] = {
    # model-id          (input $/MTok, output $/MTok)
    "claude-sonnet-4.6":  (3.00,  15.00),
    "claude-opus-4.6":    (5.00,  25.00),
    "claude-haiku-4.5":   (1.00,   5.00),
    "gpt-4o":             (2.50,  10.00),
    "gpt-4.1":            (2.00,   8.00),
    "gpt-5":              (1.25,  10.00),
    "gpt-5.1":            (1.25,  10.00),
    "gpt-5-mini":         (0.25,   2.00),
    "gpt-5-nano":         (0.05,   0.40),
    "gpt-oss-120b":       (0.50,   0.50),  # avg. via OpenRouter/DeepInfra
    "mistral-large":      (0.50,   1.50),
}
_PRICING_FALLBACK = (2.00, 8.00)  # mid-range default for unknown models
EUR_PER_USD = 0.88  # fixed exchange rate — update as needed


def _tokens_from_words(words: int) -> int:
    return max(1, int(words * 1.35))  # rough word → token conversion


# ── Usage tracking ────────────────────────────────────────────────────────────

def _load_usage() -> dict[str, dict[str, int]]:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_usage(stats: dict[str, dict[str, int]]) -> None:
    try:
        USAGE_FILE.write_text(json.dumps(stats, indent=2))
    except Exception as exc:
        print(f"[usage] warning: could not save stats: {exc}")


_usage_stats: dict[str, dict[str, int]] = _load_usage()


def _record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    entry = _usage_stats.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0})
    entry["prompt_tokens"] += prompt_tokens
    entry["completion_tokens"] += completion_tokens
    entry["requests"] += 1
    _save_usage(_usage_stats)


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


# MIME types that UvA's upload endpoint accepts natively.
_ACCEPTED_MIME_TYPES: frozenset[str] = frozenset({
    "text/plain",
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    # MS Office
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
})

# Extensions that are text-based even when mimetypes guesses wrong.
_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp",
    ".java", ".kt", ".kts", ".scala",
    ".go", ".rs", ".rb", ".php", ".swift", ".cs", ".fs",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".json", ".jsonc", ".json5",
    ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".html", ".htm", ".xhtml", ".svg",
    ".css", ".scss", ".sass", ".less",
    ".md", ".mdx", ".rst", ".txt", ".tex",
    ".sql", ".graphql", ".gql",
    ".r", ".rmd", ".jl", ".lua", ".pl", ".pm",
    ".dockerfile", ".makefile", ".mk",
    ".gitignore", ".gitattributes", ".editorconfig",
    ".lock",  # package-lock.json, Cargo.lock, etc.
    ".log",
})


def _coerce_to_accepted(
    raw: bytes, filename: str, media_type: str
) -> tuple[bytes, str, str]:
    """
    Convert *any* file to a format accepted by UvA's upload endpoint.

    Returns (new_bytes, new_filename, new_media_type).

    Strategy
    --------
    1. Already accepted  → return as-is.
    2. Text-based file   → wrap in a .txt with a header showing the original
                           filename so the model knows what it is looking at.
    3. Binary file       → produce a hex dump as .txt so the model can still
                           inspect the content (useful for small binaries).
    """
    if media_type in _ACCEPTED_MIME_TYPES:
        return raw, filename, media_type

    stem = Path(filename)
    ext = stem.suffix.lower()

    # ── Try to decode as UTF-8 text ───────────────────────────────────────────
    is_text = ext in _TEXT_EXTENSIONS
    if not is_text:
        # Heuristic: if the first 8 KB has no null bytes it's probably text.
        sample = raw[:8192]
        is_text = b"\x00" not in sample

    new_filename = stem.stem + "_" + (stem.suffix.lstrip(".") or "file") + ".txt"

    if is_text:
        try:
            text_content = raw.decode("utf-8", errors="replace")
        except Exception:
            text_content = raw.decode("latin-1", errors="replace")
        header = f"# File: {filename}\n# (converted to plain text for upload)\n\n"
        new_bytes = (header + text_content).encode("utf-8")
        print(f"[coerce] {filename} → {new_filename} (text wrap)")
    else:
        # Binary: produce a hex dump (limit to 512 KB of source to keep it sane)
        limit = 512 * 1024
        truncated = raw[:limit]
        lines: list[str] = []
        for i in range(0, len(truncated), 16):
            chunk = truncated[i : i + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            asc_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{i:08x}  {hex_part:<47}  |{asc_part}|")
        if len(raw) > limit:
            lines.append(f"... (truncated; original size {len(raw)} bytes)")
        header = f"# File: {filename}\n# Binary hex dump (converted for upload)\n\n"
        new_bytes = (header + "\n".join(lines)).encode("utf-8")
        print(f"[coerce] {filename} → {new_filename} (hex dump, {len(raw)} bytes)")

    return new_bytes, new_filename, "text/plain"


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
    raw, upload_name, media_type = _coerce_to_accepted(raw, path.name, media_type)
    resp = requests.post(
        f"{BASE_URL}/api/document/upload/stream",
        headers=headers,
        files={"file": (upload_name, raw, media_type)},
        data={"chatThreadId": thread_id, "chatState": "true"},
        timeout=60,
    )
    if resp.ok:
        print(f"[upload] {upload_name} ({len(raw)} bytes) → thread {thread_id[:8]}…")
    else:
        print(f"[upload] warning: UvA returned {resp.status_code} for {upload_name}")


def _upload_attachments(messages: List[Message], thread_id: str) -> None:
    """
    Scan the last user message for file attachments and upload them to UvA.

    Handles three formats:
    - @"/path/to/file" or @/path/to/file text references
    - OpenAI image_url blocks with data: URIs
    - Anthropic image/document blocks with base64 source
    """
    headers = {k: v for k, v in _uva_headers().items() if k.lower() != "content-type"}

    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if last_user is None:
        return

    content = last_user.content if isinstance(last_user.content, list) else []

    # Build a single string from all text blocks (or the raw string content)
    all_text = (
        " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        if content
        else (last_user.content if isinstance(last_user.content, str) else "")
    )

    # ── 1. @"path" or @path references ────────────────────────────────────────
    cwd = _extract_cwd(messages)  # Claude Code's working directory

    for m in _AT_FILE_RE.finditer(all_text):
        filepath = m.group(1) or m.group(2)
        path = Path(filepath)

        # Resolve relative paths against Claude Code's working directory
        if not path.is_absolute():
            path = cwd / path

        key = (thread_id, str(path))
        if path.is_file() and key not in _already_uploaded:
            _already_uploaded.add(key)
            try:
                _upload_file_to_uva(path, thread_id, headers)
            except Exception as exc:
                print(f"[upload] warning: {path.name}: {exc}")
        elif not path.is_file():
            print(f"[upload] warning: file not found: {path}")

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
            raw, filename, media_type = _coerce_to_accepted(raw, filename, media_type)
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
    "Whenever you create, generate, or save a file — code, scripts, documents, data, configurations, "
    "or any other written output meant to be saved — always use the artifact creation "
    "feature (create_artifact) so the file is automatically downloaded to the user's machine. "
    "The name of the artifact should be identical to the name of the file being created."
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
        _record_usage(
            model,
            _tokens_from_words(sum(len(m.text.split()) for m in req.messages)),
            _tokens_from_words(len("".join(deltas).split())),
        )
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
    prompt_tokens = _tokens_from_words(sum(len(m.text.split()) for m in req.messages))
    completion_tokens = _tokens_from_words(len(full_text.split()))
    _record_usage(model, prompt_tokens, completion_tokens)

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

    media_type = file.content_type or mimetypes.guess_type(file.filename or "")[0] or "application/octet-stream"
    content, upload_name, media_type = _coerce_to_accepted(content, file.filename or "upload", media_type)

    resp = requests.post(
        f"{BASE_URL}/api/document/upload/stream",
        headers=headers,
        files={"file": (upload_name, content, media_type)},
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
        "uploaded_as": upload_name,
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


# ── Savings tracker ───────────────────────────────────────────────────────────


@app.get("/savings")
def savings() -> dict[str, Any]:
    """
    Returns how much money you would have spent if you'd used each model's
    commercial API instead of UvA's free portal.
    """
    breakdown: list[dict[str, Any]] = []
    total_cost = 0.0
    total_requests = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for model, counts in _usage_stats.items():
        p_tok = counts.get("prompt_tokens", 0)
        c_tok = counts.get("completion_tokens", 0)
        reqs  = counts.get("requests", 0)
        in_price, out_price = _PRICING.get(model, _PRICING_FALLBACK)
        cost = (p_tok * in_price + c_tok * out_price) / 1_000_000
        total_cost += cost
        total_requests += reqs
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        breakdown.append({
            "model": model,
            "requests": reqs,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "estimated_cost_usd": round(cost, 4),
            "pricing_usd_per_mtok": {"input": in_price, "output": out_price},
        })

    breakdown.sort(key=lambda x: x["estimated_cost_usd"], reverse=True)

    return {
        "total_saved_usd": round(total_cost, 4),
        "total_saved_eur": round(total_cost * EUR_PER_USD, 4),
        "total_requests": total_requests,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "note": "Token counts are estimated (~1.35 tokens/word). Costs reflect public API pricing.",
        "breakdown_by_model": breakdown,
    }


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
