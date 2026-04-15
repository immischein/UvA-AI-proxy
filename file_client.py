"""
file_client.py — Upload / download files to/from the UvA AI proxy server
=========================================================================
Usage:
    python file_client.py upload [FILE ...]        # upload specific files
    python file_client.py upload --dir PATH        # upload all matching files in a directory
    python file_client.py download FILENAME        # download a file by name
    python file_client.py list                     # list files on the server

Configuration (environment variables or edit the defaults below):
    SERVER_URL   — base URL of the proxy server (default: http://localhost:8000)
    API_TOKEN    — bearer token if your server requires auth (default: none)
"""

import argparse
import os
import sys
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN")  # set if your server uses auth

# Only upload these extensions when scanning a directory (empty = everything)
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".csv", ".json"}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _headers() -> dict:
    h = {}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def _should_upload(filename: str) -> bool:
    if not ALLOWED_EXTENSIONS:
        return True
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── Commands ──────────────────────────────────────────────────────────────────


def upload_file(filepath: str) -> bool:
    path = Path(filepath)
    if not path.is_file():
        print(f"[SKIP] Not a file: {filepath}")
        return False

    with path.open("rb") as f:
        try:
            resp = requests.post(
                f"{SERVER_URL}/upload",
                headers=_headers(),
                files={"file": (path.name, f)},
                timeout=60,
            )
            resp.raise_for_status()
            info = resp.json()
            print(f"[OK]  {path.name}  ({info.get('size', '?')} bytes)  → {info.get('download_url', '')}")
            return True
        except requests.RequestException as e:
            print(f"[ERR] {path.name}: {e}")
            return False


def upload_directory(directory: str) -> None:
    root = Path(directory)
    if not root.is_dir():
        print(f"[ERR] Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    entries = [p for p in sorted(root.iterdir()) if p.is_file() and _should_upload(p.name)]
    if not entries:
        print("No matching files found.")
        return

    print(f"Uploading {len(entries)} file(s) from: {root.resolve()}\n")
    ok = sum(upload_file(str(p)) for p in entries)
    print(f"\nDone: {ok}/{len(entries)} uploaded successfully.")


def download_file(filename: str, dest_dir: str = ".") -> None:
    dest = Path(dest_dir) / filename
    url = f"{SERVER_URL}/download/{filename}"
    try:
        resp = requests.get(url, headers=_headers(), timeout=60, stream=True)
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[OK]  Downloaded: {filename}  →  {dest.resolve()}")
    except requests.RequestException as e:
        print(f"[ERR] {filename}: {e}", file=sys.stderr)
        sys.exit(1)


def list_files() -> None:
    try:
        resp = requests.get(f"{SERVER_URL}/files", headers=_headers(), timeout=10)
        resp.raise_for_status()
        files = resp.json().get("files", [])
    except requests.RequestException as e:
        print(f"[ERR] Could not reach server: {e}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No files on server.")
        return

    print(f"{'Filename':<40} {'Size':>10}  URL")
    print("-" * 70)
    for f in files:
        print(f"{f['filename']:<40} {f['size']:>10}  {SERVER_URL}{f['download_url']}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload / download files to the UvA AI proxy server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    p_up = sub.add_parser("upload", help="Upload file(s)")
    p_up.add_argument("files", nargs="*", help="Files to upload")
    p_up.add_argument("--dir", metavar="PATH", help="Upload all matching files in this directory")

    # download
    p_dl = sub.add_parser("download", help="Download a file by name")
    p_dl.add_argument("filename", help="Filename as stored on the server")
    p_dl.add_argument("--out", metavar="DIR", default=".", help="Directory to save into (default: .)")

    # list
    sub.add_parser("list", help="List files on the server")

    args = parser.parse_args()

    if args.command == "upload":
        if args.dir:
            upload_directory(args.dir)
        elif args.files:
            ok = sum(upload_file(f) for f in args.files)
            print(f"\nDone: {ok}/{len(args.files)} uploaded successfully.")
        else:
            # Fall back to uploading matching files from cwd (mirrors the user's original script)
            upload_directory(".")

    elif args.command == "download":
        download_file(args.filename, dest_dir=args.out)

    elif args.command == "list":
        list_files()


if __name__ == "__main__":
    main()
