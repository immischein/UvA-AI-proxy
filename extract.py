"""
extract.py — Extract the __Secure-next-auth.session-token from aichat.uva.nl

Usage:
    python extract.py                  # opens a visible browser, you log in manually
    python extract.py --headless       # headless mode (only works if credentials are
                                       # already cached / SSO auto-login succeeds)
    python extract.py --output token.txt   # also write the token to a file

Requirements:
    pip install playwright
    playwright install chromium
"""

import argparse
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    sys.exit(
        "Playwright is not installed.\n"
        "Run:  pip install playwright && playwright install chromium"
    )

TARGET_URL = "https://aichat.uva.nl"
COOKIE_NAME = "__Secure-next-auth.session-token"
# How long (ms) to wait for the cookie to appear after the page loads
COOKIE_WAIT_MS = 120_000  # 2 minutes — plenty of time for SSO / manual login


def wait_for_cookie(context, name: str, timeout_ms: int) -> str | None:
    """
    Poll the browser context's cookies until `name` appears or timeout is reached.
    Returns the cookie value, or None on timeout.
    """
    import time

    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        cookies = context.cookies(TARGET_URL)
        for c in cookies:
            if c["name"] == name:
                return c["value"]
        time.sleep(1)
    return None


def extract_token(headless: bool = False, output: str | None = None) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()

        print(f"Opening {TARGET_URL} …")
        page.goto(TARGET_URL)

        if not headless:
            print(
                "\n[ACTION REQUIRED]\n"
                "A browser window has opened. Please log in to aichat.uva.nl.\n"
                "The script will automatically continue once the session cookie appears.\n"
            )

        token = wait_for_cookie(context, COOKIE_NAME, COOKIE_WAIT_MS)

        browser.close()

    if token is None:
        sys.exit(
            f"Timed out waiting for cookie '{COOKIE_NAME}'.\n"
            "Make sure you completed the login within the allowed time."
        )

    print(f"\n✓ Session token extracted successfully.\n")
    print(f"{COOKIE_NAME}=\n{token}\n")

    if output:
        Path(output).write_text(token)
        print(f"Token written to: {output}")

    return token


def main():
    parser = argparse.ArgumentParser(
        description="Extract the NextAuth session token from aichat.uva.nl"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode (no visible window).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write the token to this file in addition to printing it.",
    )
    args = parser.parse_args()

    extract_token(headless=args.headless, output=args.output)


if __name__ == "__main__":
    main()
