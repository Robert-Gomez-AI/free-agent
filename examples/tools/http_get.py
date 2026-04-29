"""Sample tool: fetch a URL over HTTP/HTTPS using only the stdlib.

To enable, copy this file into ./free_agent_tools/ next to where you run the CLI,
or use it as inspiration for your own tools.
"""
from __future__ import annotations

import urllib.error
import urllib.request

from langchain_core.tools import tool

_MAX_BYTES = 8000


@tool
def http_get(url: str, timeout: int = 10) -> str:
    """Fetch the body of `url` over HTTP/HTTPS.

    Use this when the user asks to read a public webpage, fetch a JSON endpoint,
    or check the contents of a URL. Returns the response body (truncated to
    8 KB) or an error message — never raises.
    """
    if not url.startswith(("http://", "https://")):
        return f"refusing to fetch non-http(s) url: {url!r}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "free-agent/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(_MAX_BYTES + 1)
            text = data.decode("utf-8", errors="replace")
        truncated = " [...truncated]" if len(data) > _MAX_BYTES else ""
        return f"{resp.status} {resp.reason}\n\n{text[:_MAX_BYTES]}{truncated}"
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code} {exc.reason}"
    except urllib.error.URLError as exc:
        return f"URL error: {exc.reason}"
    except TimeoutError:
        return f"timeout after {timeout}s"
    except Exception as exc:
        return f"http_get failed: {exc}"
