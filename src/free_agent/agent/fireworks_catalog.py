"""Live Fireworks model catalog (OpenAI-compatible `/models` endpoint)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def list_models(api_key: str, base_url: str = "https://api.fireworks.ai/inference") -> list[dict[str, Any]]:
    """Return serverless chat models available to this key.

    Each row: {name, supports_tools, context_length}. Embedding/reranker
    models and pure routers like `auto` are filtered out.
    Raises RuntimeError on network/auth failure.
    """
    req = urllib.request.Request(
        f"{base_url.rstrip('/').removesuffix('/v1')}/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "free-agent"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Fireworks /models returned HTTP {exc.code} — is the API key valid?") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(f"cannot reach Fireworks: {exc}") from exc

    rows = []
    for m in data.get("data", []):
        mid = m.get("id", "")
        if not mid.startswith("accounts/") or any(k in mid for k in ("embedding", "reranker")):
            continue
        rows.append({
            "name": mid,
            "supports_tools": bool(m.get("supports_tools")),
            "context_length": m.get("context_length"),
        })
    rows.sort(key=lambda r: (not r["supports_tools"], r["name"]))
    return rows
