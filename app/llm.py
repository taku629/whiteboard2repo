"""Thin OpenAI-compatible client for Meta's open-weight Muse Spark Open.

Any provider that serves `muse-spark-open` behind an OpenAI-compatible
endpoint works (Nebius Token Factory, Baseten, Fireworks, Modal, OpenRouter...).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

BASE_URL = os.getenv("MUSE_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
MODEL = os.getenv("MUSE_MODEL", "meta-models/muse-spark-open")
API_KEY = os.getenv("MUSE_API_KEY", "")


class LLMError(RuntimeError):
    pass


def chat(messages: list[dict[str, Any]], *, tools: list[dict] | None = None,
         max_tokens: int = 4096, temperature: float = 0.2,
         json_mode: bool = False, retries: int = 3) -> dict[str, Any]:
    """Call Muse Spark Open and return the assistant message dict."""
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.post(f"{BASE_URL}/chat/completions", json=body,
                           headers=headers, timeout=180)
            if r.status_code >= 500 or r.status_code == 429:
                raise LLMError(f"{r.status_code}: {r.text[:300]}")
            if r.status_code >= 400:
                raise LLMError(f"{r.status_code}: {r.text[:500]}")
            data = r.json()
            msg = data["choices"][0]["message"]
            msg["_usage"] = data.get("usage", {})
            msg["_finish"] = data["choices"][0].get("finish_reason")
            return msg
        except (httpx.HTTPError, LLMError, KeyError) as e:  # pragma: no cover
            last = e
            if isinstance(e, LLMError) and not str(e)[:3] in ("429", "500", "502", "503", "504"):
                if attempt == retries - 1:
                    break
            time.sleep(1.5 * (attempt + 1))
    raise LLMError(str(last))


def extract_json(text: str) -> Any:
    """Pull the first JSON object out of a model reply (tolerates ``` fences)."""
    if not text:
        raise ValueError("empty reply")
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts[1::2]:
            p = p.split("\n", 1)[1] if p.lstrip().startswith(("json", "JSON")) else p
            try:
                return json.loads(p)
            except json.JSONDecodeError:
                continue
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return json.loads(t[start:end + 1])
    raise ValueError("no JSON found")
