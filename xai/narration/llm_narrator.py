# Copyright 2025 - LLM Narrator for Attention-Grounded XRL Pipeline
"""
Module 11: LLMNarrator

Makes an HTTP API call to OpenRouter (or compatible endpoint) and returns
the narration string.  This is the only module that makes a network call.
"""

import json
import os
from typing import Any, Dict, Optional

import urllib.request
import urllib.error


def narrate(
    system_prompt: str,
    user_prompt: str,
    config: Dict[str, Any],
) -> str:
    """
    Call the LLM API and return the narration string.

    Args:
        system_prompt: System-level instructions.
        user_prompt: User-level prompt with report data.
        config: Pipeline config dict — uses the ``llm`` sub-dict:
            - ``provider``: currently only ``"openrouter"``
            - ``base_url``: API endpoint
            - ``model``: model identifier (e.g. ``"qwen/qwen3-4b"``)
            - ``max_tokens``: max response length
            - ``api_key_env``: environment variable name holding the API key

    Returns:
        Narration string from the LLM, or an error placeholder.
    """
    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url", "https://openrouter.ai/api/v1/chat/completions")
    model = llm_cfg.get("model", "qwen/qwen3-4b")
    max_tokens = llm_cfg.get("max_tokens", 256)
    api_key_env = llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        return (
            f"[LLMNarrator] ERROR: API key not found in env var '{api_key_env}'. "
            f"Set it with: export {api_key_env}=sk-..."
        )

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        req = urllib.request.Request(
            base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        # OpenRouter / OpenAI-compatible response format
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return "[LLMNarrator] WARNING: Empty response from API."

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        return f"[LLMNarrator] HTTP {e.code}: {error_body[:200]}"
    except Exception as e:
        return f"[LLMNarrator] ERROR: {e}"
