"""A client for the 0G model router, with the four failure modes that matter guarded.

A model-backed paid service can return HTTP 200 with a worthless deliverable in at least four ways,
all of them observed in production on sibling services:

1. **Reasoning as the answer.** `minimax-m3` emits a `<think>` block before its response. Returned
   verbatim, the customer receives the model's internal monologue instead of the deliverable.
2. **Truncated JSON.** The response hits the token ceiling mid-object and `json.loads` fails, or
   worse, succeeds on a prefix that parses.
3. **Wrong shape rendered as nulls.** The model returns valid JSON with different keys, the code
   `.get()`s the ones it wanted, and every field comes back null while the status stays 200.
4. **Chain timeout overrun.** Each fallback model gets the full timeout, so a chain of five blows
   through the x402 300-second ceiling and the caller's socket dies.

So: reasoning is stripped, JSON is validated against required keys before it is accepted, and the
whole chain shares one deadline rather than one timeout each.

The router gates availability **per model**, not per account — a 403 BALANCE_INSUFFICIENT on one
model says nothing about the others, which is why there is a chain at all.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Sequence

import httpx

# Imported for the side effect: `core.config` loads `.env` into the environment at import time, and
# this module reads its key straight from there. Without it, a process that happens not to import
# config reports the model as unconfigured while the key is sitting in the file.
from . import config as _config  # noqa: F401

log = logging.getLogger("horos.llm")

# The `<think>…</think>` block reasoning models emit. Also handles the unterminated case, where the
# response was cut off mid-thought and there is no closing tag at all.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(RuntimeError):
    def __init__(self, message: str, code: str = "model_unavailable"):
        super().__init__(message)
        self.code = code


def strip_reasoning(text: str) -> str:
    """Remove a reasoning block. The single most common way to ship a worthless 200."""
    out = _THINK.sub("", text or "")
    out = _THINK_OPEN.sub("", out)          # unterminated: the whole tail was reasoning
    return out.strip()


def extract_json(text: str) -> Any:
    """Parse JSON out of a model response, refusing anything that is not complete.

    Tries the whole string, then a fenced block, then the outermost balanced braces. A partial parse
    is never accepted — truncated JSON that happens to parse as a prefix is the second failure mode
    and it produces a deliverable missing exactly the fields the caller paid for.
    """
    cleaned = _FENCE.sub("", strip_reasoning(text)).strip()
    if not cleaned:
        raise LLMError("the model returned nothing but reasoning", "empty_response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"the model's JSON is malformed or was truncated: {e}",
                           "malformed_json") from e
    raise LLMError("the model did not return JSON", "not_json")


class LLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 models: Sequence[str] | None = None, timeout: float = 90.0):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL")
                         or "https://router-api.0g.ai/v1").rstrip("/")
        self._key = api_key or os.environ.get("MINIMAX_API_KEY") or os.environ.get("LLM_API_KEY")
        default = os.environ.get("LLM_MODEL", "minimax-m3")
        self.models = list(models or [m.strip() for m in os.environ.get(
            "LLM_MODEL_CHAIN", f"{default},glm-5.2,deepseek-v4-pro,qwen3.7-max").split(",")
            if m.strip()])
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._key)

    def complete(self, system: str, user: str, *, max_tokens: int = 2000,
                 temperature: float = 0.2, deadline: float | None = None) -> tuple[str, str]:
        """Return (text, model_used). Raises LLMError if the whole chain fails or time runs out.

        `deadline` is an absolute wall-clock time shared by every attempt. Without it a five-model
        chain at 90 seconds each can spend 450 seconds against an x402 ceiling of 300 and the caller
        sees a dropped connection rather than an error.
        """
        if not self.configured:
            raise LLMError("no model API key is configured", "not_configured")
        deadline = deadline or (time.time() + self.timeout)
        errors: list[str] = []

        for model in self.models:
            remaining = deadline - time.time()
            if remaining < 5:
                raise LLMError(
                    f"the time budget was exhausted after trying {len(errors)} model(s): "
                    f"{'; '.join(errors[-2:])}", "deadline_exceeded")
            try:
                with httpx.Client(timeout=min(remaining, self.timeout)) as client:
                    resp = client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._key}",
                                 "Content-Type": "application/json"},
                        json={"model": model, "temperature": temperature,
                              "max_tokens": max_tokens,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}]})
                if resp.status_code != 200:
                    errors.append(f"{model}: HTTP {resp.status_code} {resp.text[:120]}")
                    continue
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                text = strip_reasoning((choice.get("message") or {}).get("content") or "")
                finish = choice.get("finish_reason")
                if not text:
                    errors.append(f"{model}: returned only reasoning, no answer")
                    continue
                if finish == "length":
                    # Truncation is reported, not silently returned. A cut-off deliverable that
                    # looks complete is worse than an error.
                    errors.append(f"{model}: response hit the token ceiling and was truncated")
                    continue
                return text, model
            except httpx.HTTPError as e:
                errors.append(f"{model}: {type(e).__name__}: {e}")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                errors.append(f"{model}: unreadable response: {e}")

        raise LLMError(f"every model in the chain failed: {'; '.join(errors)}", "all_models_failed")

    def json_complete(self, system: str, user: str, *, required: Sequence[str],
                      max_tokens: int = 2000, deadline: float | None = None) -> tuple[dict, str]:
        """Complete, parse as JSON, and verify the keys the caller actually needs are present.

        Validating the shape is what stops failure mode 3. Without it a model that answers with
        different key names produces a deliverable of nulls at HTTP 200.
        """
        text, model = self.complete(system, user, max_tokens=max_tokens, deadline=deadline)
        data = extract_json(text)
        if not isinstance(data, dict):
            raise LLMError(f"the model returned a {type(data).__name__}, not an object",
                           "wrong_shape")
        missing = [k for k in required if k not in data]
        if missing:
            raise LLMError(
                f"the model's response is missing required field(s): {', '.join(missing)}. "
                f"It returned: {', '.join(list(data)[:10])}", "wrong_shape")
        return data, model


_SHARED: LLM | None = None


def shared() -> LLM:
    global _SHARED
    if _SHARED is None:
        _SHARED = LLM()
    return _SHARED
