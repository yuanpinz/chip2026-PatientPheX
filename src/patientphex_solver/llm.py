from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_ENDPOINT = (
    "https://test.huihaohealth.com/ai-center/x/server/api/v1/big_model/chat"
)
DEFAULT_MODEL = "modelK5"
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_INDEX_LIST_RE = re.compile(r'"entity_indices"\s*:\s*\[([^]]*)', re.DOTALL)


def parse_json_response(text: str) -> Any:
    candidate = text.strip()
    fenced = _FENCED_JSON_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # The association response is deliberately a single short integer list.
        # Recover it when a service truncates the closing JSON delimiters.
        index_match = _INDEX_LIST_RE.search(candidate)
        if index_match:
            values = [
                int(value)
                for value in re.findall(r"\d+", index_match.group(1))
            ]
            return {"entity_indices": values}
        starts = [position for position in (candidate.find("{"), candidate.find("[")) if position >= 0]
        if not starts:
            raise
        start = min(starts)
        closing = "}" if candidate[start] == "{" else "]"
        end = candidate.rfind(closing)
        if end <= start:
            raise
        return json.loads(candidate[start : end + 1])


class BigModelClient:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str | None = None,
        cache_dir: str | Path = "cache/llm",
        timeout: float = 240.0,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.endpoint = endpoint or os.environ.get("PATIENTPHEX_API_ENDPOINT", DEFAULT_ENDPOINT)
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = retries

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        material = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
        return self.cache_dir / self.model / f"{digest}.json"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 8000,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str:
        payload = {
            "model_name": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
        }
        cache_path = self._cache_path(payload)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return str(cached["text"])

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.endpoint, json=payload)
                    response.raise_for_status()
                    body = response.json()
                text = body.get("result")
                if not text:
                    text = (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"model returned no text: {body}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {"model": self.model, "text": text},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
                return text
            except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"model call failed after {self.retries} attempts") from last_error

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 8000,
    ) -> Any:
        return parse_json_response(self.chat(messages, max_tokens=max_tokens))
