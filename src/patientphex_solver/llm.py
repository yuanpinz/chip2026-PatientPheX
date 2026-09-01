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
# Qwen3.5-9B is the strongest model in the supplied API guide that stays
# within the competition's 10B-parameter limit.
DEFAULT_MODEL = "modelE6-9-local"
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_INDEX_LIST_RE = re.compile(r'"entity_indices"\s*:\s*\[([^]]*)', re.DOTALL)
_ASSIGNMENT_ENTRY_RE = re.compile(r'"([^"\\]+)"\s*:\s*\[([^]]*)', re.DOTALL)
_ENTITY_LIST_RE = re.compile(r'"entities"\s*:\s*\[', re.DOTALL)


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
            values = [int(value) for value in re.findall(r"\d+", index_match.group(1))]
            return {"entity_indices": values}
        assignments_start = re.search(r'"assignments"\s*:\s*\{', candidate)
        if assignments_start:
            object_assignments: dict[str, list[dict[str, Any]]] = {}
            assignment_text = candidate[assignments_start.end() :]
            patient_entries = list(
                re.finditer(r'"([^"\\]+)"\s*:\s*\[', assignment_text)
            )
            for entry_index, entry in enumerate(patient_entries):
                end = (
                    patient_entries[entry_index + 1].start()
                    if entry_index + 1 < len(patient_entries)
                    else len(assignment_text)
                )
                rows: list[dict[str, Any]] = []
                for match in re.finditer(
                    r"\{[^{}]*\}", assignment_text[entry.end() : end], re.DOTALL
                ):
                    try:
                        value = json.loads(match.group(0))
                    except json.JSONDecodeError:
                        continue
                    if (
                        isinstance(value, dict)
                        and "passage_index" in value
                        and "text" in value
                    ):
                        rows.append(value)
                if rows:
                    object_assignments[entry.group(1)] = rows
            if object_assignments:
                return {"assignments": object_assignments}

            assignments: dict[str, list[int]] = {}
            for match in _ASSIGNMENT_ENTRY_RE.finditer(
                candidate, assignments_start.end()
            ):
                values = [int(value) for value in re.findall(r"\d+", match.group(2))]
                assignments[match.group(1)] = values
            if assignments:
                return {"assignments": assignments}
        # Entity discovery responses are often truncated after an opening or
        # partially completed array. Keep complete object entries and ignore
        # the unfinished tail so one bad passage does not abort a run.
        if _ENTITY_LIST_RE.search(candidate):
            objects = []
            for match in re.finditer(r"\{[^{}]*\}", candidate, re.DOTALL):
                fragment = match.group(0)
                try:
                    value = json.loads(fragment)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and "text" in value:
                    objects.append(value)
            return {"entities": objects}
        starts = [
            position
            for position in (candidate.find("{"), candidate.find("["))
            if position >= 0
        ]
        if not starts:
            raise
        start = min(starts)
        # Some models append a second JSON value or a prose explanation after
        # the requested object. Decode the first complete value instead of
        # extending to the final closing delimiter and producing "Extra data".
        try:
            value, _ = json.JSONDecoder().raw_decode(candidate[start:])
            return value
        except json.JSONDecodeError:
            pass
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
        self.endpoint = endpoint or os.environ.get(
            "PATIENTPHEX_API_ENDPOINT", DEFAULT_ENDPOINT
        )
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = retries

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        material = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        digest = hashlib.sha256(material).hexdigest()
        return self.cache_dir / self.model / f"{digest}.json"

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        return {
            "model_name": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 8000,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str:
        payload = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
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
                    text = ((body.get("choices") or [{}])[0].get("message") or {}).get(
                        "content"
                    )
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"model returned no text: {body}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Separate writers must not share a temporary path when model
                # experiments run concurrently with the same cache key.
                temporary = cache_path.with_name(
                    f".{cache_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
                )
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
        raise RuntimeError(
            f"model call failed after {self.retries} attempts"
        ) from last_error

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 8000,
    ) -> Any:
        payload = self._payload(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            enable_thinking=False,
        )
        cache_path = self._cache_path(payload)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                return parse_json_response(
                    self.chat(messages, max_tokens=max_tokens)
                )
            except (json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                cache_path.unlink(missing_ok=True)
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"model returned invalid JSON after {self.retries} attempts"
        ) from last_error
