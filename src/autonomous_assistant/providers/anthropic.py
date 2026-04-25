from __future__ import annotations

from typing import Any

import httpx

from autonomous_assistant.models import CompletionRequest, CompletionResult
from autonomous_assistant.providers.base import LanguageModelProvider


class AnthropicProvider(LanguageModelProvider):
    def __init__(self, api_key: str, base_url: str, timeout_seconds: float) -> None:
        super().__init__(name="anthropic")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(self, model_name: str, request: CompletionRequest) -> CompletionResult:
        prompt = request.prompt
        if request.response_format == "json":
            prompt = (
                f"{request.prompt}\n\nReturn a valid JSON object only. "
                "Do not include code fences or additional commentary."
            )

        payload: dict[str, Any] = {
            "model": model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
            )

        response.raise_for_status()
        data = response.json()
        text = _extract_anthropic_text(data)
        return CompletionResult(text=text, raw=data)


def _extract_anthropic_text(data: dict[str, Any]) -> str:
    content = data.get("content") or []
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts).strip()

