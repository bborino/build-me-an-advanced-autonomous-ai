from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from autonomous_assistant.models import ToolResult, ToolSpec
from autonomous_assistant.tools.base import BaseTool, ToolRuntimeContext
from autonomous_assistant.utils import truncate_text


class HttpRequestTool(BaseTool):
    spec = ToolSpec(
        name="http_request",
        description="Call HTTP APIs with GET, POST, PUT, PATCH, or DELETE.",
        risk_level="medium",
        input_schema={"method": "GET", "url": "https://example.com/api"},
    )

    def execute(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        method = str(tool_input.get("method", "GET")).upper()
        url = str(tool_input.get("url", "")).strip()
        if not url:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="HTTP request failed.",
                error="A URL is required.",
            )

        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="HTTP request failed.",
                error="Only http and https URLs are allowed.",
            )

        try:
            with httpx.Client(
                timeout=context.settings.http_timeout_seconds,
                follow_redirects=True,
            ) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=tool_input.get("headers"),
                    params=tool_input.get("params"),
                    json=tool_input.get("json"),
                    data=tool_input.get("data"),
                )
            body = self._serialize_response(response, context.settings.tool_output_char_limit)
            return ToolResult(
                ok=response.status_code < 400,
                tool_name=self.spec.name,
                summary=f"{method} {url} returned HTTP {response.status_code}.",
                content=body,
                error=None if response.status_code < 400 else truncate_text(response.text, 600),
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="HTTP request failed.",
                error=str(exc),
            )

    @staticmethod
    def _serialize_response(response: httpx.Response, limit: int) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "")
        payload: dict[str, Any] = {
            "status_code": response.status_code,
            "headers": {
                key: value
                for key, value in response.headers.items()
                if key.lower() in {"content-type", "content-length", "location"}
            },
        }
        if "application/json" in content_type:
            try:
                payload["json"] = response.json()
            except Exception:
                payload["text"] = truncate_text(response.text, limit)
        else:
            payload["text"] = truncate_text(response.text, limit)
        return payload

