from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx
import re

from autonomous_assistant.models import ToolResult, ToolSpec
from autonomous_assistant.tools.base import BaseTool, ToolRuntimeContext
from autonomous_assistant.utils import strip_html_tags


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class WebSearchTool(BaseTool):
    spec = ToolSpec(
        name="web_search",
        description="Search the web for public information.",
        risk_level="low",
        input_schema={"query": "best vector databases", "max_results": 5},
    )

    def execute(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        query = str(tool_input.get("query", "")).strip()
        max_results = max(1, min(int(tool_input.get("max_results", context.settings.web_search_results)), 10))
        if not query:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Web search failed.",
                error="A search query is required.",
            )

        try:
            results: list[dict[str, Any]] = []
            errors: list[str] = []
            successful_backend = False
            if context.settings.brave_search_api_key:
                try:
                    results = self._search_brave(query, max_results, context)
                    successful_backend = True
                except Exception as exc:
                    errors.append(str(exc))

            if not results:
                try:
                    results = self._search_duckduckgo(query, max_results, context)
                    successful_backend = True
                except Exception as exc:
                    errors.append(str(exc))

            if not results:
                try:
                    results = self._search_wikipedia(query, max_results, context)
                    successful_backend = True
                except Exception as exc:
                    errors.append(str(exc))

            if not results and errors:
                if successful_backend:
                    return ToolResult(
                        ok=True,
                        tool_name=self.spec.name,
                        summary=f"No web results were found for '{query}'.",
                        content={"query": query, "results": [], "warnings": errors[:2]},
                    )
                return ToolResult(
                    ok=False,
                    tool_name=self.spec.name,
                    summary="Web search failed.",
                    error=" | ".join(errors[:2]),
                )

            return ToolResult(
                ok=True,
                tool_name=self.spec.name,
                summary=f"Found {len(results)} web results for '{query}'.",
                content={"query": query, "results": results},
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Web search failed.",
                error=str(exc),
            )

    def _search_brave(
        self, query: str, max_results: int, context: ToolRuntimeContext
    ) -> list[dict[str, Any]]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": context.settings.brave_search_api_key or "",
        }
        params = {"q": query, "count": max_results}
        with httpx.Client(timeout=context.settings.http_timeout_seconds) as client:
            response = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
            )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in (data.get("web", {}) or {}).get("results", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("description"),
                }
            )
        return results

    def _search_duckduckgo(
        self, query: str, max_results: int, context: ToolRuntimeContext
    ) -> list[dict[str, Any]]:
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=context.settings.http_timeout_seconds, headers=headers) as client:
            response = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
            )
        response.raise_for_status()
        html = response.text

        link_matches = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="(.*?)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet_matches = re.findall(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>|<div[^>]+class="result__snippet"[^>]*>(.*?)</div>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )

        snippets = [
            strip_html_tags(first or second or "")
            for first, second in snippet_matches
        ]

        results: list[dict[str, Any]] = []
        for index, (raw_url, raw_title) in enumerate(link_matches[:max_results]):
            resolved_url = self._resolve_duckduckgo_url(raw_url)
            results.append(
                {
                    "title": strip_html_tags(raw_title),
                    "url": resolved_url,
                    "snippet": snippets[index] if index < len(snippets) else "",
                }
            )
        return results

    def _search_wikipedia(
        self, query: str, max_results: int, context: ToolRuntimeContext
    ) -> list[dict[str, Any]]:
        with httpx.Client(timeout=context.settings.http_timeout_seconds) as client:
            response = client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "utf8": 1,
                    "srlimit": max_results,
                },
            )
        response.raise_for_status()
        data = response.json()
        results = []
        for item in data.get("query", {}).get("search", [])[:max_results]:
            title = item.get("title", "")
            results.append(
                {
                    "title": title,
                    "url": f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                    "snippet": strip_html_tags(item.get("snippet", "")),
                }
            )
        return results

    @staticmethod
    def _resolve_duckduckgo_url(raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if parsed.path.startswith("/l/"):
            query = parse_qs(parsed.query)
            uddg = query.get("uddg")
            if uddg:
                return uddg[0]
        return raw_url
