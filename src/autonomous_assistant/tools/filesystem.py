from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from autonomous_assistant.models import ToolResult, ToolSpec
from autonomous_assistant.tools.base import BaseTool, ToolRuntimeContext
from autonomous_assistant.utils import truncate_text


class FilesystemTool(BaseTool):
    spec = ToolSpec(
        name="filesystem",
        description="List, read, search, and write files inside the workspace root.",
        risk_level="medium",
        input_schema={
            "operation": "list | read | search | write",
            "path": "relative path inside the workspace",
        },
    )

    def execute(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        operation = str(tool_input.get("operation", "")).strip().lower()
        try:
            if operation == "list":
                return self._list(tool_input, context)
            if operation == "read":
                return self._read(tool_input, context)
            if operation == "search":
                return self._search(tool_input, context)
            if operation == "write":
                return self._write(tool_input, context)
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Filesystem operation failed.",
                error=f"Unsupported operation: {operation}",
            )
        except Exception as exc:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Filesystem operation failed.",
                error=str(exc),
            )

    def _resolve(self, user_path: str | None, context: ToolRuntimeContext) -> Path:
        candidate = (context.workspace_root / (user_path or ".")).resolve()
        workspace_root = context.workspace_root.resolve()
        candidate_norm = os.path.normcase(str(candidate))
        workspace_norm = os.path.normcase(str(workspace_root))
        if os.path.commonpath([candidate_norm, workspace_norm]) != workspace_norm:
            raise ValueError("Filesystem access is restricted to the workspace root.")
        return candidate

    def _list(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        target = self._resolve(tool_input.get("path"), context)
        recursive = bool(tool_input.get("recursive", False))
        limit = max(1, min(int(tool_input.get("limit", 50)), 200))
        if not target.exists():
            raise FileNotFoundError(f"{target} does not exist.")

        if recursive:
            entries = list(target.rglob("*"))
        else:
            entries = list(target.iterdir())

        items: list[dict[str, Any]] = []
        for entry in entries[:limit]:
            items.append(
                {
                    "path": self._relative_to_workspace(entry, context),
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )

        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            summary=f"Listed {len(items)} entries from {self._relative_to_workspace(target, context)}.",
            content={"entries": items},
        )

    def _read(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        target = self._resolve(tool_input.get("path"), context)
        start_line = int(tool_input.get("start_line", 1))
        end_line = int(tool_input.get("end_line", 200))
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"{target} is not a readable file.")

        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        start_index = max(start_line - 1, 0)
        end_index = max(end_line, start_index + 1)
        selected = lines[start_index:end_index]
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_index + 1)
        )
        relative = self._relative_to_workspace(target, context)

        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            summary=f"Read {len(selected)} lines from {relative}.",
            content={
                "path": str(relative),
                "text": truncate_text(numbered, context.settings.tool_output_char_limit),
                "total_lines": len(lines),
            },
        )

    def _search(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        root = self._resolve(tool_input.get("path"), context)
        query = str(tool_input.get("query", "")).strip().lower()
        limit = max(1, min(int(tool_input.get("limit", 20)), 50))
        if not query:
            raise ValueError("Search query is required.")

        matches: list[dict[str, Any]] = []
        for file_path in root.rglob("*"):
            if len(matches) >= limit:
                break
            if not file_path.is_file():
                continue
            if file_path.stat().st_size > 1_000_000:
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            lowered = text.lower()
            if query not in lowered:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query in line.lower():
                    matches.append(
                        {
                            "path": self._relative_to_workspace(file_path, context),
                            "line_number": line_number,
                            "line": truncate_text(line, 240),
                        }
                    )
                    if len(matches) >= limit:
                        break

        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            summary=f"Found {len(matches)} filesystem matches for '{query}'.",
            content={"matches": matches},
        )

    def _write(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        target = self._resolve(tool_input.get("path"), context)
        mode = str(tool_input.get("mode", "overwrite")).strip().lower()
        content = str(tool_input.get("content", ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with target.open("a", encoding="utf-8", errors="ignore") as handle:
                handle.write(content)
        else:
            target.write_text(content, encoding="utf-8")

        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            summary=f"Wrote {len(content)} characters to {self._relative_to_workspace(target, context)}.",
            content={"path": self._relative_to_workspace(target, context)},
        )

    def _relative_to_workspace(self, path: Path, context: ToolRuntimeContext) -> str:
        absolute_path = os.path.realpath(path)
        workspace_root = os.path.realpath(context.workspace_root)
        return os.path.relpath(absolute_path, start=workspace_root)
