from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from autonomous_assistant.models import ToolResult, ToolSpec
from autonomous_assistant.tools.base import BaseTool, ToolRuntimeContext
from autonomous_assistant.utils import truncate_text


BLOCKED_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdiskpart\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b",
    r"\bRemove-Item\b",
    r"\bClear-Disk\b",
    r"\bSet-ExecutionPolicy\b",
]

CONFIRM_PATTERNS = [
    r"\bpip\s+install\b",
    r"\bwinget\b",
    r"\bchoco\b",
    r"\bapt\b",
    r"\byum\b",
    r"\bdnf\b",
    r"\bsc\s+",
    r"\bsystemctl\b",
    r"\bservice\b",
]


class ShellTool(BaseTool):
    spec = ToolSpec(
        name="shell",
        description="Run shell commands inside the workspace with safety checks.",
        risk_level="high",
        input_schema={"command": "git status", "timeout_seconds": 10},
    )

    def execute(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        if not context.settings.shell_enabled:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Shell execution is disabled.",
                error="The shell tool is disabled in settings.",
            )

        command = str(tool_input.get("command", "")).strip()
        timeout_seconds = max(
            1,
            min(
                int(tool_input.get("timeout_seconds", context.settings.shell_timeout_seconds)),
                180,
            ),
        )
        if not command:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Shell execution failed.",
                error="A command is required.",
            )

        lowered = command.lower()
        if any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in BLOCKED_PATTERNS):
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Shell command blocked by safety policy.",
                error="The requested shell command is considered destructive and will not be executed.",
            )

        if any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in CONFIRM_PATTERNS):
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Shell command requires confirmation.",
                error="This shell command may change the local system state.",
                requires_confirmation=True,
            )

        try:
            completed = subprocess.run(
                self._build_command(command),
                cwd=str(context.workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            stdout = truncate_text(completed.stdout, context.settings.tool_output_char_limit)
            stderr = truncate_text(completed.stderr, context.settings.tool_output_char_limit)
            ok = completed.returncode == 0
            return ToolResult(
                ok=ok,
                tool_name=self.spec.name,
                summary=(
                    f"Shell command completed with exit code {completed.returncode}."
                    if ok
                    else f"Shell command failed with exit code {completed.returncode}."
                ),
                content={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": completed.returncode,
                },
                error=None if ok else stderr or f"Command failed: {lowered}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Shell command timed out.",
                error=f"The command exceeded {timeout_seconds} seconds.",
            )

    @staticmethod
    def _build_command(command: str) -> list[str]:
        if os.name == "nt":
            return ["powershell", "-NoProfile", "-Command", command]
        return ["bash", "-lc", command]

