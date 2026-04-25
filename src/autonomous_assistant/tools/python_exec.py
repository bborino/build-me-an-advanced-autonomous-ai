from __future__ import annotations

import subprocess
import sys
from typing import Any

from autonomous_assistant.models import ToolResult, ToolSpec
from autonomous_assistant.tools.base import BaseTool, ToolRuntimeContext
from autonomous_assistant.utils import truncate_text


class PythonExecutionTool(BaseTool):
    spec = ToolSpec(
        name="python",
        description="Execute short Python snippets inside the workspace.",
        risk_level="medium",
        input_schema={"code": "print('hello world')", "timeout_seconds": 10},
    )

    def execute(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        code = str(tool_input.get("code", ""))
        timeout_seconds = max(1, min(int(tool_input.get("timeout_seconds", 10)), 120))
        if not code.strip():
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Python execution failed.",
                error="Python code is required.",
            )

        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(context.workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            stdout = truncate_text(completed.stdout, context.settings.tool_output_char_limit)
            stderr = truncate_text(completed.stderr, context.settings.tool_output_char_limit)
            ok = completed.returncode == 0
            summary = (
                f"Python snippet executed successfully with {len(stdout)} characters of stdout."
                if ok
                else f"Python snippet exited with code {completed.returncode}."
            )
            return ToolResult(
                ok=ok,
                tool_name=self.spec.name,
                summary=summary,
                content={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": completed.returncode,
                },
                error=None if ok else stderr or "Python execution failed.",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                tool_name=self.spec.name,
                summary="Python execution timed out.",
                error=f"Execution exceeded {timeout_seconds} seconds.",
            )

