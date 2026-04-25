from __future__ import annotations

from autonomous_assistant.config import AssistantSettings
from autonomous_assistant.models import ToolResult, ToolSpec
from autonomous_assistant.tools.base import BaseTool, ToolRuntimeContext
from autonomous_assistant.tools.filesystem import FilesystemTool
from autonomous_assistant.tools.http_request import HttpRequestTool
from autonomous_assistant.tools.python_exec import PythonExecutionTool
from autonomous_assistant.tools.shell import ShellTool
from autonomous_assistant.tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.spec.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(
        self, name: str, tool_input: dict, context: ToolRuntimeContext
    ) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                ok=False,
                tool_name=name,
                summary="Tool execution failed.",
                error=f"Unknown tool: {name}",
            )
        return tool.execute(tool_input=tool_input, context=context)


def create_default_tool_registry(settings: AssistantSettings) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(HttpRequestTool())
    registry.register(FilesystemTool())
    registry.register(PythonExecutionTool())
    if settings.shell_enabled:
        registry.register(ShellTool())
    return registry

