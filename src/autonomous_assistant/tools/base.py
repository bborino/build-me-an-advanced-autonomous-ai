from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autonomous_assistant.config import AssistantSettings
from autonomous_assistant.models import ToolResult, ToolSpec


@dataclass(slots=True)
class ToolRuntimeContext:
    workspace_root: Path
    settings: AssistantSettings
    session_id: str


class BaseTool(ABC):
    spec: ToolSpec

    @abstractmethod
    def execute(self, tool_input: dict[str, Any], context: ToolRuntimeContext) -> ToolResult:
        raise NotImplementedError

