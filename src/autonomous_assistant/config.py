from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _to_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


class AssistantSettings(BaseModel):
    workspace_root: Path
    data_dir: Path
    memory_db_path: Path

    execution_profile: str = "auto"
    max_iterations: int = 8
    max_task_attempts: int = 2
    max_tool_steps_per_task: int = 4
    tool_output_char_limit: int = 5000
    web_search_results: int = 5
    shell_timeout_seconds: int = 30
    http_timeout_seconds: float = 30.0
    provider_timeout_seconds: float = 60.0
    local_model_max_tokens: int = 450
    shell_enabled: bool = True

    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_reasoning_model: str = "gpt-4.1-mini"
    openai_code_model: str = "gpt-4.1-mini"
    openai_synthesis_model: str = "gpt-4.1-mini"

    anthropic_api_key: str | None = None
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_reasoning_model: str = "claude-3-5-sonnet-latest"
    anthropic_synthesis_model: str = "claude-3-5-sonnet-latest"

    brave_search_api_key: str | None = None

    @classmethod
    def from_env(cls, workspace_root: str | Path | None = None) -> "AssistantSettings":
        root = Path(workspace_root or os.getcwd()).resolve()
        env_values = _read_env_file(root / ".env")

        def get(name: str, default: str | None = None) -> str | None:
            return os.getenv(name, env_values.get(name, default))

        data_dir = Path(get("ASSISTANT_DATA_DIR", str(root / ".assistant_data"))).resolve()
        memory_db_path = Path(
            get("ASSISTANT_MEMORY_DB", str(data_dir / "memory.sqlite3"))
        ).resolve()

        settings = cls(
            workspace_root=root,
            data_dir=data_dir,
            memory_db_path=memory_db_path,
            execution_profile=(get("ASSISTANT_EXECUTION_PROFILE", "auto") or "auto").lower(),
            max_iterations=_to_int(get("ASSISTANT_MAX_ITERATIONS"), 8),
            max_task_attempts=_to_int(get("ASSISTANT_MAX_TASK_ATTEMPTS"), 2),
            max_tool_steps_per_task=_to_int(get("ASSISTANT_MAX_TOOL_STEPS"), 4),
            tool_output_char_limit=_to_int(get("ASSISTANT_TOOL_OUTPUT_LIMIT"), 5000),
            web_search_results=_to_int(get("ASSISTANT_WEB_SEARCH_RESULTS"), 5),
            shell_timeout_seconds=_to_int(get("ASSISTANT_SHELL_TIMEOUT"), 30),
            http_timeout_seconds=_to_float(get("ASSISTANT_HTTP_TIMEOUT"), 30.0),
            provider_timeout_seconds=_to_float(
                get("ASSISTANT_PROVIDER_TIMEOUT"), 60.0
            ),
            local_model_max_tokens=_to_int(
                get("ASSISTANT_LOCAL_MODEL_MAX_TOKENS"), 450
            ),
            shell_enabled=_to_bool(get("ASSISTANT_SHELL_ENABLED"), True),
            openai_api_key=get("OPENAI_API_KEY"),
            openai_base_url=get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            or "https://api.openai.com/v1",
            openai_reasoning_model=get("OPENAI_REASONING_MODEL", "gpt-4.1-mini")
            or "gpt-4.1-mini",
            openai_code_model=get("OPENAI_CODE_MODEL", "gpt-4.1-mini")
            or "gpt-4.1-mini",
            openai_synthesis_model=get("OPENAI_SYNTHESIS_MODEL", "gpt-4.1-mini")
            or "gpt-4.1-mini",
            anthropic_api_key=get("ANTHROPIC_API_KEY"),
            anthropic_base_url=get(
                "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
            )
            or "https://api.anthropic.com",
            anthropic_reasoning_model=get(
                "ANTHROPIC_REASONING_MODEL", "claude-3-5-sonnet-latest"
            )
            or "claude-3-5-sonnet-latest",
            anthropic_synthesis_model=get(
                "ANTHROPIC_SYNTHESIS_MODEL", "claude-3-5-sonnet-latest"
            )
            or "claude-3-5-sonnet-latest",
            brave_search_api_key=get("BRAVE_SEARCH_API_KEY"),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_db_path.parent.mkdir(parents=True, exist_ok=True)

    def use_local_fast_path(self) -> bool:
        if self.execution_profile == "local_fast":
            return True
        if self.execution_profile == "balanced":
            return False

        base_url = self.openai_base_url.lower()
        local_markers = ("localhost:11434", "127.0.0.1:11434", "0.0.0.0:11434")
        return bool(self.openai_api_key) and any(marker in base_url for marker in local_markers)
