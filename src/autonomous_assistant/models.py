from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from autonomous_assistant.utils import utc_now_iso


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


class IntentProfile(BaseModel):
    objective: str
    deliverables: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    preferred_output: str = "concise structured result"


class PlanTask(BaseModel):
    id: str
    title: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    priority: int = 1
    preferred_modes: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    result_summary: str | None = None
    last_error: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str
    risk_level: str = "low"
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    summary: str
    content: Any | None = None
    error: str | None = None
    requires_confirmation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolDecision(BaseModel):
    action: Literal["tool", "complete", "replan"]
    working_summary: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    completion_note: str | None = None
    follow_up_title: str | None = None
    follow_up_description: str | None = None


class EvaluationResult(BaseModel):
    success: bool
    confidence: float = 0.5
    next_action: Literal["complete", "retry", "replan", "halt"] = "complete"
    note: str = ""


class ExecutionStep(BaseModel):
    step_number: int
    kind: Literal["plan", "model", "tool", "evaluation", "synthesis"]
    status: str
    summary: str
    task_id: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now_iso)


class ModelProfile(BaseModel):
    alias: str
    provider: str
    model_name: str
    capabilities: list[str] = Field(default_factory=list)
    priority: int = 0


class RunReport(BaseModel):
    session_id: str
    goal: str
    intent: IntentProfile
    plan: list[PlanTask]
    steps: list[ExecutionStep] = Field(default_factory=list)
    final_output: str = ""
    status: RunStatus = RunStatus.COMPLETED
    created_at: str = Field(default_factory=utc_now_iso)
    completed_at: str | None = None


class CompletionRequest(BaseModel):
    system: str
    prompt: str
    temperature: float = 0.2
    response_format: Literal["text", "json"] = "text"
    max_tokens: int = 1200


class CompletionResult(BaseModel):
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)

