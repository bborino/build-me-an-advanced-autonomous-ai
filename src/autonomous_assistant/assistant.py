from __future__ import annotations

from typing import Any

from autonomous_assistant.config import AssistantSettings
from autonomous_assistant.memory import MemoryStore
from autonomous_assistant.models import (
    EvaluationResult,
    ExecutionStep,
    PlanTask,
    RunReport,
    RunStatus,
    TaskStatus,
    ToolResult,
)
from autonomous_assistant.reasoning import ReasoningEngine
from autonomous_assistant.router import ModelRegistry, ModelRouter
from autonomous_assistant.tools.base import ToolRuntimeContext
from autonomous_assistant.tools.registry import ToolRegistry, create_default_tool_registry
from autonomous_assistant.utils import dump_json, truncate_text, utc_now_iso


class AutonomousAssistant:
    def __init__(self, settings: AssistantSettings | None = None) -> None:
        self.settings = settings or AssistantSettings.from_env()
        self.settings.ensure_directories()
        self.memory = MemoryStore(self.settings.memory_db_path)
        self.models = ModelRegistry(self.settings)
        self.router = ModelRouter(self.models)
        self.tools: ToolRegistry = create_default_tool_registry(self.settings)
        self.reasoning = ReasoningEngine(self.models, self.router)

    def capabilities(self) -> dict[str, Any]:
        return {
            "tools": [spec.model_dump() for spec in self.tools.specs()],
            "models": [profile.model_dump() for profile in self.models.list_profiles()],
            "safety": {
                "shell_destructive_commands_blocked": True,
                "sensitive_shell_changes_require_confirmation": True,
                "filesystem_restricted_to_workspace": True,
            },
        }

    def run(self, goal: str, session_id: str | None = None) -> RunReport:
        self._step_counter = 0
        session_id = self.memory.start_session(goal, session_id=session_id)
        recent_memory = self.memory.recent_notes(session_id, limit=10)
        intent = self.reasoning.interpret_goal(goal, recent_memory)
        plan = self.reasoning.create_plan(
            goal=goal,
            intent=intent,
            tool_descriptions=[spec.model_dump() for spec in self.tools.specs()],
        )

        report = RunReport(session_id=session_id, goal=goal, intent=intent, plan=plan)
        report.steps.append(
            self._step(
                kind="plan",
                status="ok",
                summary="Interpreted the user goal.",
                detail=intent.model_dump(),
            )
        )
        report.steps.append(
            self._step(
                kind="plan",
                status="ok",
                summary=f"Generated a plan with {len(plan)} tasks.",
                detail={"tasks": [task.model_dump() for task in plan]},
            )
        )

        self.memory.add_note(session_id, "intent", dump_json(intent.model_dump()))
        self.memory.add_note(
            session_id,
            "plan",
            dump_json([task.model_dump() for task in plan]),
        )

        execution_notes: list[str] = []
        iterations = 0
        while iterations < self.settings.max_iterations:
            task = self._next_ready_task(report.plan)
            if task is None:
                break

            iterations += 1
            task.attempts += 1
            task.status = TaskStatus.IN_PROGRESS

            completion_note, tool_history, blocked = self._execute_task(
                goal=goal,
                task=task,
                plan=report.plan,
                session_id=session_id,
                report=report,
            )

            evaluation, evaluation_model = self.reasoning.evaluate_task(
                task=task,
                completion_note=completion_note,
                tool_history=tool_history,
            )
            report.steps.append(
                self._step(
                    kind="evaluation",
                    status="ok" if evaluation.success else "review",
                    summary=evaluation.note or "Evaluated the task outcome.",
                    task_id=task.id,
                    detail={
                        "evaluation": evaluation.model_dump(),
                        "model": evaluation_model.model_dump() if evaluation_model else None,
                    },
                )
            )

            self._apply_evaluation(task, completion_note, evaluation, blocked)
            if task.result_summary:
                execution_notes.append(f"{task.title}: {task.result_summary}")
                self.memory.add_note(
                    session_id,
                    "task_result",
                    task.result_summary,
                    metadata={"task_id": task.id, "status": task.status.value},
                )
            elif task.last_error:
                self.memory.add_note(
                    session_id,
                    "task_issue",
                    task.last_error,
                    metadata={"task_id": task.id, "status": task.status.value},
                )

        final_output, synthesis_model = self.reasoning.synthesize_output(
            goal=goal,
            intent=intent,
            plan=report.plan,
            execution_notes=execution_notes,
        )
        report.final_output = final_output
        report.completed_at = utc_now_iso()
        report.status = self._resolve_run_status(report.plan)
        report.steps.append(
            self._step(
                kind="synthesis",
                status="ok",
                summary="Prepared the final response.",
                detail={"model": synthesis_model.model_dump() if synthesis_model else None},
            )
        )
        self.memory.add_note(
            session_id,
            "final_output",
            final_output,
            metadata={"status": report.status.value},
        )
        return report

    def _execute_task(
        self,
        goal: str,
        task: PlanTask,
        plan: list[PlanTask],
        session_id: str,
        report: RunReport,
    ) -> tuple[str, list[ToolResult], bool]:
        tool_history: list[ToolResult] = []
        blocked = False
        completion_note = ""

        for _ in range(self.settings.max_tool_steps_per_task):
            relevant_memory = self.memory.search(session_id, task.title, limit=6)
            decision, model = self.reasoning.next_action(
                goal=goal,
                task=task,
                plan=plan,
                tool_descriptions=[spec.model_dump() for spec in self.tools.specs()],
                relevant_memory=relevant_memory,
                tool_history=tool_history,
            )
            report.steps.append(
                self._step(
                    kind="model",
                    status="ok",
                    summary=decision.working_summary or "Selected the next action.",
                    task_id=task.id,
                    detail={
                        "decision": decision.model_dump(),
                        "model": model.model_dump() if model else None,
                    },
                )
            )

            if decision.action == "complete":
                completion_note = decision.completion_note or decision.working_summary
                break

            if decision.action == "replan":
                completion_note = decision.completion_note or "The task requested replanning."
                break

            if not decision.tool_name:
                completion_note = "The model did not provide a valid tool selection."
                break

            result = self.tools.execute(
                name=decision.tool_name,
                tool_input=decision.tool_input,
                context=ToolRuntimeContext(
                    workspace_root=self.settings.workspace_root,
                    settings=self.settings,
                    session_id=session_id,
                ),
            )
            tool_history.append(result)
            report.steps.append(
                self._step(
                    kind="tool",
                    status="ok" if result.ok else "error",
                    summary=result.summary,
                    task_id=task.id,
                    detail={
                        "tool_name": result.tool_name,
                        "tool_input": decision.tool_input,
                        "result": result.model_dump(),
                    },
                )
            )
            self.memory.add_note(
                session_id,
                "tool_call",
                result.summary,
                metadata={
                    "task_id": task.id,
                    "tool_name": result.tool_name,
                    "ok": result.ok,
                },
            )

            if result.requires_confirmation:
                blocked = True
                completion_note = result.error or result.summary
                break
            if result.ok:
                completion_note = self._summarize_tool_result(result)
            else:
                completion_note = result.error or result.summary

        if not completion_note and tool_history:
            completion_note = tool_history[-1].summary

        return completion_note, tool_history, blocked

    def _apply_evaluation(
        self,
        task: PlanTask,
        completion_note: str,
        evaluation: EvaluationResult,
        blocked: bool,
    ) -> None:
        if blocked or evaluation.next_action == "halt":
            task.status = TaskStatus.BLOCKED
            task.last_error = completion_note or evaluation.note
            return

        if evaluation.success:
            task.status = TaskStatus.COMPLETED
            task.result_summary = truncate_text(completion_note or evaluation.note, 1200)
            task.last_error = None
            return

        task.last_error = truncate_text(completion_note or evaluation.note, 1200)
        if task.attempts < self.settings.max_task_attempts and evaluation.next_action in {"retry", "replan"}:
            task.status = TaskStatus.PENDING
            return

        task.status = TaskStatus.FAILED

    def _next_ready_task(self, plan: list[PlanTask]) -> PlanTask | None:
        completed = {task.id for task in plan if task.status == TaskStatus.COMPLETED}
        candidates = [
            task
            for task in plan
            if task.status == TaskStatus.PENDING
            and all(dependency in completed for dependency in task.dependencies)
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item.priority, item.id))[0]

    def _resolve_run_status(self, plan: list[PlanTask]) -> RunStatus:
        statuses = {task.status for task in plan}
        if statuses == {TaskStatus.COMPLETED}:
            return RunStatus.COMPLETED
        if TaskStatus.BLOCKED in statuses:
            return RunStatus.BLOCKED
        if TaskStatus.COMPLETED in statuses:
            return RunStatus.PARTIAL
        return RunStatus.FAILED

    @staticmethod
    def _summarize_tool_result(result: ToolResult) -> str:
        if isinstance(result.content, dict):
            if "results" in result.content:
                results = result.content.get("results", [])
                if results:
                    top = results[0]
                    return truncate_text(
                        f"{result.summary} Top result: {top.get('title')} ({top.get('url')})",
                        800,
                    )
            if "text" in result.content:
                return truncate_text(
                    f"{result.summary} {result.content.get('text')}",
                    800,
                )
            if "stdout" in result.content:
                stdout = result.content.get("stdout") or ""
                return truncate_text(f"{result.summary} {stdout}", 800)
        return truncate_text(result.summary, 800)

    def _step(
        self,
        kind: str,
        status: str,
        summary: str,
        task_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> ExecutionStep:
        return ExecutionStep(
            step_number=0,
            kind=kind,  # type: ignore[arg-type]
            status=status,
            summary=summary,
            task_id=task_id,
            detail=detail or {},
        ).model_copy(
            update={"step_number": self._next_step_number()}
        )

    def _next_step_number(self) -> int:
        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        self._step_counter += 1
        return self._step_counter
