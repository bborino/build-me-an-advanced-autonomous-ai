from __future__ import annotations

from typing import Any

from autonomous_assistant.models import (
    CompletionRequest,
    EvaluationResult,
    IntentProfile,
    ModelProfile,
    PlanTask,
    ToolDecision,
    ToolResult,
)
from autonomous_assistant.router import ModelRegistry, ModelRouter
from autonomous_assistant.utils import extract_json_payload, make_task_id, truncate_text


class ReasoningEngine:
    def __init__(self, models: ModelRegistry, router: ModelRouter) -> None:
        self.models = models
        self.router = router

    def interpret_goal(self, goal: str, memory_items: list[dict[str, Any]]) -> IntentProfile:
        profile = self.router.select("planning")
        if profile:
            system = (
                "You are an executive autonomous assistant. "
                "Infer the user's intent and respond with JSON only. "
                "Do not reveal chain-of-thought."
            )
            prompt = f"""
Goal:
{goal}

Relevant memory:
{memory_items}

Return a JSON object with:
- objective
- deliverables
- success_criteria
- constraints
- assumptions
- preferred_output
"""
            try:
                payload = self._complete_json(profile, system, prompt)
                return IntentProfile.model_validate(payload)
            except Exception:
                pass

        return self._heuristic_intent(goal)

    def create_plan(
        self,
        goal: str,
        intent: IntentProfile,
        tool_descriptions: list[dict[str, Any]],
    ) -> list[PlanTask]:
        profile = self.router.select("planning")
        if profile:
            system = (
                "You are an autonomous planning system. "
                "Break goals into structured, dependency-aware tasks and respond with JSON only."
            )
            prompt = f"""
Goal:
{goal}

Intent:
{intent.model_dump()}

Available tools:
{tool_descriptions}

Return a JSON object with a 'tasks' array. Each task must include:
- id
- title
- description
- dependencies
- priority
- preferred_modes

Rules:
- Create 3 to 7 tasks
- Keep dependencies explicit
- Use preferred_modes from: reasoning, web, http, code, python, filesystem, analysis, image, audio, video
- Include a verification or review task near the end
"""
            try:
                payload = self._complete_json(profile, system, prompt)
                raw_tasks = payload.get("tasks", [])
                tasks = [PlanTask.model_validate(task) for task in raw_tasks]
                return self._sanitize_plan(tasks)
            except Exception:
                pass

        return self._heuristic_plan(goal, intent)

    def next_action(
        self,
        goal: str,
        task: PlanTask,
        plan: list[PlanTask],
        tool_descriptions: list[dict[str, Any]],
        relevant_memory: list[dict[str, Any]],
        tool_history: list[ToolResult],
    ) -> tuple[ToolDecision, ModelProfile | None]:
        attempt_index = max(task.attempts - 1, 0)
        profile = self.router.select("execution", task=task, attempt_index=attempt_index)
        if profile:
            system = (
                "You are an autonomous task executor. "
                "Select exactly one next action and return JSON only. "
                "Do not reveal chain-of-thought."
            )
            prompt = f"""
Goal:
{goal}

Current task:
{task.model_dump()}

Plan state:
{[item.model_dump() for item in plan]}

Relevant memory:
{relevant_memory}

Previous tool results:
{[result.model_dump() for result in tool_history]}

Available tools:
{tool_descriptions}

Return a JSON object with:
- action: tool | complete | replan
- working_summary
- tool_name
- tool_input
- completion_note
- follow_up_title
- follow_up_description

Rules:
- If a tool is needed, choose only one tool
- If enough evidence already exists, use action=complete
- Keep working_summary concise
- Never invent tool names that are not listed
"""
            try:
                payload = self._complete_json(profile, system, prompt)
                return ToolDecision.model_validate(payload), profile
            except Exception:
                pass

        return self._heuristic_next_action(goal, task, tool_history), profile

    def evaluate_task(
        self,
        task: PlanTask,
        completion_note: str,
        tool_history: list[ToolResult],
    ) -> tuple[EvaluationResult, ModelProfile | None]:
        attempt_index = max(task.attempts - 1, 0)
        profile = self.router.select("evaluation", task=task, attempt_index=attempt_index)
        if profile:
            system = (
                "You are an autonomous evaluator. "
                "Judge whether the task succeeded and return JSON only."
            )
            prompt = f"""
Task:
{task.model_dump()}

Completion note:
{completion_note}

Tool history:
{[result.model_dump() for result in tool_history]}

Return a JSON object with:
- success
- confidence
- next_action: complete | retry | replan | halt
- note
"""
            try:
                payload = self._complete_json(profile, system, prompt)
                return EvaluationResult.model_validate(payload), profile
            except Exception:
                pass

        return self._heuristic_evaluation(task, completion_note, tool_history), profile

    def synthesize_output(
        self,
        goal: str,
        intent: IntentProfile,
        plan: list[PlanTask],
        execution_notes: list[str],
    ) -> tuple[str, ModelProfile | None]:
        profile = self.router.select("synthesis")
        if profile:
            system = (
                "You are a concise executive assistant. "
                "Summarize outcomes clearly without exposing chain-of-thought."
            )
            prompt = f"""
Goal:
{goal}

Intent:
{intent.model_dump()}

Plan:
{[task.model_dump() for task in plan]}

Execution notes:
{execution_notes}

Return a concise final response for the user that focuses on results, notable blockers, and next steps if needed.
"""
            try:
                result = self.models.complete(
                    profile,
                    CompletionRequest(
                        system=system,
                        prompt=prompt,
                        response_format="text",
                        max_tokens=900,
                    ),
                )
                return result.text.strip(), profile
            except Exception:
                pass

        return self._heuristic_synthesis(goal, plan), profile

    def _complete_json(
        self, profile: ModelProfile, system: str, prompt: str
    ) -> dict[str, Any]:
        result = self.models.complete(
            profile,
            CompletionRequest(
                system=system,
                prompt=prompt,
                response_format="json",
                max_tokens=1400,
            ),
        )
        payload = extract_json_payload(result.text)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        return payload

    def _sanitize_plan(self, tasks: list[PlanTask]) -> list[PlanTask]:
        if not tasks:
            return self._heuristic_plan("", self._heuristic_intent("")).copy()

        valid_ids: set[str] = set()
        cleaned: list[PlanTask] = []
        for index, task in enumerate(tasks):
            task_id = task.id or make_task_id(index)
            dependencies = [dependency for dependency in task.dependencies if dependency in valid_ids]
            cleaned.append(
                task.model_copy(
                    update={
                        "id": task_id,
                        "dependencies": dependencies,
                        "preferred_modes": task.preferred_modes or ["reasoning"],
                    }
                )
            )
            valid_ids.add(task_id)
        return cleaned

    @staticmethod
    def _heuristic_intent(goal: str) -> IntentProfile:
        return IntentProfile(
            objective=goal.strip() or "Complete the requested task.",
            deliverables=["A concise, structured result"],
            success_criteria=[
                "The goal is decomposed into manageable tasks",
                "Important actions are executed or clearly blocked",
                "The final answer is concise and actionable",
            ],
            constraints=[
                "Avoid destructive or high-risk actions",
                "Ask for confirmation before sensitive system changes",
            ],
            assumptions=["Use available tools only when they add clear value"],
            preferred_output="concise structured executive summary",
        )

    def _heuristic_plan(self, goal: str, intent: IntentProfile) -> list[PlanTask]:
        lowered = goal.lower()
        tasks: list[PlanTask] = [
            PlanTask(
                id="T1",
                title="Interpret goal",
                description="Confirm the desired outcome, constraints, and success criteria.",
                priority=1,
                preferred_modes=["reasoning", "analysis"],
            )
        ]

        if any(keyword in lowered for keyword in ["research", "search", "latest", "find", "compare"]):
            tasks.append(
                PlanTask(
                    id="T2",
                    title="Gather external context",
                    description="Search the web for relevant information and collect the strongest signals.",
                    dependencies=["T1"],
                    priority=2,
                    preferred_modes=["web", "analysis"],
                )
            )
        elif any(keyword in lowered for keyword in ["build", "create", "design", "implement", "develop"]):
            tasks.append(
                PlanTask(
                    id="T2",
                    title="Design execution approach",
                    description="Create the solution outline, major components, and execution strategy.",
                    dependencies=["T1"],
                    priority=2,
                    preferred_modes=["reasoning", "analysis", "code"],
                )
            )
        else:
            tasks.append(
                PlanTask(
                    id="T2",
                    title="Develop solution",
                    description="Produce the main answer or artifact for the request.",
                    dependencies=["T1"],
                    priority=2,
                    preferred_modes=["reasoning", "analysis"],
                )
            )

        tasks.append(
            PlanTask(
                id="T3",
                title="Execute primary task",
                description="Use the best available tool or model to make concrete progress on the goal.",
                dependencies=["T2"],
                priority=3,
                preferred_modes=["reasoning", "analysis", "code", "python", "filesystem"],
            )
        )
        tasks.append(
            PlanTask(
                id="T4",
                title="Verify outcome",
                description="Check the result for correctness, completeness, and safety.",
                dependencies=["T3"],
                priority=4,
                preferred_modes=["reasoning", "evaluation", "analysis"],
            )
        )
        tasks.append(
            PlanTask(
                id="T5",
                title="Prepare final response",
                description="Summarize the result for the user with clear outcomes and any follow-up.",
                dependencies=["T4"],
                priority=5,
                preferred_modes=["synthesis", "reasoning"],
            )
        )
        return tasks

    @staticmethod
    def _heuristic_next_action(
        goal: str, task: PlanTask, tool_history: list[ToolResult]
    ) -> ToolDecision:
        task_text = f"{task.title} {task.description}".lower()
        if not tool_history:
            if "web" in task.preferred_modes or any(
                keyword in task_text for keyword in ["research", "search", "latest", "find", "web"]
            ):
                return ToolDecision(
                    action="tool",
                    working_summary="Searching the web for context",
                    tool_name="web_search",
                    tool_input={"query": goal, "max_results": 5},
                )
            if "filesystem" in task.preferred_modes and any(
                keyword in task_text for keyword in ["file", "workspace", "repository", "project"]
            ):
                return ToolDecision(
                    action="tool",
                    working_summary="Inspecting workspace files",
                    tool_name="filesystem",
                    tool_input={"operation": "list", "path": ".", "recursive": False, "limit": 40},
                )
            if any(keyword in task_text for keyword in ["file", "workspace", "repository", "project"]) and any(
                keyword in task_text for keyword in ["inspect", "read", "list", "analyze"]
            ):
                return ToolDecision(
                    action="tool",
                    working_summary="Inspecting workspace files",
                    tool_name="filesystem",
                    tool_input={"operation": "list", "path": ".", "recursive": False, "limit": 40},
                )

        if tool_history and tool_history[-1].ok:
            return ToolDecision(
                action="complete",
                working_summary="Summarizing evidence",
                completion_note=truncate_text(tool_history[-1].summary, 600),
            )

        return ToolDecision(
            action="complete",
            working_summary="Completing reasoning step",
            completion_note=f"{task.title}: completed using the internal planning heuristics.",
        )

    @staticmethod
    def _heuristic_evaluation(
        task: PlanTask, completion_note: str, tool_history: list[ToolResult]
    ) -> EvaluationResult:
        if any(result.requires_confirmation for result in tool_history):
            return EvaluationResult(
                success=False,
                confidence=0.9,
                next_action="halt",
                note="The proposed action requires confirmation before execution.",
            )

        if tool_history and not any(result.ok for result in tool_history):
            return EvaluationResult(
                success=False,
                confidence=0.7,
                next_action="retry",
                note="The tools did not return a successful result yet.",
            )

        if completion_note.strip():
            return EvaluationResult(
                success=True,
                confidence=0.72,
                next_action="complete",
                note=f"{task.title} completed with a concise result.",
            )

        return EvaluationResult(
            success=False,
            confidence=0.4,
            next_action="retry",
            note="The task result was empty.",
        )

    @staticmethod
    def _heuristic_synthesis(goal: str, plan: list[PlanTask]) -> str:
        completed = [task for task in plan if task.status.value == "completed"]
        blocked = [task for task in plan if task.status.value == "blocked"]
        failed = [task for task in plan if task.status.value == "failed"]

        lines = [f"Goal: {goal}", ""]
        if completed:
            lines.append("Completed:")
            lines.extend(f"- {task.title}: {task.result_summary or 'done'}" for task in completed)
        if blocked:
            lines.append("")
            lines.append("Blocked:")
            lines.extend(f"- {task.title}: {task.last_error or 'needs confirmation'}" for task in blocked)
        if failed:
            lines.append("")
            lines.append("Failed:")
            lines.extend(f"- {task.title}: {task.last_error or 'unspecified error'}" for task in failed)
        return "\n".join(lines).strip()
