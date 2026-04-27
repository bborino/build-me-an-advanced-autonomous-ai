from __future__ import annotations

from typing import Any

from autonomous_assistant.config import AssistantSettings
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
    def __init__(
        self,
        models: ModelRegistry,
        router: ModelRouter,
        settings: AssistantSettings,
    ) -> None:
        self.models = models
        self.router = router
        self.settings = settings

    def interpret_goal(self, goal: str, memory_items: list[dict[str, Any]]) -> IntentProfile:
        if self.settings.use_local_fast_path():
            return self._heuristic_intent(goal)

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
        if self.settings.use_local_fast_path():
            return self._lightweight_plan(goal, intent)

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
        if self.settings.use_local_fast_path():
            return self._local_fast_next_action(
                goal=goal,
                task=task,
                plan=plan,
                relevant_memory=relevant_memory,
                tool_history=tool_history,
            )

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
        if self.settings.use_local_fast_path():
            return self._heuristic_evaluation(task, completion_note, tool_history), None

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
        if self.settings.use_local_fast_path():
            final_task = self._select_primary_answer_task(plan)
            if final_task and final_task.result_summary:
                return final_task.result_summary, None
            return self._heuristic_synthesis(goal, plan), None

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

    def _lightweight_plan(self, goal: str, intent: IntentProfile) -> list[PlanTask]:
        lowered = goal.lower()
        tasks: list[PlanTask] = []
        next_priority = 1

        if any(keyword in lowered for keyword in ["research", "search", "latest", "find", "compare"]):
            tasks.append(
                PlanTask(
                    id=make_task_id(len(tasks)),
                    title="Gather external context",
                    description="Search the web for the most relevant public information.",
                    priority=next_priority,
                    preferred_modes=["web", "analysis"],
                )
            )
            next_priority += 1

        if any(keyword in lowered for keyword in ["repo", "repository", "workspace", "project", "codebase", "file"]):
            tasks.append(
                PlanTask(
                    id=make_task_id(len(tasks)),
                    title="Inspect workspace context",
                    description="Collect the most relevant local files and repository context.",
                    dependencies=[task.id for task in tasks],
                    priority=next_priority,
                    preferred_modes=["filesystem", "analysis", "code"],
                )
            )
            next_priority += 1

        preferred_modes = ["reasoning", "analysis"]
        if any(keyword in lowered for keyword in ["build", "create", "design", "implement", "develop", "code"]):
            preferred_modes.append("code")

        tasks.append(
            PlanTask(
                id=make_task_id(len(tasks)),
                title="Answer goal",
                description="Produce the main answer or artifact using the available evidence.",
                dependencies=[task.id for task in tasks],
                priority=next_priority,
                preferred_modes=preferred_modes,
            )
        )
        return tasks

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

    def _local_fast_next_action(
        self,
        goal: str,
        task: PlanTask,
        plan: list[PlanTask],
        relevant_memory: list[dict[str, Any]],
        tool_history: list[ToolResult],
    ) -> tuple[ToolDecision, ModelProfile | None]:
        task_text = f"{task.title} {task.description}".lower()

        if not tool_history:
            if "web" in task.preferred_modes:
                return (
                    ToolDecision(
                        action="tool",
                        working_summary="Collecting lightweight web context",
                        tool_name="web_search",
                        tool_input={"query": goal, "max_results": 4},
                    ),
                    None,
                )

            if "filesystem" in task.preferred_modes:
                operation = "search" if any(
                    keyword in goal.lower()
                    for keyword in ["readme", "config", "router", "assistant", "file"]
                ) else "list"
                tool_input = (
                    {"operation": "search", "path": ".", "query": "assistant", "limit": 12}
                    if operation == "search"
                    else {"operation": "list", "path": ".", "recursive": False, "limit": 40}
                )
                return (
                    ToolDecision(
                        action="tool",
                        working_summary="Collecting lightweight workspace context",
                        tool_name="filesystem",
                        tool_input=tool_input,
                    ),
                    None,
                )

        if "answer goal" in task_text:
            answer, profile = self._generate_local_fast_answer(
                goal=goal,
                task=task,
                plan=plan,
                relevant_memory=relevant_memory,
                tool_history=tool_history,
            )
            return (
                ToolDecision(
                    action="complete",
                    working_summary="Generating a direct answer with the local model",
                    completion_note=answer,
                ),
                profile,
            )

        if tool_history and tool_history[-1].ok:
            return (
                ToolDecision(
                    action="complete",
                    working_summary="Summarizing collected context",
                    completion_note=self._summarize_tool_history(tool_history),
                ),
                None,
            )

        return self._heuristic_next_action(goal, task, tool_history), None

    def _generate_local_fast_answer(
        self,
        goal: str,
        task: PlanTask,
        plan: list[PlanTask],
        relevant_memory: list[dict[str, Any]],
        tool_history: list[ToolResult],
    ) -> tuple[str, ModelProfile | None]:
        profile = self.router.select("execution", task=task)
        if not profile:
            return self._fallback_answer(goal, task, tool_history), None

        context_notes = self._format_local_fast_context(relevant_memory, tool_history)
        system = (
            "You are a concise local autonomous assistant running on a small model. "
            "Give the best direct answer you can. Do not explain your chain-of-thought. "
            "Use short sections or bullets only when they help clarity."
        )
        prompt = f"""
User goal:
{goal}

Current task:
{task.title}: {task.description}

Available context:
{context_notes}

Instructions:
- Produce the actual answer for the user, not a status update
- Be concise and practical
- If context is limited, say that briefly and still give the best answer you can
"""
        try:
            result = self.models.complete(
                profile,
                CompletionRequest(
                    system=system,
                    prompt=prompt,
                    response_format="text",
                    max_tokens=self.settings.local_model_max_tokens,
                ),
            )
            text = result.text.strip()
            if text:
                return text, profile
        except Exception:
            pass

        return self._fallback_answer(goal, task, tool_history), None

    @staticmethod
    def _summarize_tool_history(tool_history: list[ToolResult]) -> str:
        if not tool_history:
            return "No supporting context was collected."
        summaries = [truncate_text(result.summary, 220) for result in tool_history[-2:]]
        return " ".join(summaries)

    def _format_local_fast_context(
        self,
        relevant_memory: list[dict[str, Any]],
        tool_history: list[ToolResult],
    ) -> str:
        lines: list[str] = []
        for note in relevant_memory[-8:]:
            content = truncate_text(note.get("content", ""), 260)
            if content:
                lines.append(f"- {note.get('kind', 'note')}: {content}")

        for result in tool_history[-2:]:
            lines.append(f"- tool:{result.tool_name}: {truncate_text(result.summary, 260)}")
            if isinstance(result.content, dict):
                if "results" in result.content:
                    results = result.content.get("results", [])
                    for item in results[:2]:
                        lines.append(
                            "- source: "
                            f"{truncate_text(item.get('title', ''), 80)} | "
                            f"{truncate_text(item.get('snippet', ''), 140)}"
                        )
                elif "text" in result.content:
                    lines.append(f"- text: {truncate_text(result.content.get('text', ''), 220)}")
                elif "matches" in result.content:
                    for match in result.content.get("matches", [])[:4]:
                        lines.append(
                            "- file match: "
                            f"{match.get('path')}:{match.get('line_number')} "
                            f"{truncate_text(match.get('line', ''), 140)}"
                        )
                elif "entries" in result.content:
                    entries = result.content.get("entries", [])[:6]
                    lines.append(
                        "- files: " + ", ".join(entry.get("path", "") for entry in entries if entry.get("path"))
                    )

        return "\n".join(lines) if lines else "- No extra context was collected."

    @staticmethod
    def _fallback_answer(goal: str, task: PlanTask, tool_history: list[ToolResult]) -> str:
        if tool_history and tool_history[-1].ok:
            return f"{task.title}: {truncate_text(tool_history[-1].summary, 500)}"
        return f"Best-effort answer requested for: {goal}"

    @staticmethod
    def _select_primary_answer_task(plan: list[PlanTask]) -> PlanTask | None:
        for task in plan:
            if task.status.value == "completed" and "answer goal" in task.title.lower():
                return task
        completed = [task for task in plan if task.status.value == "completed" and task.result_summary]
        if completed:
            return completed[-1]
        return None

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
