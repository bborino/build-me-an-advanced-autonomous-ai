from __future__ import annotations

from dataclasses import dataclass

from autonomous_assistant.config import AssistantSettings
from autonomous_assistant.models import CompletionRequest, CompletionResult, ModelProfile, PlanTask
from autonomous_assistant.providers.anthropic import AnthropicProvider
from autonomous_assistant.providers.base import LanguageModelProvider
from autonomous_assistant.providers.openai_compatible import OpenAICompatibleProvider


@dataclass(slots=True)
class RankedProfile:
    profile: ModelProfile
    score: int


class ModelRegistry:
    def __init__(self, settings: AssistantSettings) -> None:
        self.settings = settings
        self.providers: dict[str, LanguageModelProvider] = {}
        self.profiles: list[ModelProfile] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        if self.settings.openai_api_key:
            self.providers["openai"] = OpenAICompatibleProvider(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
                timeout_seconds=self.settings.provider_timeout_seconds,
            )
            self.profiles.extend(
                [
                    ModelProfile(
                        alias="openai-reasoner",
                        provider="openai",
                        model_name=self.settings.openai_reasoning_model,
                        capabilities=["reasoning", "planning", "evaluation", "analysis"],
                        priority=8,
                    ),
                    ModelProfile(
                        alias="openai-code",
                        provider="openai",
                        model_name=self.settings.openai_code_model,
                        capabilities=[
                            "reasoning",
                            "planning",
                            "evaluation",
                            "analysis",
                            "code",
                        ],
                        priority=9,
                    ),
                    ModelProfile(
                        alias="openai-synthesis",
                        provider="openai",
                        model_name=self.settings.openai_synthesis_model,
                        capabilities=["synthesis", "reasoning"],
                        priority=7,
                    ),
                ]
            )

        if self.settings.anthropic_api_key:
            self.providers["anthropic"] = AnthropicProvider(
                api_key=self.settings.anthropic_api_key,
                base_url=self.settings.anthropic_base_url,
                timeout_seconds=self.settings.provider_timeout_seconds,
            )
            self.profiles.extend(
                [
                    ModelProfile(
                        alias="anthropic-reasoner",
                        provider="anthropic",
                        model_name=self.settings.anthropic_reasoning_model,
                        capabilities=["reasoning", "planning", "evaluation", "analysis"],
                        priority=8,
                    ),
                    ModelProfile(
                        alias="anthropic-synthesis",
                        provider="anthropic",
                        model_name=self.settings.anthropic_synthesis_model,
                        capabilities=["synthesis", "reasoning"],
                        priority=7,
                    ),
                ]
            )

    def list_profiles(self) -> list[ModelProfile]:
        return list(self.profiles)

    def get_profile(self, alias: str) -> ModelProfile | None:
        for profile in self.profiles:
            if profile.alias == alias:
                return profile
        return None

    def complete(
        self, profile: ModelProfile, request: CompletionRequest
    ) -> CompletionResult:
        provider = self.providers[profile.provider]
        return provider.complete(profile.model_name, request)


class ModelRouter:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def rank_profiles(
        self, phase: str, task: PlanTask | None = None
    ) -> list[ModelProfile]:
        desired = self._desired_capabilities(phase, task)
        ranked: list[RankedProfile] = []

        for profile in self.registry.list_profiles():
            score = profile.priority
            for capability in desired:
                if capability in profile.capabilities:
                    score += 4
                elif capability in {"planning", "evaluation"} and "reasoning" in profile.capabilities:
                    score += 2
                elif capability == "analysis" and "reasoning" in profile.capabilities:
                    score += 1
            if phase == "execution" and task and "code" in task.preferred_modes and "code" in profile.capabilities:
                score += 3
            ranked.append(RankedProfile(profile=profile, score=score))

        ranked.sort(key=lambda item: item.score, reverse=True)
        return [item.profile for item in ranked]

    def select(
        self, phase: str, task: PlanTask | None = None, attempt_index: int = 0
    ) -> ModelProfile | None:
        ranked = self.rank_profiles(phase=phase, task=task)
        if not ranked:
            return None
        index = min(max(attempt_index, 0), len(ranked) - 1)
        return ranked[index]

    @staticmethod
    def _desired_capabilities(phase: str, task: PlanTask | None) -> set[str]:
        desired: set[str] = set()
        if phase in {"planning", "execution", "evaluation"}:
            desired.add("reasoning")
        if phase == "planning":
            desired.add("planning")
        if phase == "evaluation":
            desired.add("evaluation")
        if phase == "synthesis":
            desired.add("synthesis")
        if task:
            desired.update(task.preferred_modes)
        return desired

