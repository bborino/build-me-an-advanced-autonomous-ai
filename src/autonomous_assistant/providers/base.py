from __future__ import annotations

from abc import ABC, abstractmethod

from autonomous_assistant.models import CompletionRequest, CompletionResult


class LanguageModelProvider(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def complete(self, model_name: str, request: CompletionRequest) -> CompletionResult:
        raise NotImplementedError

