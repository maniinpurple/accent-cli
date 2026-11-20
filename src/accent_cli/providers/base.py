"""Provider interface for direct LLM analysis."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator


class ProviderError(RuntimeError):
    pass


class LLMProvider(ABC):
    name: str = "base"
    env_key: str = ""
    model_env: str = ""
    default_model: str = ""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv(self.env_key, "").strip()
        if not self.api_key:
            raise ProviderError(f"{self.env_key} is not set (add it to .env).")
        self.model = model or os.getenv(self.model_env, "").strip() or self.default_model

    @classmethod
    def is_configured(cls) -> bool:
        return bool(os.getenv(cls.env_key, "").strip())

    @abstractmethod
    def stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Yield response text chunks as they arrive."""

    def analyze(self, system_prompt: str, user_message: str) -> str:
        return "".join(self.stream(system_prompt, user_message))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
