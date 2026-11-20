"""LLM providers for direct in-terminal analysis."""

from __future__ import annotations

from .base import LLMProvider, ProviderError
from .claude import ClaudeProvider
from .gemini import GeminiProvider

PROVIDERS: dict[str, type[LLMProvider]] = {
    GeminiProvider.name: GeminiProvider,
    ClaudeProvider.name: ClaudeProvider,
}


def get_provider(name: str, **kwargs) -> LLMProvider:
    try:
        cls = PROVIDERS[name.lower()]
    except KeyError:
        raise ProviderError(f"Unknown provider {name!r}. Choose from: {', '.join(PROVIDERS)}") from None
    return cls(**kwargs)


def configured_providers() -> list[str]:
    return [name for name, cls in PROVIDERS.items() if cls.is_configured()]


__all__ = [
    "LLMProvider",
    "ProviderError",
    "GeminiProvider",
    "ClaudeProvider",
    "PROVIDERS",
    "get_provider",
    "configured_providers",
]
