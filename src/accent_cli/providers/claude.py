"""Anthropic Claude provider (anthropic SDK)."""

from __future__ import annotations

from collections.abc import Iterator

from .base import LLMProvider, ProviderError


class ClaudeProvider(LLMProvider):
    name = "claude"
    env_key = "ANTHROPIC_API_KEY"
    model_env = "ANTHROPIC_MODEL"
    default_model = "claude-opus-5-5"
    max_tokens = 4096

    def stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError("anthropic is not installed: pip install anthropic") from exc

        client = anthropic.Anthropic(api_key=self.api_key)
        try:
            with client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                yield from stream.text_stream
        except anthropic.APIError as exc:
            raise ProviderError(f"Claude request failed: {exc}") from exc
