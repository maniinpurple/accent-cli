"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

from collections.abc import Iterator

from .base import LLMProvider, ProviderError


class GeminiProvider(LLMProvider):
    name = "gemini"
    env_key = "GEMINI_API_KEY"
    model_env = "GEMINI_MODEL"
    default_model = "gemini-flash-latest"

    def stream(self, system_prompt: str, user_message: str) -> Iterator[str]:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ProviderError("google-genai is not installed: pip install google-genai") from exc

        client = genai.Client(api_key=self.api_key)
        try:
            for chunk in client.models.generate_content_stream(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # noqa: BLE001 — surface SDK errors uniformly
            raise ProviderError(f"Gemini request failed: {exc}") from exc
