"""Google Gemini provider."""

from __future__ import annotations

import os
from typing import Any, Optional

from chorusagents.providers.base import LLMProvider

DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_MAX_TOKENS = 4096


class GeminiProvider(LLMProvider):
    """
    LLM provider backed by Google Gemini (via ``google-generativeai``).

    Requires::

        pip install chorusagents[gemini]
        # or: pip install google-generativeai

    Parameters
    ----------
    api_key:
        Google AI Studio API key.
        Falls back to ``GOOGLE_API_KEY`` env var.
    model:
        Gemini model ID. Defaults to ``"gemini-1.5-flash"``.
        Options: ``"gemini-1.5-pro"``, ``"gemini-1.5-flash"``,
        ``"gemini-2.0-flash"``, ``"gemini-2.5-pro"``.
    max_tokens:
        Maximum tokens in the response.

    Example::

        from chorusagents.providers import GeminiProvider
        from chorusagents import ChorusAgents

        provider = GeminiProvider(api_key="AIza...", model="gemini-1.5-pro")
        network = ChorusAgents.create("Research Lab", provider=provider)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "GeminiProvider requires 'google-generativeai'. "
                "Install it with: pip install chorusagents[gemini]"
            ) from e

        self._model_name = model or DEFAULT_MODEL
        self._max_tokens = max_tokens
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Google API key is required. Pass 'api_key' or set "
                "the GOOGLE_API_KEY environment variable."
            )
        genai.configure(api_key=resolved_key)
        self._client = genai.GenerativeModel(self._model_name)

    @property
    def model(self) -> str:
        return self._model_name

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> str:
        import asyncio

        prompt_parts = []
        if system:
            prompt_parts.append(f"[System instructions]\n{system}\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            prompt_parts.append(f"{prefix}: {content}")

        prompt = "\n\n".join(prompt_parts)

        generation_config = {"max_output_tokens": self._max_tokens}
        generation_config.update(kwargs)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._client.generate_content(
                prompt, generation_config=generation_config
            ),
        )
        if not response.text:
            raise RuntimeError(
                f"Gemini returned an empty response for model {self._model_name!r}."
            )
        return response.text
