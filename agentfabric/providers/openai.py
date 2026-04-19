"""OpenAI LLM provider."""

from __future__ import annotations

import os
from typing import Any, Optional

from agentfabric.providers.base import LLMProvider

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 4096


class OpenAIProvider(LLMProvider):
    """
    LLM provider backed by OpenAI's API.

    Requires the ``openai`` package (installed automatically with agentfabric).

    Parameters
    ----------
    model:
        OpenAI model ID. Defaults to ``gpt-4o``.
    api_key:
        OpenAI API key. Falls back to the ``OPENAI_API_KEY`` env var.
    max_tokens:
        Maximum tokens in the completion response.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it with: pip install openai"
            ) from e

        self._model = model or DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> str:
        all_messages: list[dict[str, str]] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=all_messages,  # type: ignore[arg-type]
            max_tokens=kwargs.pop("max_tokens", self._max_tokens),
            **kwargs,
        )
        if not response.choices:
            raise RuntimeError(
                f"OpenAI API returned no choices for model {self._model!r}."
            )
        return response.choices[0].message.content or ""
