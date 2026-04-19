"""Anthropic / Claude LLM provider."""

from __future__ import annotations

import os
from typing import Any, Optional

from agentfabric.providers.base import LLMProvider

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(LLMProvider):
    """
    LLM provider backed by Anthropic's Claude API.

    Requires the ``anthropic`` package (installed automatically with agentfabric).

    Parameters
    ----------
    model:
        Claude model ID. Defaults to ``claude-sonnet-4-6``.
    api_key:
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY`` env var.
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
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it with: pip install anthropic"
            ) from e

        self._model = model or DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
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
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "messages": messages,
            **kwargs,
        }
        if system:
            params["system"] = system

        response = await self._client.messages.create(**params)
        if not response.content:
            raise RuntimeError(
                f"Anthropic API returned an empty content list for model {self._model!r}."
            )
        return response.content[0].text
