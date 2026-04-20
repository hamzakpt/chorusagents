"""Ollama provider for local LLM inference."""

from __future__ import annotations

from typing import Any, Optional

from chorusagents.providers.base import LLMProvider

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """
    LLM provider backed by Ollama for local, on-device model inference.

    Run any open-source model locally with zero API keys or costs.
    Supports Llama 3, Mistral, Phi-3, Qwen, Gemma, DeepSeek, and more.

    Requires Ollama to be installed and running::

        # Install: https://ollama.com/download
        ollama serve
        ollama pull llama3.1   # download the model first

    And the Python client::

        pip install chorusagents[ollama]
        # or: pip install ollama

    Parameters
    ----------
    model:
        Ollama model name, e.g. ``"llama3.1"``, ``"mistral"``,
        ``"phi3"``, ``"qwen2.5"``, ``"deepseek-r1"``, ``"gemma2"``.
    host:
        Ollama server URL. Defaults to ``"http://localhost:11434"``.

    Example::

        from chorusagents.providers import OllamaProvider
        from chorusagents import ChorusAgents

        provider = OllamaProvider(model="llama3.1")
        network = ChorusAgents.create("Software Team", provider=provider)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: str = DEFAULT_HOST,
        **kwargs: Any,
    ) -> None:
        try:
            from ollama import AsyncClient
        except ImportError as e:
            raise ImportError(
                "OllamaProvider requires the 'ollama' package and a running Ollama server. "
                "Install with: pip install chorusagents[ollama]\n"
                "Then start Ollama: ollama serve"
            ) from e

        self._model_name = model or DEFAULT_MODEL
        self._client = AsyncClient(host=host, **kwargs)

    @property
    def model(self) -> str:
        return self._model_name

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> str:
        ollama_messages = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(messages)

        response = await self._client.chat(
            model=self._model_name,
            messages=ollama_messages,
            **kwargs,
        )
        content = response.message.content
        if not content:
            raise RuntimeError(
                f"Ollama returned an empty response for model {self._model_name!r}. "
                "Make sure the model is downloaded: ollama pull " + self._model_name
            )
        return content
