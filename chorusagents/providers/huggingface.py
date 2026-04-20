"""HuggingFace Inference API provider."""

from __future__ import annotations

import os
from typing import Any, Optional

from chorusagents.providers.base import LLMProvider

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class HuggingFaceProvider(LLMProvider):
    """
    LLM provider backed by HuggingFace Inference API (serverless or dedicated endpoints).

    Access thousands of open-source models hosted on HuggingFace Hub.

    Requires::

        pip install chorusagents[huggingface]
        # or: pip install huggingface-hub

    Parameters
    ----------
    model:
        HuggingFace model ID, e.g.
        ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``,
        ``"mistralai/Mistral-7B-Instruct-v0.3"``,
        ``"microsoft/Phi-3-mini-4k-instruct"``.
    api_key:
        HuggingFace API token (``hf_...``).
        Falls back to ``HF_TOKEN`` or ``HUGGINGFACEHUB_API_TOKEN`` env vars.
    endpoint_url:
        Custom Inference Endpoint URL for dedicated deployments.
        When set, overrides the model ID for routing.
    max_tokens:
        Maximum tokens in the response.

    Example::

        from chorusagents.providers import HuggingFaceProvider
        from chorusagents import ChorusAgents

        provider = HuggingFaceProvider(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            api_key="hf_...",
        )
        network = ChorusAgents.create("Research Lab", provider=provider)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> None:
        try:
            from huggingface_hub import AsyncInferenceClient
        except ImportError as e:
            raise ImportError(
                "HuggingFaceProvider requires 'huggingface-hub'. "
                "Install it with: pip install chorusagents[huggingface]"
            ) from e

        self._model_name = model or DEFAULT_MODEL
        self._max_tokens = max_tokens
        resolved_key = (
            api_key
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )

        client_kwargs: dict[str, Any] = {"token": resolved_key}
        if endpoint_url:
            client_kwargs["base_url"] = endpoint_url

        self._client = AsyncInferenceClient(
            model=self._model_name,
            **client_kwargs,
        )

    @property
    def model(self) -> str:
        return self._model_name

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> str:
        hf_messages = []
        if system:
            hf_messages.append({"role": "system", "content": system})
        hf_messages.extend(messages)

        response = await self._client.chat_completion(
            messages=hf_messages,
            max_tokens=kwargs.pop("max_tokens", self._max_tokens),
            **kwargs,
        )
        if not response.choices:
            raise RuntimeError(
                f"HuggingFace returned no choices for model {self._model_name!r}."
            )
        return response.choices[0].message.content or ""
