"""Azure OpenAI provider."""

from __future__ import annotations

import os
from typing import Any, Optional

from agentfabric.providers.base import LLMProvider

DEFAULT_API_VERSION = "2024-02-01"
DEFAULT_MAX_TOKENS = 4096


class AzureOpenAIProvider(LLMProvider):
    """
    LLM provider backed by Azure OpenAI Service.

    Requires the ``openai`` package::

        pip install agentfabric[azure]
        # or: pip install openai

    Parameters
    ----------
    azure_endpoint:
        Your Azure OpenAI endpoint URL, e.g.
        ``"https://my-resource.openai.azure.com/"``.
        Falls back to ``AZURE_OPENAI_ENDPOINT`` env var.
    azure_deployment:
        The deployment name (not the model name) configured in Azure,
        e.g. ``"gpt-4o-deployment"``.
        Falls back to ``AZURE_OPENAI_DEPLOYMENT`` env var.
    api_key:
        Azure OpenAI API key.
        Falls back to ``AZURE_OPENAI_API_KEY`` env var.
    api_version:
        Azure OpenAI API version. Defaults to ``"2024-02-01"``.
    max_tokens:
        Maximum tokens in the response.

    Example::

        from agentfabric.providers import AzureOpenAIProvider
        from agentfabric import AgentFabric

        provider = AzureOpenAIProvider(
            azure_endpoint="https://my-resource.openai.azure.com/",
            azure_deployment="gpt-4o-prod",
            api_key="your-azure-key",
            api_version="2024-02-01",
        )
        network = AgentFabric.create("Law Firm", provider=provider)
    """

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as e:
            raise ImportError(
                "AzureOpenAIProvider requires 'openai'. "
                "Install it with: pip install agentfabric[azure]"
            ) from e

        self._endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._deployment = azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        self._api_version = api_version
        self._max_tokens = max_tokens

        if not self._endpoint:
            raise ValueError(
                "Azure endpoint is required. Pass 'azure_endpoint' or set "
                "the AZURE_OPENAI_ENDPOINT environment variable."
            )
        if not self._deployment:
            raise ValueError(
                "Azure deployment name is required. Pass 'azure_deployment' or set "
                "the AZURE_OPENAI_DEPLOYMENT environment variable."
            )

        self._client = AsyncAzureOpenAI(
            azure_endpoint=self._endpoint,
            azure_deployment=self._deployment,
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
            **kwargs,
        )

    @property
    def model(self) -> str:
        return f"azure/{self._deployment}"

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
            model=self._deployment,
            messages=all_messages,  # type: ignore[arg-type]
            max_tokens=kwargs.pop("max_tokens", self._max_tokens),
            **kwargs,
        )
        if not response.choices:
            raise RuntimeError(
                f"Azure OpenAI returned no choices for deployment {self._deployment!r}."
            )
        return response.choices[0].message.content or ""
