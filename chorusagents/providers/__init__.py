"""
ChorusAgents LLM Providers
=========================

Provider classes for every major LLM platform. Import the one you need,
pass it to ``ChorusAgents.create()``, and the rest is automatic.

Quick reference
---------------

+----------------------+---------------------------------+---------------------------------------+
| Provider class       | Install extra                   | Env var(s)                            |
+======================+=================================+=======================================+
| AnthropicProvider    | (included)                      | ANTHROPIC_API_KEY                     |
| OpenAIProvider       | (included)                      | OPENAI_API_KEY                        |
| AzureOpenAIProvider  | ``pip install chorusagents[azure]``  | AZURE_OPENAI_ENDPOINT, _API_KEY      |
| GeminiProvider       | ``pip install chorusagents[gemini]`` | GOOGLE_API_KEY                       |
| BedrockProvider      | ``pip install chorusagents[bedrock]``| AWS_ACCESS_KEY_ID, etc.              |
| OllamaProvider       | ``pip install chorusagents[ollama]`` | (none — local server)                |
| HuggingFaceProvider  | ``pip install chorusagents[huggingface]`` | HF_TOKEN                       |
| LangChainProvider    | ``pip install langchain-core`` + integration | (depends on model)    |
+----------------------+---------------------------------+---------------------------------------+

Usage::

    from chorusagents.providers import AnthropicProvider
    provider = AnthropicProvider(api_key="sk-ant-...", model="claude-opus-4-7")

    from chorusagents.providers import OpenAIProvider
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

    from chorusagents.providers import AzureOpenAIProvider
    provider = AzureOpenAIProvider(
        azure_endpoint="https://my.openai.azure.com/",
        azure_deployment="gpt-4o",
        api_key="...",
    )

    from chorusagents.providers import GeminiProvider
    provider = GeminiProvider(api_key="AIza...", model="gemini-1.5-pro")

    from chorusagents.providers import BedrockProvider
    provider = BedrockProvider(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")

    from chorusagents.providers import OllamaProvider
    provider = OllamaProvider(model="llama3.1")   # no API key needed

    from chorusagents.providers import HuggingFaceProvider
    provider = HuggingFaceProvider(model="meta-llama/Meta-Llama-3.1-8B-Instruct", api_key="hf_...")

    # Any LangChain BaseChatModel
    from langchain_openai import ChatOpenAI
    from chorusagents.providers import LangChainProvider
    provider = LangChainProvider(ChatOpenAI(model="gpt-4o", api_key="..."))

    # Then create your network
    from chorusagents import ChorusAgents
    network = ChorusAgents.create("Criminal Defense Law Firm", provider=provider)
"""

from __future__ import annotations

from typing import Any, Optional

from chorusagents.providers.base import LLMProvider
from chorusagents.providers.anthropic import AnthropicProvider
from chorusagents.providers.openai import OpenAIProvider
from chorusagents.providers.azure_openai import AzureOpenAIProvider
from chorusagents.providers.gemini import GeminiProvider
from chorusagents.providers.bedrock import BedrockProvider
from chorusagents.providers.ollama import OllamaProvider
from chorusagents.providers.huggingface import HuggingFaceProvider
from chorusagents.providers.langchain_provider import LangChainProvider

__all__ = [
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider",
    "GeminiProvider",
    "BedrockProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "LangChainProvider",
    "get_provider",
]

#: All supported provider shorthand names for use with ``get_provider()``.
SUPPORTED_PROVIDERS = [
    "anthropic",
    "openai",
    "azure",
    "azure-openai",
    "gemini",
    "google",
    "bedrock",
    "aws",
    "ollama",
    "huggingface",
    "hf",
    "langchain",
]


def get_provider(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory that returns a configured ``LLMProvider`` by name.

    Parameters
    ----------
    provider:
        Provider shorthand string. One of:
        ``"anthropic"``, ``"openai"``, ``"azure"`` / ``"azure-openai"``,
        ``"gemini"`` / ``"google"``, ``"bedrock"`` / ``"aws"``,
        ``"ollama"``, ``"huggingface"`` / ``"hf"``.
    model:
        Model identifier passed to the provider constructor.
        Each provider has a sensible default if omitted.
    api_key:
        API key / token for the provider.
        Falls back to the provider's standard environment variable.
    **kwargs:
        Additional keyword arguments forwarded to the provider constructor.
        For example, ``azure_endpoint`` and ``azure_deployment`` for Azure,
        or ``region_name`` for Bedrock.

    Returns
    -------
    LLMProvider
        A ready-to-use provider instance.

    Raises
    ------
    ValueError
        If the provider name is not recognised.

    Examples::

        # String shorthand
        p = get_provider("anthropic", model="claude-opus-4-7", api_key="sk-ant-...")
        p = get_provider("openai", model="gpt-4o", api_key="sk-...")
        p = get_provider("gemini", model="gemini-1.5-pro", api_key="AIza...")
        p = get_provider("ollama", model="llama3.1")
        p = get_provider(
            "azure",
            model="gpt-4o",
            api_key="...",
            azure_endpoint="https://my.openai.azure.com/",
            azure_deployment="gpt-4o-prod",
        )
    """
    name = provider.lower().strip()

    if name == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key, **kwargs)

    if name == "openai":
        return OpenAIProvider(model=model, api_key=api_key, **kwargs)

    if name in ("azure", "azure-openai", "azure_openai"):
        return AzureOpenAIProvider(api_key=api_key, **kwargs)

    if name in ("gemini", "google", "google-gemini"):
        return GeminiProvider(model=model, api_key=api_key, **kwargs)

    if name in ("bedrock", "aws", "aws-bedrock"):
        return BedrockProvider(model_id=model, **kwargs)

    if name == "ollama":
        return OllamaProvider(model=model, **kwargs)

    if name in ("huggingface", "hf", "hugging-face"):
        return HuggingFaceProvider(model=model, api_key=api_key, **kwargs)

    raise ValueError(
        f"Unknown provider {provider!r}. "
        f"Supported values: {', '.join(SUPPORTED_PROVIDERS)}.\n"
        "For any other LangChain model, use LangChainProvider directly:\n"
        "  from chorusagents.providers import LangChainProvider\n"
        "  from langchain_mistralai import ChatMistralAI\n"
        "  provider = LangChainProvider(ChatMistralAI(api_key='...'))"
    )
