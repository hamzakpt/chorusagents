"""LangChain provider: wraps any BaseChatModel for universal LLM support."""

from __future__ import annotations

from typing import Any, Optional

from chorusagents.providers.base import LLMProvider


class LangChainProvider(LLMProvider):
    """
    Universal LLM provider that wraps any LangChain ``BaseChatModel``.

    This lets you plug in any of the 100+ LLMs supported by LangChain
    (OpenAI, Azure OpenAI, Gemini, Bedrock, Ollama, HuggingFace, Cohere,
    Mistral, Together AI, etc.) without needing a dedicated ChorusAgents provider.

    Requires ``langchain-core``::

        pip install langchain-core

    Plus the specific integration package for your model, e.g.::

        pip install langchain-openai        # OpenAI / Azure
        pip install langchain-google-genai  # Gemini
        pip install langchain-aws           # AWS Bedrock
        pip install langchain-ollama        # Ollama (local)
        pip install langchain-huggingface   # HuggingFace

    Examples::

        # OpenAI via LangChain
        from langchain_openai import ChatOpenAI
        from chorusagents.providers import LangChainProvider

        provider = LangChainProvider(ChatOpenAI(model="gpt-4o", api_key="..."))
        network = ChorusAgents.create("Law Firm", provider=provider)

        # Azure OpenAI via LangChain
        from langchain_openai import AzureChatOpenAI
        provider = LangChainProvider(
            AzureChatOpenAI(
                azure_endpoint="https://your-resource.openai.azure.com/",
                azure_deployment="gpt-4o",
                api_version="2024-02-01",
                api_key="...",
            )
        )

        # Gemini via LangChain
        from langchain_google_genai import ChatGoogleGenerativeAI
        provider = LangChainProvider(
            ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key="...")
        )

        # Ollama (local, no API key needed)
        from langchain_ollama import ChatOllama
        provider = LangChainProvider(ChatOllama(model="llama3.1"))
    """

    def __init__(self, llm: Any) -> None:
        """
        Parameters
        ----------
        llm:
            Any LangChain ``BaseChatModel`` instance (e.g. ``ChatOpenAI``,
            ``AzureChatOpenAI``, ``ChatGoogleGenerativeAI``, ``ChatOllama``).
        """
        self._validate_langchain_model(llm)
        self._llm = llm

    @property
    def model(self) -> str:
        # LangChain models expose model_name or model on their config
        return getattr(self._llm, "model_name", None) or getattr(self._llm, "model", None) or str(type(self._llm).__name__)

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        lc_messages = []
        if system:
            lc_messages.append(SystemMessage(content=system))
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        response = await self._llm.ainvoke(lc_messages, **kwargs)
        return response.content

    @staticmethod
    def _validate_langchain_model(llm: Any) -> None:
        try:
            from langchain_core.language_models import BaseChatModel
        except ImportError as e:
            raise ImportError(
                "LangChainProvider requires 'langchain-core'. "
                "Install it with: pip install langchain-core"
            ) from e
        if not isinstance(llm, BaseChatModel):
            raise TypeError(
                f"LangChainProvider expects a LangChain BaseChatModel instance, "
                f"got {type(llm).__name__!r}. "
                "Examples: ChatOpenAI, AzureChatOpenAI, ChatGoogleGenerativeAI, ChatOllama."
            )
