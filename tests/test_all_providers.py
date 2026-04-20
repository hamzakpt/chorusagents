"""Tests for all LLM provider implementations (mock-based, no real API calls)."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from chorusagents.providers import (
    get_provider,
    SUPPORTED_PROVIDERS,
    AnthropicProvider,
    OpenAIProvider,
    AzureOpenAIProvider,
    GeminiProvider,
    BedrockProvider,
    OllamaProvider,
    HuggingFaceProvider,
    LangChainProvider,
    LLMProvider,
)
from tests.conftest import MockLLMProvider


# ── get_provider factory ──────────────────────────────────────────────────────

def test_get_provider_anthropic():
    p = get_provider("anthropic", api_key="fake")
    assert isinstance(p, AnthropicProvider)
    assert p.model == "claude-sonnet-4-6"


def test_get_provider_anthropic_custom_model():
    p = get_provider("anthropic", model="claude-opus-4-7", api_key="fake")
    assert p.model == "claude-opus-4-7"


def test_get_provider_openai():
    p = get_provider("openai", api_key="fake")
    assert isinstance(p, OpenAIProvider)
    assert p.model == "gpt-4o"


def test_get_provider_openai_custom_model():
    p = get_provider("openai", model="gpt-4o-mini", api_key="fake")
    assert p.model == "gpt-4o-mini"


def test_get_provider_azure_alias_azure_openai():
    p = get_provider(
        "azure-openai",
        api_key="fake",
        azure_endpoint="https://x.openai.azure.com/",
        azure_deployment="gpt-4o",
    )
    assert isinstance(p, AzureOpenAIProvider)


def test_get_provider_azure_alias_azure():
    p = get_provider(
        "azure",
        api_key="fake",
        azure_endpoint="https://x.openai.azure.com/",
        azure_deployment="dep",
    )
    assert isinstance(p, AzureOpenAIProvider)


def test_get_provider_gemini():
    pytest.importorskip("google.generativeai", reason="google-generativeai not installed")
    p = get_provider("gemini", api_key="AIza-fake")
    assert isinstance(p, GeminiProvider)
    assert p.model == "gemini-1.5-flash"


def test_get_provider_google_alias():
    pytest.importorskip("google.generativeai", reason="google-generativeai not installed")
    p = get_provider("google", api_key="AIza-fake")
    assert isinstance(p, GeminiProvider)


def test_get_provider_bedrock():
    pytest.importorskip("boto3", reason="boto3 not installed")
    p = get_provider("bedrock")
    assert isinstance(p, BedrockProvider)
    assert "claude" in p.model.lower() or "anthropic" in p.model.lower()


def test_get_provider_aws_alias():
    pytest.importorskip("boto3", reason="boto3 not installed")
    p = get_provider("aws")
    assert isinstance(p, BedrockProvider)


def test_get_provider_ollama():
    pytest.importorskip("ollama", reason="ollama not installed")
    p = get_provider("ollama", model="mistral")
    assert isinstance(p, OllamaProvider)
    assert p.model == "mistral"


def test_get_provider_huggingface():
    pytest.importorskip("huggingface_hub", reason="huggingface-hub not installed")
    p = get_provider("huggingface", api_key="hf_fake")
    assert isinstance(p, HuggingFaceProvider)


def test_get_provider_hf_alias():
    pytest.importorskip("huggingface_hub", reason="huggingface-hub not installed")
    p = get_provider("hf", api_key="hf_fake")
    assert isinstance(p, HuggingFaceProvider)


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("cohere")


def test_get_provider_error_message_mentions_langchain():
    with pytest.raises(ValueError, match="LangChainProvider"):
        get_provider("cohere")


def test_supported_providers_list_is_complete():
    assert "anthropic" in SUPPORTED_PROVIDERS
    assert "openai" in SUPPORTED_PROVIDERS
    assert "azure" in SUPPORTED_PROVIDERS
    assert "gemini" in SUPPORTED_PROVIDERS
    assert "bedrock" in SUPPORTED_PROVIDERS
    assert "ollama" in SUPPORTED_PROVIDERS
    assert "huggingface" in SUPPORTED_PROVIDERS


# ── AzureOpenAIProvider ───────────────────────────────────────────────────────

def test_azure_provider_requires_endpoint():
    with pytest.raises(ValueError, match="endpoint"):
        AzureOpenAIProvider(api_key="fake", azure_deployment="dep")


def test_azure_provider_requires_deployment():
    with pytest.raises(ValueError, match="deployment"):
        AzureOpenAIProvider(api_key="fake", azure_endpoint="https://x.openai.azure.com/")


def test_azure_provider_model_name():
    p = AzureOpenAIProvider(
        api_key="fake",
        azure_endpoint="https://x.openai.azure.com/",
        azure_deployment="gpt-4o-prod",
    )
    assert "gpt-4o-prod" in p.model


@pytest.mark.asyncio
async def test_azure_provider_complete():
    p = AzureOpenAIProvider(
        api_key="fake",
        azure_endpoint="https://x.openai.azure.com/",
        azure_deployment="gpt-4o",
    )
    mock_choice = MagicMock()
    mock_choice.message.content = "Azure response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    p._client.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await p.complete([{"role": "user", "content": "Hello"}], system="You are helpful.")
    assert result == "Azure response"


@pytest.mark.asyncio
async def test_azure_provider_empty_choices_raises():
    p = AzureOpenAIProvider(
        api_key="fake",
        azure_endpoint="https://x.openai.azure.com/",
        azure_deployment="gpt-4o",
    )
    mock_response = MagicMock()
    mock_response.choices = []
    p._client.chat.completions.create = AsyncMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="no choices"):
        await p.complete([{"role": "user", "content": "Hi"}])


# ── GeminiProvider ────────────────────────────────────────────────────────────

def test_gemini_provider_requires_api_key(monkeypatch):
    pytest.importorskip("google.generativeai", reason="google-generativeai not installed")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        GeminiProvider(api_key=None)


def test_gemini_provider_default_model():
    pytest.importorskip("google.generativeai", reason="google-generativeai not installed")
    p = GeminiProvider(api_key="AIza-fake")
    assert p.model == "gemini-1.5-flash"


def test_gemini_provider_custom_model():
    pytest.importorskip("google.generativeai", reason="google-generativeai not installed")
    p = GeminiProvider(api_key="AIza-fake", model="gemini-2.0-flash")
    assert p.model == "gemini-2.0-flash"


@pytest.mark.asyncio
async def test_gemini_provider_complete():
    pytest.importorskip("google.generativeai", reason="google-generativeai not installed")
    p = GeminiProvider(api_key="AIza-fake", model="gemini-1.5-flash")
    mock_response = MagicMock()
    mock_response.text = "Gemini answer"
    p._client = MagicMock()
    p._client.generate_content = MagicMock(return_value=mock_response)

    result = await p.complete([{"role": "user", "content": "Hello"}])
    assert result == "Gemini answer"


@pytest.mark.asyncio
async def test_gemini_provider_empty_response_raises():
    pytest.importorskip("google.generativeai", reason="google-generativeai not installed")
    p = GeminiProvider(api_key="AIza-fake")
    mock_response = MagicMock()
    mock_response.text = ""
    p._client = MagicMock()
    p._client.generate_content = MagicMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="empty response"):
        await p.complete([{"role": "user", "content": "Hi"}])


# ── BedrockProvider ───────────────────────────────────────────────────────────

def test_bedrock_provider_default_model():
    pytest.importorskip("boto3", reason="boto3 not installed")
    p = BedrockProvider()
    assert "claude" in p.model.lower() or "anthropic" in p.model.lower()


def test_bedrock_provider_custom_model():
    pytest.importorskip("boto3", reason="boto3 not installed")
    p = BedrockProvider(model_id="meta.llama3-70b-instruct-v1:0")
    assert p.model == "meta.llama3-70b-instruct-v1:0"


def test_bedrock_provider_complete_sync():
    pytest.importorskip("boto3", reason="boto3 not installed")
    p = BedrockProvider(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
    mock_response = {
        "output": {
            "message": {
                "content": [{"text": "Bedrock response"}]
            }
        }
    }
    p._client = MagicMock()
    p._client.converse = MagicMock(return_value=mock_response)

    result = p._complete_sync(
        [{"role": "user", "content": "Hello"}],
        system="Be helpful.",
        extra_kwargs={},
    )
    assert result == "Bedrock response"


def test_bedrock_provider_empty_content_raises():
    pytest.importorskip("boto3", reason="boto3 not installed")
    p = BedrockProvider()
    mock_response = {"output": {"message": {"content": []}}}
    p._client = MagicMock()
    p._client.converse = MagicMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="empty content"):
        p._complete_sync([], system="", extra_kwargs={})


# ── OllamaProvider ────────────────────────────────────────────────────────────

def test_ollama_provider_default_model():
    pytest.importorskip("ollama", reason="ollama not installed")
    p = OllamaProvider()
    assert p.model == "llama3.1"


def test_ollama_provider_custom_model():
    pytest.importorskip("ollama", reason="ollama not installed")
    p = OllamaProvider(model="mistral")
    assert p.model == "mistral"


@pytest.mark.asyncio
async def test_ollama_provider_complete():
    pytest.importorskip("ollama", reason="ollama not installed")
    p = OllamaProvider(model="llama3.1")
    mock_response = MagicMock()
    mock_response.message.content = "Ollama answer"
    p._client = MagicMock()
    p._client.chat = AsyncMock(return_value=mock_response)

    result = await p.complete([{"role": "user", "content": "Hello"}], system="Be helpful.")
    assert result == "Ollama answer"


@pytest.mark.asyncio
async def test_ollama_provider_empty_response_raises():
    pytest.importorskip("ollama", reason="ollama not installed")
    p = OllamaProvider()
    mock_response = MagicMock()
    mock_response.message.content = ""
    p._client = MagicMock()
    p._client.chat = AsyncMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="empty response"):
        await p.complete([{"role": "user", "content": "Hi"}])


# ── HuggingFaceProvider ───────────────────────────────────────────────────────

def test_huggingface_provider_default_model():
    pytest.importorskip("huggingface_hub", reason="huggingface-hub not installed")
    p = HuggingFaceProvider(api_key="hf_fake")
    assert "llama" in p.model.lower() or "meta" in p.model.lower()


def test_huggingface_provider_custom_model():
    pytest.importorskip("huggingface_hub", reason="huggingface-hub not installed")
    p = HuggingFaceProvider(model="mistralai/Mistral-7B-Instruct-v0.3", api_key="hf_fake")
    assert p.model == "mistralai/Mistral-7B-Instruct-v0.3"


@pytest.mark.asyncio
async def test_huggingface_provider_complete():
    pytest.importorskip("huggingface_hub", reason="huggingface-hub not installed")
    p = HuggingFaceProvider(api_key="hf_fake")
    mock_choice = MagicMock()
    mock_choice.message.content = "HF response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    p._client = MagicMock()
    p._client.chat_completion = AsyncMock(return_value=mock_response)

    result = await p.complete([{"role": "user", "content": "Hello"}])
    assert result == "HF response"


@pytest.mark.asyncio
async def test_huggingface_provider_empty_choices_raises():
    pytest.importorskip("huggingface_hub", reason="huggingface-hub not installed")
    p = HuggingFaceProvider(api_key="hf_fake")
    mock_response = MagicMock()
    mock_response.choices = []
    p._client = MagicMock()
    p._client.chat_completion = AsyncMock(return_value=mock_response)

    with pytest.raises(RuntimeError, match="no choices"):
        await p.complete([{"role": "user", "content": "Hi"}])


# ── LangChainProvider ─────────────────────────────────────────────────────────

def test_langchain_provider_rejects_non_chat_model():
    pytest.importorskip("langchain_core", reason="langchain-core not installed")
    with pytest.raises(TypeError, match="BaseChatModel"):
        LangChainProvider(llm="not-a-model")


def test_langchain_provider_rejects_plain_object():
    pytest.importorskip("langchain_core", reason="langchain-core not installed")
    with pytest.raises(TypeError, match="BaseChatModel"):
        LangChainProvider(llm=42)


@pytest.mark.asyncio
async def test_langchain_provider_complete():
    """LangChainProvider wraps any BaseChatModel and calls ainvoke."""
    try:
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
    except ImportError:
        pytest.skip("langchain-core not installed")

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="LangChain response"))
    mock_llm.model_name = "mock-lc-model"

    p = LangChainProvider(llm=mock_llm)
    assert p.model == "mock-lc-model"

    result = await p.complete(
        [{"role": "user", "content": "Hello"}],
        system="Be helpful.",
    )
    assert result == "LangChain response"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_langchain_provider_passes_system_as_system_message():
    try:
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage, SystemMessage
    except ImportError:
        pytest.skip("langchain-core not installed")

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    p = LangChainProvider(llm=mock_llm)
    await p.complete([{"role": "user", "content": "hi"}], system="You are a lawyer.")

    call_args = mock_llm.ainvoke.call_args[0][0]
    assert any(isinstance(m, SystemMessage) and "lawyer" in m.content for m in call_args)


# ── Import sanity check ───────────────────────────────────────────────────────

def test_all_providers_importable_from_top_level():
    from chorusagents import (
        AnthropicProvider,
        OpenAIProvider,
        AzureOpenAIProvider,
        GeminiProvider,
        BedrockProvider,
        OllamaProvider,
        HuggingFaceProvider,
        LangChainProvider,
        get_provider,
    )
    for cls in [
        AnthropicProvider, OpenAIProvider, AzureOpenAIProvider,
        GeminiProvider, BedrockProvider, OllamaProvider,
        HuggingFaceProvider, LangChainProvider,
    ]:
        assert issubclass(cls, LLMProvider)
