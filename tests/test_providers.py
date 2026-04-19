"""Tests for LLM provider implementations."""

import pytest
from tests.conftest import MockLLMProvider
from agentfabric.providers import get_provider


def test_mock_provider_model():
    p = MockLLMProvider(response="hello")
    assert p.model == "mock-v1"


@pytest.mark.asyncio
async def test_mock_provider_returns_configured_response():
    p = MockLLMProvider(response="expected output")
    result = await p.complete(messages=[{"role": "user", "content": "hi"}])
    assert result == "expected output"


@pytest.mark.asyncio
async def test_mock_provider_records_calls():
    p = MockLLMProvider()
    await p.complete(messages=[{"role": "user", "content": "test"}], system="sys")
    assert len(p.calls) == 1
    assert p.calls[0]["system"] == "sys"


def test_get_provider_anthropic():
    p = get_provider("anthropic", api_key="fake-key")
    from agentfabric.providers.anthropic import AnthropicProvider
    assert isinstance(p, AnthropicProvider)


def test_get_provider_openai():
    p = get_provider("openai", api_key="fake-key")
    from agentfabric.providers.openai import OpenAIProvider
    assert isinstance(p, OpenAIProvider)


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("unknown_provider")


def test_provider_repr():
    p = MockLLMProvider()
    assert "MockLLMProvider" in repr(p)
    assert "mock-v1" in repr(p)
