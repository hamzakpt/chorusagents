"""
ChorusAgents — Autonomously synthesize a multi-agent network from a single role description.

Quickstart::

    from chorusagents import ChorusAgents
    from chorusagents.providers import OpenAIProvider

    # 1. Initialize your LLM provider
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

    # 2. Initialize ChorusAgents
    fabric = ChorusAgents(provider)

    # 3. Synthesize a network
    network = fabric.create("Criminal Defense Law Firm")

    # 4. Visualize and query
    network.visualize()
    result = network.query("Draft a motion to suppress illegally obtained evidence.")
    print(result.answer)
"""

from chorusagents.chorus import ChorusAgents, ChorusNetwork
from chorusagents.core.agent import Agent
from chorusagents.core.network import AgentNetwork
from chorusagents.core.topology import TopologyType
from chorusagents.providers.base import LLMProvider
from chorusagents.providers import (
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

__all__ = [
    # Core
    "ChorusAgents",
    "ChorusNetwork",
    "Agent",
    "AgentNetwork",
    "TopologyType",
    # Providers
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

__version__ = "0.1.0"
