"""
AgentFabric — Autonomously synthesize a multi-agent network from a single role description.

Usage::

    from agentfabric import AgentFabric

    network = AgentFabric.create("Criminal Defense Law Firm")
    network.visualize()
    response = network.query("Draft a motion to suppress illegally obtained evidence.")
    print(response)
"""

from agentfabric.fabric import AgentFabric
from agentfabric.core.agent import Agent
from agentfabric.core.network import AgentNetwork
from agentfabric.core.topology import TopologyType
from agentfabric.providers.base import LLMProvider
from agentfabric.providers import (
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
    "AgentFabric",
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
