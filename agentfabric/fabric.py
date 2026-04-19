"""
AgentFabric: the top-level public API.

This module exposes the single entry point users interact with::

    from agentfabric import AgentFabric

    network = AgentFabric.create("Criminal Defense Law Firm")
    network.visualize()
    result = network.query("Draft a motion to suppress evidence.")
    print(result)
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Union

from agentfabric.core.architect import MetaArchitect, NetworkBlueprint
from agentfabric.core.factory import AgentFactory
from agentfabric.core.network import AgentNetwork, NetworkQueryResult
from agentfabric.providers.base import LLMProvider
from agentfabric.providers import get_provider
from agentfabric.utils.logger import get_logger

logger = get_logger(__name__)


class AgentFabric:
    """
    Top-level factory and network manager for AgentFabric.

    Typical usage::

        # Quickest start — uses ANTHROPIC_API_KEY from environment
        network = AgentFabric.create("High School Operations")

        # ── Anthropic / Claude ──────────────────────────────────────────
        from agentfabric.providers import AnthropicProvider
        network = AgentFabric.create(
            "Law Firm",
            provider=AnthropicProvider(api_key="sk-ant-...", model="claude-opus-4-7"),
        )

        # ── OpenAI ──────────────────────────────────────────────────────
        from agentfabric.providers import OpenAIProvider
        network = AgentFabric.create(
            "Law Firm",
            provider=OpenAIProvider(api_key="sk-...", model="gpt-4o"),
        )

        # ── Azure OpenAI ─────────────────────────────────────────────────
        from agentfabric.providers import AzureOpenAIProvider
        network = AgentFabric.create(
            "Law Firm",
            provider=AzureOpenAIProvider(
                azure_endpoint="https://my-resource.openai.azure.com/",
                azure_deployment="gpt-4o-prod",
                api_key="...",
            ),
        )

        # ── Google Gemini ────────────────────────────────────────────────
        from agentfabric.providers import GeminiProvider
        network = AgentFabric.create(
            "Research Lab",
            provider=GeminiProvider(api_key="AIza...", model="gemini-1.5-pro"),
        )

        # ── AWS Bedrock ──────────────────────────────────────────────────
        from agentfabric.providers import BedrockProvider
        network = AgentFabric.create(
            "Healthcare Network",
            provider=BedrockProvider(
                model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                region_name="us-east-1",
            ),
        )

        # ── Ollama (local, no API key) ───────────────────────────────────
        from agentfabric.providers import OllamaProvider
        network = AgentFabric.create(
            "Software Team",
            provider=OllamaProvider(model="llama3.1"),
        )

        # ── HuggingFace ──────────────────────────────────────────────────
        from agentfabric.providers import HuggingFaceProvider
        network = AgentFabric.create(
            "Research Lab",
            provider=HuggingFaceProvider(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                api_key="hf_...",
            ),
        )

        # ── Any LangChain model ──────────────────────────────────────────
        from langchain_mistralai import ChatMistralAI
        from agentfabric.providers import LangChainProvider
        network = AgentFabric.create(
            "Software Team",
            provider=LangChainProvider(ChatMistralAI(api_key="...")),
        )

        # Inspect the network
        print(network.describe())
        network.visualize()

        # Run a query
        result = network.query("A student reported a recurring Wi-Fi issue.")
        print(result.answer)
        print(result.full_report())
    """

    # ------------------------------------------------------------------
    # Class-level factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        meta_role: str,
        provider: Optional[Union[LLMProvider, str]] = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **provider_kwargs,
    ) -> "FabricNetwork":
        """
        Synthesize a multi-agent network from a high-level role description.

        Parameters
        ----------
        meta_role:
            A natural-language description of the organization or role to model.
            Examples: ``"Criminal Defense Law Firm"``, ``"E-commerce Platform"``,
            ``"Hospital Emergency Department"``.
        provider:
            LLM provider to use. One of:

            * A pre-built ``LLMProvider`` instance (``AnthropicProvider``,
              ``OpenAIProvider``, ``AzureOpenAIProvider``, ``GeminiProvider``,
              ``BedrockProvider``, ``OllamaProvider``, ``HuggingFaceProvider``,
              ``LangChainProvider``).
            * A string shorthand: ``"anthropic"``, ``"openai"``, ``"azure"``,
              ``"gemini"``, ``"bedrock"``, ``"ollama"``, ``"huggingface"``.

            Defaults to ``"anthropic"`` (requires ``ANTHROPIC_API_KEY``).
        model:
            Model override. When ``provider`` is a string, this is passed to the
            provider constructor. Ignored if ``provider`` is already an instance.
        api_key:
            API key override. Falls back to environment variables.
        **provider_kwargs:
            Additional kwargs forwarded to the provider constructor.

        Returns
        -------
        FabricNetwork
            The synthesized and ready-to-query agent network.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed into a valid blueprint.
        """
        llm = cls._resolve_provider(provider, model, api_key, **provider_kwargs)
        architect = MetaArchitect(provider=llm)
        factory = AgentFactory(provider=llm)

        blueprint = architect.decompose_sync(meta_role)
        agents = factory.build(blueprint)
        network = AgentNetwork(blueprint=blueprint, agents=agents)

        return FabricNetwork(network=network, blueprint=blueprint)

    @classmethod
    async def create_async(
        cls,
        meta_role: str,
        provider: Optional[Union[LLMProvider, str]] = "anthropic",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **provider_kwargs,
    ) -> "FabricNetwork":
        """Async version of :meth:`create`."""
        llm = cls._resolve_provider(provider, model, api_key, **provider_kwargs)
        architect = MetaArchitect(provider=llm)
        factory = AgentFactory(provider=llm)

        blueprint = await architect.decompose(meta_role)
        agents = factory.build(blueprint)
        network = AgentNetwork(blueprint=blueprint, agents=agents)

        return FabricNetwork(network=network, blueprint=blueprint)

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_provider(
        provider: Optional[Union[LLMProvider, str]],
        model: Optional[str],
        api_key: Optional[str],
        **kwargs,
    ) -> LLMProvider:
        if isinstance(provider, LLMProvider):
            return provider
        provider_name = provider or "anthropic"
        return get_provider(provider=provider_name, model=model, api_key=api_key, **kwargs)


class FabricNetwork:
    """
    A synthesized, ready-to-use multi-agent network.

    This class wraps :class:`AgentNetwork` and :class:`NetworkBlueprint`
    with additional convenience methods like :meth:`visualize`.
    """

    def __init__(self, network: AgentNetwork, blueprint: NetworkBlueprint) -> None:
        self._network = network
        self._blueprint = blueprint

    # ------------------------------------------------------------------
    # Core delegation
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
    ) -> NetworkQueryResult:
        """
        Route a query through the agent network and return the result.

        Parameters
        ----------
        user_query:
            The question or task to send to the network.
        entry_agent:
            Name of the specific agent to handle the query first.
            If omitted, the network selects the best entry point automatically.
        broadcast:
            If True, all agents receive the query simultaneously and responses
            are merged into a single result.

        Returns
        -------
        NetworkQueryResult
            Contains ``.answer`` (primary response text), ``.all_responses``,
            and ``.routed_path`` (which agents handled the query).
        """
        return self._network.query(user_query, entry_agent=entry_agent, broadcast=broadcast)

    async def query_async(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
    ) -> NetworkQueryResult:
        """Async version of :meth:`query`."""
        return await self._network.query_async(user_query, entry_agent=entry_agent, broadcast=broadcast)

    def describe(self) -> str:
        """Return a human-readable summary of the network structure."""
        return self._network.describe()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(
        self,
        backend: str = "mermaid",
        output_path: Optional[str] = None,
        fmt: str = "svg",
        view: bool = False,
    ) -> str:
        """
        Visualize the agent network.

        Parameters
        ----------
        backend:
            ``"mermaid"`` (text-based, no extra deps) or
            ``"graphviz"`` (requires ``pip install agentfabric[visualization]``).
        output_path:
            File path to save the output (without extension for graphviz).
            If omitted, prints to stdout (mermaid) or saves to ``agent_network``
            in the current directory (graphviz).
        fmt:
            Output format for graphviz: ``"svg"``, ``"png"``, or ``"pdf"``.
        view:
            If True and backend is graphviz, open the file automatically.

        Returns
        -------
        str
            The diagram string (mermaid) or output file path (graphviz).
        """
        if backend == "mermaid":
            return self._visualize_mermaid(output_path)
        if backend == "graphviz":
            return self._visualize_graphviz(output_path or "agent_network", fmt, view)
        raise ValueError(f"Unknown visualization backend {backend!r}. Use 'mermaid' or 'graphviz'.")

    def mermaid(self) -> str:
        """Return the Mermaid diagram string directly."""
        from agentfabric.visualization.mermaid import MermaidRenderer
        return MermaidRenderer().render(self._blueprint)

    def _visualize_mermaid(self, output_path: Optional[str]) -> str:
        from agentfabric.visualization.mermaid import MermaidRenderer
        renderer = MermaidRenderer()
        diagram = renderer.render_to_markdown(self._blueprint)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(diagram)
            print(f"Mermaid diagram saved to: {output_path}")
        else:
            print(diagram)
        return diagram

    def _visualize_graphviz(self, output_path: str, fmt: str, view: bool) -> str:
        from agentfabric.visualization.graphviz import GraphvizRenderer
        renderer = GraphvizRenderer()
        return renderer.render_to_file(self._blueprint, output_path, fmt=fmt, view=view)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def meta_role(self) -> str:
        return self._blueprint.meta_role

    @property
    def topology(self):
        return self._blueprint.topology_type

    @property
    def blueprint(self) -> NetworkBlueprint:
        return self._blueprint

    @property
    def agents(self):
        return self._network.agents

    @property
    def agent_names(self) -> list[str]:
        return self._network.agent_names

    def get_agent(self, name: str):
        return self._network.get_agent(name)

    def __repr__(self) -> str:
        return (
            f"FabricNetwork("
            f"meta_role={self.meta_role!r}, "
            f"topology={self.topology.value}, "
            f"agents={self.agent_names})"
        )
