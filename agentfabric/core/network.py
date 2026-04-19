"""AgentNetwork: the runtime that routes queries through the synthesized agent graph."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import networkx as nx

from agentfabric.core.agent import Agent, AgentMessage, AgentResponse
from agentfabric.core.architect import NetworkBlueprint
from agentfabric.core.topology import TopologyType, TopologyEdge, EdgeDirection
from agentfabric.utils.logger import get_logger

logger = get_logger(__name__)


class NetworkQueryResult:
    """Aggregated result of running a query through the network."""

    def __init__(
        self,
        query: str,
        primary_response: AgentResponse,
        all_responses: list[AgentResponse],
        routed_path: list[str],
    ) -> None:
        self.query = query
        self.primary_response = primary_response
        self.all_responses = all_responses
        self.routed_path = routed_path

    @property
    def answer(self) -> str:
        return self.primary_response.content

    def __str__(self) -> str:
        return self.answer

    def __repr__(self) -> str:
        return (
            f"NetworkQueryResult(agents_involved={self.routed_path}, "
            f"answer_length={len(self.answer)})"
        )

    def full_report(self) -> str:
        """Return a formatted multi-agent response report."""
        lines = [
            f"Query: {self.query}",
            f"Routing path: {' → '.join(self.routed_path)}",
            "",
        ]
        for resp in self.all_responses:
            lines.append(f"[{resp.agent_name}]")
            lines.append(resp.content)
            lines.append("")
        return "\n".join(lines)


class AgentNetwork:
    """
    The live multi-agent network synthesized by AgentFabric.

    Holds the agent instances, the communication graph, and the routing
    logic that dispatches a user query to the appropriate agent(s).
    """

    def __init__(
        self,
        blueprint: NetworkBlueprint,
        agents: dict[str, Agent],
    ) -> None:
        self.blueprint = blueprint
        self.agents = agents
        self._graph = self._build_graph(blueprint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def meta_role(self) -> str:
        return self.blueprint.meta_role

    @property
    def topology(self) -> TopologyType:
        return self.blueprint.topology_type

    @property
    def agent_names(self) -> list[str]:
        return list(self.agents.keys())

    def get_agent(self, name: str) -> Agent:
        if name not in self.agents:
            raise KeyError(f"No agent named {name!r} in this network.")
        return self.agents[name]

    async def query_async(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
    ) -> NetworkQueryResult:
        """
        Route a user query through the agent network.

        Parameters
        ----------
        user_query:
            The natural-language question or task.
        entry_agent:
            Name of the agent to handle the query first.
            Defaults to the hub agent (star) or first agent (pipeline/others).
        broadcast:
            If True, all agents receive the query and responses are aggregated.
        """
        if broadcast:
            return await self._broadcast(user_query)

        entry = self._resolve_entry(entry_agent)
        return await self._route(user_query, entry)

    def query(
        self,
        user_query: str,
        entry_agent: Optional[str] = None,
        broadcast: bool = False,
    ) -> NetworkQueryResult:
        """Synchronous wrapper around query_async."""
        return asyncio.run(self.query_async(user_query, entry_agent, broadcast))

    def describe(self) -> str:
        """Return a human-readable summary of the network."""
        lines = [
            f"Network: {self.meta_role}",
            f"Topology: {self.topology.value}",
            f"Agents ({len(self.agents)}):",
        ]
        for agent in self.agents.values():
            lines.append(f"  - {agent.name}: {agent.sub_role}")
            if agent.neighbors:
                lines.append(f"      connects to: {', '.join(agent.neighbors)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    async def _route(self, query: str, entry_agent_name: str) -> NetworkQueryResult:
        """Route through the network starting from entry_agent."""
        routed_path: list[str] = []
        all_responses: list[AgentResponse] = []

        current_name = entry_agent_name
        visited: set[str] = set()

        while current_name and current_name not in visited:
            visited.add(current_name)
            routed_path.append(current_name)

            agent = self.agents[current_name]
            message = AgentMessage(
                sender="user" if len(routed_path) == 1 else routed_path[-2],
                content=query,
            )
            response = await agent.process(message)
            all_responses.append(response)

            # Check if the response mentions routing to a neighbor
            next_agent = self._detect_handoff(response.content, agent.neighbors, visited)
            current_name = next_agent

        primary = all_responses[-1] if all_responses else AgentResponse(
            agent_name="network", content="No agents processed the query."
        )
        return NetworkQueryResult(
            query=query,
            primary_response=primary,
            all_responses=all_responses,
            routed_path=routed_path,
        )

    async def _broadcast(self, query: str) -> NetworkQueryResult:
        """Send the query to every agent in parallel and aggregate results."""
        tasks = []
        for name, agent in self.agents.items():
            msg = AgentMessage(sender="user", content=query)
            tasks.append(agent.process(msg))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[AgentResponse] = []
        for name, result in zip(self.agents.keys(), results):
            if isinstance(result, Exception):
                logger.warning(f"Agent {name!r} failed during broadcast: {result}")
                responses.append(AgentResponse(
                    agent_name=name,
                    content=f"[Error: {result}]",
                ))
            else:
                responses.append(result)

        combined = "\n\n".join(
            f"[{r.agent_name}]\n{r.content}" for r in responses
        )
        primary = AgentResponse(
            agent_name="network_broadcast",
            content=combined,
        )
        return NetworkQueryResult(
            query=query,
            primary_response=primary,
            all_responses=responses,
            routed_path=[r.agent_name for r in responses],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_entry(self, entry_agent: Optional[str]) -> str:
        if entry_agent:
            if entry_agent not in self.agents:
                raise KeyError(f"Entry agent {entry_agent!r} not found in network.")
            return entry_agent

        if self.topology == TopologyType.STAR:
            return self._find_hub()

        if self.topology == TopologyType.PIPELINE:
            return self._pipeline_start()

        # Default: first agent alphabetically
        return sorted(self.agents.keys())[0]

    def _find_hub(self) -> str:
        """Return the highest-degree node (the star's center)."""
        if not self.agents:
            raise RuntimeError("Cannot find hub: network has no agents.")
        degrees = list(self._graph.degree())
        if not degrees:
            return next(iter(self.agents))
        return max(degrees, key=lambda x: x[1])[0]

    def _pipeline_start(self) -> str:
        """Return the node with in-degree 0 (pipeline source)."""
        for node, in_deg in self._graph.in_degree():
            if in_deg == 0:
                return node
        return next(iter(self.agents))

    @staticmethod
    def _detect_handoff(
        response_text: str,
        neighbors: list[str],
        visited: set[str],
    ) -> Optional[str]:
        """
        Detect if an agent response explicitly routes to a neighbor.

        Uses word-boundary matching to avoid false positives (e.g.,
        "Engineer" matching "engineering").
        """
        import re
        for neighbor in neighbors:
            if neighbor not in visited:
                pattern = r"\b" + re.escape(neighbor) + r"\b"
                if re.search(pattern, response_text, re.IGNORECASE):
                    return neighbor
        return None

    @staticmethod
    def _build_graph(blueprint: NetworkBlueprint) -> nx.DiGraph:
        G = nx.DiGraph()
        for agent in blueprint.agents:
            G.add_node(agent.name, sub_role=agent.sub_role)
        for edge in blueprint.edges:
            G.add_edge(edge.source, edge.target, label=edge.label)
            if edge.direction == EdgeDirection.BIDIRECTIONAL:
                G.add_edge(edge.target, edge.source, label=edge.label)
        return G

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
