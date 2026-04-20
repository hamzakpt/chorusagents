"""Agent Factory: instantiates live Agent objects from a NetworkBlueprint."""

from __future__ import annotations

from typing import Any

from chorusagents.core.agent import Agent, AgentRole
from chorusagents.core.architect import NetworkBlueprint, SubAgentSpec
from chorusagents.core.topology import EdgeDirection
from chorusagents.utils.logger import get_logger

logger = get_logger(__name__)


class AgentFactory:
    """
    Spins up live Agent instances from a NetworkBlueprint.

    Wires neighbor relationships based on the blueprint's edge list so
    that the network knows which agents can talk to which.
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    def build(self, blueprint: NetworkBlueprint) -> dict[str, Agent]:
        """
        Instantiate and wire all agents defined in the blueprint.

        Returns a dict mapping agent name → Agent instance.
        """
        agents: dict[str, Agent] = {}

        # Instantiate each agent
        for spec in blueprint.agents:
            agent = self._create_agent(spec)
            agents[agent.name] = agent
            logger.debug(f"Created agent: {agent}")

        # Wire neighbors from edge list
        for edge in blueprint.edges:
            src = edge.source
            tgt = edge.target

            if src not in agents or tgt not in agents:
                logger.warning(
                    f"Edge references unknown agent(s): {src!r} → {tgt!r}. Skipping."
                )
                continue

            agents[src].add_neighbor(tgt)
            if edge.direction == EdgeDirection.BIDIRECTIONAL:
                agents[tgt].add_neighbor(src)

        logger.info(f"AgentFactory built {len(agents)} agents")
        return agents

    # ------------------------------------------------------------------

    def _create_agent(self, spec: SubAgentSpec) -> Agent:
        role = AgentRole(
            name=spec.name,
            sub_role=spec.sub_role,
            description=spec.description,
            responsibilities=spec.responsibilities,
            tools=spec.tools,
            constraints=spec.constraints,
            knowledge_mode=getattr(spec, "knowledge_mode", "grounded"),
        )
        return Agent(role=role, provider=self.provider)
