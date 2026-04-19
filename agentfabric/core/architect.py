"""Meta-Architect: the LLM brain that decomposes a role into sub-agents."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from agentfabric.core.topology import TopologyType, TopologyEdge, EdgeDirection
from agentfabric.utils.logger import get_logger

logger = get_logger(__name__)


class SubAgentSpec(BaseModel):
    """Specification for a single sub-agent produced by the architect."""

    name: str
    sub_role: str
    description: str
    responsibilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class NetworkBlueprint(BaseModel):
    """The full blueprint returned by the Meta-Architect."""

    meta_role: str
    topology_type: TopologyType
    agents: list[SubAgentSpec]
    edges: list[TopologyEdge]
    rationale: str = ""


_ARCHITECT_SYSTEM_PROMPT = """\
You are the Meta-Architect of an autonomous multi-agent system (MAS) framework.

Your task: given a high-level "Meta-Role" (e.g., "Criminal Defense Law Firm"), perform
a functional domain decomposition to produce a complete, ready-to-run agent network.

You must return ONLY a valid JSON object — no markdown, no commentary, no code fences.

Required JSON schema:
{
  "meta_role": "<the original role string>",
  "topology_type": "<star|pipeline|mesh|hierarchical|custom>",
  "rationale": "<1-2 sentences explaining topology choice>",
  "agents": [
    {
      "name": "<PascalCase agent name>",
      "sub_role": "<concise job title>",
      "description": "<what this agent does in 1-2 sentences>",
      "responsibilities": ["<responsibility 1>", "..."],
      "tools": ["<tool 1>", "..."],
      "constraints": ["<constraint 1>", "..."]
    }
  ],
  "edges": [
    {
      "source": "<AgentName>",
      "target": "<AgentName>",
      "direction": "<bidirectional|unidirectional>",
      "label": "<brief description of the communication>"
    }
  ]
}

Guidelines:
- Produce 3 to 8 specialized agents (scale with domain complexity).
- Choose topology thoughtfully:
    star        → one clear coordinator/hub
    pipeline    → strict sequential handoff
    mesh        → every agent needs peer access
    hierarchical → layered authority structure
    custom      → mixed or irregular patterns
- Edges must reference only agent names defined in the "agents" array.
- Keep names unique, short, and in PascalCase (e.g., LeadAttorney, ITSupport).
- Return ONLY the JSON object.
"""


_DECOMPOSE_PROMPT_TEMPLATE = """\
Meta-Role: {meta_role}

Decompose this role into a specialized multi-agent network.
Follow the schema exactly and return ONLY the JSON object.
"""


class MetaArchitect:
    """
    Uses an LLM to decompose a Meta-Role into a NetworkBlueprint.

    The architect performs functional domain analysis and determines
    both the agents needed and the optimal communication topology.
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    async def decompose(self, meta_role: str) -> NetworkBlueprint:
        """Send the meta-role to the LLM and parse the returned blueprint."""
        logger.info(f"MetaArchitect decomposing role: '{meta_role}'")

        prompt = _DECOMPOSE_PROMPT_TEMPLATE.format(meta_role=meta_role)
        raw = await self.provider.complete(
            system=_ARCHITECT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        blueprint = self._parse_blueprint(raw, meta_role)
        logger.info(
            f"Blueprint ready: {len(blueprint.agents)} agents, "
            f"topology={blueprint.topology_type.value}"
        )
        return blueprint

    def decompose_sync(self, meta_role: str) -> NetworkBlueprint:
        import asyncio
        return asyncio.run(self.decompose(meta_role))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_blueprint(self, raw: str, meta_role: str) -> NetworkBlueprint:
        data = self._extract_json(raw)

        # Normalise topology string
        topology_str = data.get("topology_type", "custom").lower()
        try:
            topology = TopologyType(topology_str)
        except ValueError:
            topology = TopologyType.CUSTOM

        # Parse agents
        agents = [SubAgentSpec(**a) for a in data.get("agents", [])]

        if len(agents) < 2:
            raise ValueError(
                f"MetaArchitect produced fewer than 2 agents ({len(agents)}). "
                "The LLM response may be malformed. Please try again."
            )

        # Parse edges
        edges: list[TopologyEdge] = []
        for e in data.get("edges", []):
            direction_str = e.get("direction", "bidirectional").lower()
            direction = (
                EdgeDirection.BIDIRECTIONAL
                if direction_str == "bidirectional"
                else EdgeDirection.UNIDIRECTIONAL
            )
            edges.append(
                TopologyEdge(
                    source=e["source"],
                    target=e["target"],
                    direction=direction,
                    label=e.get("label", ""),
                )
            )

        return NetworkBlueprint(
            meta_role=data.get("meta_role", meta_role),
            topology_type=topology,
            agents=agents,
            edges=edges,
            rationale=data.get("rationale", ""),
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract a JSON object from LLM output robustly."""
        text = text.strip()

        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Find the outermost { ... } block by brace counting (safer than greedy regex)
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        raise ValueError(
            f"MetaArchitect could not parse a valid JSON blueprint from LLM response.\n"
            f"Raw output (first 500 chars): {text[:500]}"
        )
