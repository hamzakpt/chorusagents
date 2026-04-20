"""Meta-Architect: the LLM brain that decomposes a role into sub-agents."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from chorusagents.core.topology import TopologyType, TopologyEdge, EdgeDirection
from chorusagents.utils.logger import get_logger

logger = get_logger(__name__)


class SubAgentSpec(BaseModel):
    """Specification for a single sub-agent produced by the architect."""

    name: str
    sub_role: str
    description: str
    responsibilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    knowledge_mode: str = "grounded"


class NetworkBlueprint(BaseModel):
    """The full blueprint returned by the Meta-Architect."""

    meta_role: str
    topology_type: TopologyType
    agents: list[SubAgentSpec]
    edges: list[TopologyEdge]
    rationale: str = ""


_ARCHITECT_SYSTEM_PROMPT = """\
You are the Meta-Architect of an autonomous multi-agent system (MAS) framework.

Your task: given a high-level "Meta-Role" (e.g., "School Administration" or
"Criminal Defense Law Firm"), perform a functional domain decomposition to
produce a complete, ready-to-run agent network.

You must return ONLY a valid JSON object — no markdown, no commentary, no code fences.

=== MANDATORY DESIGN RULES ===

1. COORDINATOR AGENT (required in every network)
   - Always include exactly one agent named "Coordinator".
   - sub_role: "Query Router & Synthesiser"
   - Its job: receive every incoming query, decide which specialist agent(s)
     should handle it, collect their responses, and synthesise ONE final answer.
   - In star/hierarchical topologies the Coordinator is the hub node.

2. CONTEXT-AWARENESS (required for every specialist agent)
   - Each specialist agent must have this responsibility:
     "Identify missing critical context (jurisdiction, date, version, etc.) and
      request clarification before answering — never assume."
   - Each specialist agent must have this constraint:
     "knowledge_mode: grounded — do not use pre-training knowledge unless the
      user explicitly says to use general knowledge."

3. DOMAIN-SPECIFIC CONTEXT VARIABLES
   Think carefully about what context variables are critical for this domain:
   - Legal     → jurisdiction (country/state), governing law, incident date
   - Medical   → country/healthcare system, patient demographics
   - Financial → country/region, currency, fiscal year
   - Technical → OS, language/framework version, environment
   - Academic  → country, institution type, admission year
   These must appear as responsibilities in the relevant specialist agents.

4. TOPOLOGY CHOICE
   star         → one clear coordinator hub (most common)
   pipeline     → strict sequential handoff (e.g. intake → review → decision)
   mesh         → every agent peers with every other (e.g. research networks)
   hierarchical → layered authority (e.g. principal → heads → teachers)
   custom       → mixed patterns

=== JSON SCHEMA ===
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
      "constraints": [
        "knowledge_mode: grounded — only use explicitly provided information",
        "Always ask for missing critical context before answering",
        "<additional domain-specific constraints>"
      ]
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

Additional guidelines:
- Produce 3 to 8 specialized agents (scale with domain complexity).
- The Coordinator must have edges TO every specialist agent.
- Edges must reference only agent names defined in the "agents" array.
- Keep names unique, short, and in PascalCase (e.g., LeadAttorney, PaymentDept).
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Inside Jupyter or another running event loop — use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise ImportError(
                    "Running inside an active event loop (e.g. Jupyter). "
                    "Install nest_asyncio to use the sync API:\n"
                    "  pip install nest_asyncio\n"
                    "Or use 'await fabric.create_async(...)' instead."
                )
            return loop.run_until_complete(self.decompose(meta_role))

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

        # Parse agents — infer knowledge_mode from constraints if not explicit
        raw_agents = data.get("agents", [])
        agents: list[SubAgentSpec] = []
        for a in raw_agents:
            if "knowledge_mode" not in a:
                # If any constraint mentions "informed" mode, honour it
                constraints_text = " ".join(a.get("constraints", [])).lower()
                a["knowledge_mode"] = (
                    "informed" if "informed" in constraints_text else "grounded"
                )
            agents.append(SubAgentSpec(**a))

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
