"""Shared fixtures and mock providers for the AgentFabric test suite."""

import json
import pytest

from agentfabric.providers.base import LLMProvider
from agentfabric.core.architect import NetworkBlueprint
from agentfabric.core.topology import TopologyType, TopologyEdge, EdgeDirection


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock LLM provider that returns pre-configured responses
    without making any network calls.
    """

    def __init__(self, response: str = "") -> None:
        self._response = response
        self.calls: list[dict] = []

    @property
    def model(self) -> str:
        return "mock-v1"

    async def complete(self, messages, system="", **kwargs) -> str:
        self.calls.append({"messages": messages, "system": system})
        return self._response


def make_blueprint_json(
    meta_role: str = "Test Organization",
    topology: str = "star",
    agents=None,
    edges=None,
) -> str:
    if agents is None:
        agents = [
            {
                "name": "Director",
                "sub_role": "Executive Director",
                "description": "Oversees all operations.",
                "responsibilities": ["Coordinate teams", "Set strategy"],
                "tools": ["email", "calendar"],
                "constraints": ["Must approve major decisions"],
            },
            {
                "name": "Engineer",
                "sub_role": "Software Engineer",
                "description": "Builds technical systems.",
                "responsibilities": ["Write code", "Review PRs"],
                "tools": ["github", "ide"],
                "constraints": ["Follow coding standards"],
            },
            {
                "name": "Analyst",
                "sub_role": "Data Analyst",
                "description": "Analyses data and generates reports.",
                "responsibilities": ["Build dashboards", "Run queries"],
                "tools": ["sql", "tableau"],
                "constraints": ["Data must be anonymised"],
            },
        ]
    if edges is None:
        edges = [
            {"source": "Director", "target": "Engineer", "direction": "bidirectional", "label": "tasking"},
            {"source": "Director", "target": "Analyst", "direction": "bidirectional", "label": "reporting"},
        ]
    return json.dumps({
        "meta_role": meta_role,
        "topology_type": topology,
        "rationale": "Test rationale.",
        "agents": agents,
        "edges": edges,
    })


@pytest.fixture
def mock_provider():
    return MockLLMProvider(response="This is a mock agent response.")


@pytest.fixture
def architect_provider():
    """Provider that returns a valid blueprint JSON."""
    return MockLLMProvider(response=make_blueprint_json())


@pytest.fixture
def sample_blueprint():
    from agentfabric.core.architect import SubAgentSpec, NetworkBlueprint
    return NetworkBlueprint(
        meta_role="Test Organization",
        topology_type=TopologyType.STAR,
        agents=[
            SubAgentSpec(
                name="Director",
                sub_role="Executive Director",
                description="Oversees all operations.",
                responsibilities=["Coordinate teams"],
                tools=["email"],
                constraints=[],
            ),
            SubAgentSpec(
                name="Engineer",
                sub_role="Software Engineer",
                description="Builds systems.",
                responsibilities=["Write code"],
                tools=["github"],
                constraints=[],
            ),
        ],
        edges=[
            TopologyEdge(
                source="Director",
                target="Engineer",
                direction=EdgeDirection.BIDIRECTIONAL,
                label="tasking",
            )
        ],
        rationale="Hub-and-spoke for clear authority.",
    )
