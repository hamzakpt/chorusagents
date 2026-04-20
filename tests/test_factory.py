"""Tests for the AgentFactory."""

import pytest

from chorusagents.core.factory import AgentFactory
from chorusagents.core.topology import EdgeDirection
from tests.conftest import MockLLMProvider, make_blueprint_json
from chorusagents.core.architect import MetaArchitect


@pytest.fixture
def blueprint(architect_provider):
    architect = MetaArchitect(provider=architect_provider)
    return architect.decompose_sync("Test Organization")


def test_factory_creates_correct_number_of_agents(blueprint):
    factory = AgentFactory(provider=MockLLMProvider())
    agents = factory.build(blueprint)
    assert len(agents) == len(blueprint.agents)


def test_factory_agents_have_correct_names(blueprint):
    factory = AgentFactory(provider=MockLLMProvider())
    agents = factory.build(blueprint)
    expected_names = {a.name for a in blueprint.agents}
    assert set(agents.keys()) == expected_names


def test_factory_wires_bidirectional_neighbors(blueprint):
    factory = AgentFactory(provider=MockLLMProvider())
    agents = factory.build(blueprint)

    for edge in blueprint.edges:
        if edge.direction == EdgeDirection.BIDIRECTIONAL:
            assert edge.target in agents[edge.source].neighbors
            assert edge.source in agents[edge.target].neighbors


def test_factory_wires_unidirectional_neighbors():
    import json
    from chorusagents.core.architect import MetaArchitect
    data = json.loads(make_blueprint_json())
    data["edges"] = [
        {"source": "Director", "target": "Engineer", "direction": "unidirectional", "label": "tasks"}
    ]
    provider = MockLLMProvider(response=json.dumps(data))
    architect = MetaArchitect(provider=provider)
    blueprint = architect.decompose_sync("Test")

    factory = AgentFactory(provider=MockLLMProvider())
    agents = factory.build(blueprint)

    assert "Engineer" in agents["Director"].neighbors
    assert "Director" not in agents["Engineer"].neighbors
