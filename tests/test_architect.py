"""Tests for the MetaArchitect: JSON parsing and blueprint validation."""

import json
import pytest

from chorusagents.core.architect import MetaArchitect
from chorusagents.core.topology import TopologyType
from tests.conftest import MockLLMProvider, make_blueprint_json


@pytest.mark.asyncio
async def test_decompose_returns_blueprint():
    provider = MockLLMProvider(response=make_blueprint_json())
    architect = MetaArchitect(provider=provider)
    blueprint = await architect.decompose("Test Organization")

    assert blueprint.meta_role == "Test Organization"
    assert len(blueprint.agents) == 3
    assert blueprint.topology_type == TopologyType.STAR


@pytest.mark.asyncio
async def test_blueprint_agents_have_required_fields():
    provider = MockLLMProvider(response=make_blueprint_json())
    architect = MetaArchitect(provider=provider)
    blueprint = await architect.decompose("Test Organization")

    for agent in blueprint.agents:
        assert agent.name
        assert agent.sub_role
        assert agent.description
        assert isinstance(agent.responsibilities, list)


@pytest.mark.asyncio
async def test_blueprint_edges_reference_valid_agents():
    provider = MockLLMProvider(response=make_blueprint_json())
    architect = MetaArchitect(provider=provider)
    blueprint = await architect.decompose("Test Organization")

    agent_names = {a.name for a in blueprint.agents}
    for edge in blueprint.edges:
        assert edge.source in agent_names, f"Edge source {edge.source!r} not in agents"
        assert edge.target in agent_names, f"Edge target {edge.target!r} not in agents"


@pytest.mark.asyncio
async def test_json_with_markdown_fences_is_parsed():
    raw = f"```json\n{make_blueprint_json()}\n```"
    provider = MockLLMProvider(response=raw)
    architect = MetaArchitect(provider=provider)
    blueprint = await architect.decompose("Test")
    assert len(blueprint.agents) > 0


@pytest.mark.asyncio
async def test_invalid_topology_falls_back_to_custom():
    data = json.loads(make_blueprint_json())
    data["topology_type"] = "unknown_topology"
    provider = MockLLMProvider(response=json.dumps(data))
    architect = MetaArchitect(provider=provider)
    blueprint = await architect.decompose("Test")
    assert blueprint.topology_type == TopologyType.CUSTOM


@pytest.mark.asyncio
async def test_bad_json_raises_value_error():
    provider = MockLLMProvider(response="This is not JSON at all.")
    architect = MetaArchitect(provider=provider)
    with pytest.raises(ValueError, match="could not parse"):
        await architect.decompose("Test")


def test_decompose_sync(architect_provider):
    architect = MetaArchitect(provider=architect_provider)
    blueprint = architect.decompose_sync("Test Org")
    assert blueprint.meta_role == "Test Organization"
