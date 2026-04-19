"""Integration tests for the AgentFabric top-level API."""

import pytest

from agentfabric import AgentFabric, AgentNetwork
from agentfabric.fabric import FabricNetwork
from tests.conftest import MockLLMProvider, make_blueprint_json


@pytest.fixture
def fabric_network():
    """A FabricNetwork built entirely with mock providers."""
    provider = MockLLMProvider(response=make_blueprint_json())
    return AgentFabric.create(
        "Test Organization",
        provider=provider,
    )


def test_create_returns_fabric_network(fabric_network):
    assert isinstance(fabric_network, FabricNetwork)


def test_fabric_network_meta_role(fabric_network):
    assert fabric_network.meta_role == "Test Organization"


def test_fabric_network_has_agents(fabric_network):
    assert len(fabric_network.agents) == 3


def test_fabric_network_agent_names(fabric_network):
    assert "Director" in fabric_network.agent_names


def test_fabric_network_describe(fabric_network):
    desc = fabric_network.describe()
    assert "Test Organization" in desc


def test_fabric_network_repr(fabric_network):
    r = repr(fabric_network)
    assert "FabricNetwork" in r
    assert "Test Organization" in r


def test_fabric_get_agent(fabric_network):
    agent = fabric_network.get_agent("Director")
    assert agent.name == "Director"


def test_fabric_query_returns_result(fabric_network):
    result = fabric_network.query("Who is in charge?", entry_agent="Director")
    assert result.answer


def test_fabric_mermaid_diagram(fabric_network):
    diagram = fabric_network.mermaid()
    assert "Director" in diagram
    assert "graph" in diagram


@pytest.mark.asyncio
async def test_create_async_returns_fabric_network():
    provider = MockLLMProvider(response=make_blueprint_json())
    network = await AgentFabric.create_async("Test Organization", provider=provider)
    assert isinstance(network, FabricNetwork)
    assert len(network.agents) == 3


@pytest.mark.asyncio
async def test_fabric_query_async(fabric_network):
    result = await fabric_network.query_async("Async question?", entry_agent="Director")
    assert result.answer


def test_fabric_visualize_mermaid_returns_string(fabric_network, capsys):
    out = fabric_network.visualize(backend="mermaid")
    captured = capsys.readouterr()
    assert "graph" in captured.out or "graph" in out


def test_fabric_visualize_unknown_backend_raises(fabric_network):
    with pytest.raises(ValueError, match="Unknown visualization backend"):
        fabric_network.visualize(backend="unknown")
