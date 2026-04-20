"""Integration tests for the ChorusAgents top-level API."""

import pytest

from chorusagents import ChorusAgents, AgentNetwork
from chorusagents.chorus import ChorusNetwork
from chorusagents.providers.base import LLMProvider
from tests.conftest import MockLLMProvider, make_blueprint_json


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def provider():
    return MockLLMProvider(response=make_blueprint_json())


@pytest.fixture
def fabric(provider):
    return ChorusAgents(provider)


@pytest.fixture
def fabric_network(fabric):
    return fabric.create("Test Organization")


# ── ChorusAgents instance construction ────────────────────────────────────────

def test_chorusagents_requires_provider_instance():
    with pytest.raises(TypeError, match="LLMProvider"):
        ChorusAgents("openai")   # string not allowed — must be an instance


def test_chorusagents_requires_provider_not_plain_object():
    with pytest.raises(TypeError, match="LLMProvider"):
        ChorusAgents(42)


def test_chorusagents_accepts_valid_provider(provider):
    fabric = ChorusAgents(provider)
    assert isinstance(fabric, ChorusAgents)


def test_chorusagents_provider_property(provider):
    fabric = ChorusAgents(provider)
    assert fabric.provider is provider


def test_chorusagents_repr(provider):
    fabric = ChorusAgents(provider)
    assert "ChorusAgents" in repr(fabric)


# ── fabric.create() ───────────────────────────────────────────────────────────

def test_create_returns_fabric_network(fabric_network):
    assert isinstance(fabric_network, ChorusNetwork)


def test_create_meta_role(fabric_network):
    assert fabric_network.meta_role == "Test Organization"


def test_create_has_correct_agents(fabric_network):
    assert len(fabric_network.agents) == 3


def test_create_agent_names(fabric_network):
    assert "Director" in fabric_network.agent_names


def test_fabric_reuse_for_multiple_networks(provider):
    """One ChorusAgents instance should create multiple independent networks."""
    fabric = ChorusAgents(provider)
    n1 = fabric.create("Law Firm")
    n2 = fabric.create("Hospital")
    assert n1.meta_role == "Test Organization"  # both come from same mock
    assert n2.meta_role == "Test Organization"
    assert n1 is not n2


# ── ChorusNetwork inspection ──────────────────────────────────────────────────

def test_fabric_network_describe(fabric_network):
    desc = fabric_network.describe()
    assert "Test Organization" in desc


def test_fabric_network_repr(fabric_network):
    r = repr(fabric_network)
    assert "ChorusNetwork" in r
    assert "Test Organization" in r


def test_fabric_get_agent(fabric_network):
    agent = fabric_network.get_agent("Director")
    assert agent.name == "Director"


def test_fabric_get_agent_missing_raises(fabric_network):
    with pytest.raises(KeyError):
        fabric_network.get_agent("NonExistent")


def test_fabric_mermaid_diagram(fabric_network):
    diagram = fabric_network.mermaid()
    assert "Director" in diagram
    assert "graph" in diagram


def test_fabric_topology_property(fabric_network):
    from chorusagents.core.topology import TopologyType
    assert isinstance(fabric_network.topology, TopologyType)


# ── Querying ──────────────────────────────────────────────────────────────────

def test_fabric_query_returns_result(fabric_network):
    result = fabric_network.query("Who is in charge?", entry_agent="Director")
    assert result.answer


def test_fabric_query_broadcast(fabric_network):
    result = fabric_network.query("Status?", broadcast=True)
    assert len(result.all_responses) == 3


# ── Async ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_async_returns_fabric_network():
    provider = MockLLMProvider(response=make_blueprint_json())
    fabric = ChorusAgents(provider)
    network = await fabric.create_async("Test Organization")
    assert isinstance(network, ChorusNetwork)
    assert len(network.agents) == 3


@pytest.mark.asyncio
async def test_fabric_query_async(fabric_network):
    result = await fabric_network.query_async("Async question?", entry_agent="Director")
    assert result.answer


# ── Visualization ─────────────────────────────────────────────────────────────

def test_fabric_visualize_mermaid_returns_string(fabric_network, capsys):
    out = fabric_network.visualize(backend="mermaid")
    captured = capsys.readouterr()
    assert "graph" in captured.out or "graph" in out


def test_fabric_visualize_unknown_backend_raises(fabric_network):
    with pytest.raises(ValueError, match="Unknown visualization backend"):
        fabric_network.visualize(backend="unknown")


# ── Import sanity ─────────────────────────────────────────────────────────────

def test_fabric_importable_from_top_level():
    from chorusagents import ChorusAgents, ChorusNetwork
    assert ChorusAgents
    assert ChorusNetwork
