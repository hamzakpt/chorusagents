"""Tests for AgentNetwork routing and query logic."""

import pytest

from agentfabric.core.network import AgentNetwork
from agentfabric.core.factory import AgentFactory
from agentfabric.core.architect import MetaArchitect
from tests.conftest import MockLLMProvider, make_blueprint_json


@pytest.fixture
def network():
    json_resp = make_blueprint_json()
    architect_provider = MockLLMProvider(response=json_resp)
    agent_provider = MockLLMProvider(response="Mock agent answer.")

    architect = MetaArchitect(provider=architect_provider)
    blueprint = architect.decompose_sync("Test Organization")

    factory = AgentFactory(provider=agent_provider)
    agents = factory.build(blueprint)

    return AgentNetwork(blueprint=blueprint, agents=agents)


def test_network_has_correct_agents(network):
    assert "Director" in network.agents
    assert "Engineer" in network.agents
    assert "Analyst" in network.agents


def test_network_describe_returns_string(network):
    desc = network.describe()
    assert "Test Organization" in desc
    assert "Director" in desc


def test_network_agent_names(network):
    names = network.agent_names
    assert isinstance(names, list)
    assert len(names) == 3


def test_network_get_agent(network):
    agent = network.get_agent("Director")
    assert agent.name == "Director"


def test_network_get_agent_missing_raises(network):
    with pytest.raises(KeyError):
        network.get_agent("NonExistent")


@pytest.mark.asyncio
async def test_network_query_returns_result(network):
    result = await network.query_async("What should we do?", entry_agent="Director")
    assert result.answer == "Mock agent answer."
    assert "Director" in result.routed_path


@pytest.mark.asyncio
async def test_network_broadcast_queries_all_agents(network):
    result = await network.query_async("Broadcast question?", broadcast=True)
    assert len(result.all_responses) == 3
    assert len(result.routed_path) == 3


def test_network_query_sync(network):
    result = network.query("Sync query?", entry_agent="Director")
    assert result.answer


def test_network_find_hub(network):
    hub = network._find_hub()
    assert hub == "Director"


@pytest.mark.asyncio
async def test_query_result_full_report(network):
    result = await network.query_async("Test query?", entry_agent="Engineer")
    report = result.full_report()
    assert "Test query?" in report
    assert "Engineer" in report
