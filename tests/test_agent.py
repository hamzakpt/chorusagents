"""Tests for the Agent class."""

import pytest
from chorusagents.core.agent import Agent, AgentRole, AgentMessage, AgentResponse
from tests.conftest import MockLLMProvider


def make_agent(name="TestAgent", sub_role="Tester", provider_response="Test response"):
    role = AgentRole(
        name=name,
        sub_role=sub_role,
        description="A test agent.",
        responsibilities=["Test things"],
        tools=["pytest"],
        constraints=["Must pass tests"],
    )
    provider = MockLLMProvider(response=provider_response)
    return Agent(role=role, provider=provider)


def test_agent_name_and_sub_role():
    agent = make_agent(name="Director", sub_role="Executive")
    assert agent.name == "Director"
    assert agent.sub_role == "Executive"


def test_agent_repr():
    agent = make_agent()
    assert "TestAgent" in repr(agent)


def test_add_neighbor():
    agent = make_agent()
    agent.add_neighbor("OtherAgent")
    assert "OtherAgent" in agent.neighbors


def test_add_neighbor_no_duplicates():
    agent = make_agent()
    agent.add_neighbor("X")
    agent.add_neighbor("X")
    assert agent.neighbors.count("X") == 1


@pytest.mark.asyncio
async def test_agent_process_returns_response():
    agent = make_agent(provider_response="This is the answer.")
    msg = AgentMessage(sender="user", content="What is the answer?")
    response = await agent.process(msg)
    assert isinstance(response, AgentResponse)
    assert response.agent_name == "TestAgent"
    assert response.content == "This is the answer."


@pytest.mark.asyncio
async def test_agent_process_records_history():
    agent = make_agent()
    msg = AgentMessage(sender="user", content="Hello")
    await agent.process(msg)
    assert len(agent._history) == 1


def test_agent_clear_history():
    agent = make_agent()
    agent._history.append(AgentMessage(sender="x", content="y"))
    agent.clear_history()
    assert len(agent._history) == 0


def test_agent_system_prompt_built_from_role():
    agent = make_agent(name="Lawyer", sub_role="Defense Attorney")
    prompt = agent._build_system_prompt()
    assert "Lawyer" in prompt
    assert "Defense Attorney" in prompt


def test_agent_custom_system_prompt_used():
    role = AgentRole(
        name="Custom",
        sub_role="Agent",
        description="Custom.",
        system_prompt="CUSTOM_PROMPT",
    )
    agent = Agent(role=role, provider=MockLLMProvider())
    assert agent._build_system_prompt() == "CUSTOM_PROMPT"
