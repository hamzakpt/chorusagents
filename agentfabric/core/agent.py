"""Individual agent definition within an AgentFabric network."""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from pydantic import BaseModel, Field

from agentfabric.utils.logger import get_logger

logger = get_logger(__name__)


class AgentRole(BaseModel):
    """Metadata describing an agent's specialization."""

    name: str
    sub_role: str
    description: str
    responsibilities: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    system_prompt: str = ""


class AgentMessage(BaseModel):
    """A message passed between agents or from a user."""

    sender: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Response produced by an agent."""

    agent_name: str
    content: str
    routed_to: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Agent:
    """
    A single specialized agent within an AgentFabric network.

    Each agent has a defined role, system prompt, and set of responsibilities.
    Agents communicate with each other through the AgentNetwork router.
    """

    def __init__(
        self,
        role: AgentRole,
        provider: Any,
        neighbors: Optional[list[str]] = None,
    ) -> None:
        self.role = role
        self.provider = provider
        self.neighbors: list[str] = neighbors or []
        self._history: list[AgentMessage] = []

    @property
    def name(self) -> str:
        return self.role.name

    @property
    def sub_role(self) -> str:
        return self.role.sub_role

    @property
    def description(self) -> str:
        return self.role.description

    def _build_system_prompt(self) -> str:
        if self.role.system_prompt:
            return self.role.system_prompt

        responsibilities = "\n".join(f"  - {r}" for r in self.role.responsibilities)
        constraints = "\n".join(f"  - {c}" for c in self.role.constraints)
        tools = ", ".join(self.role.tools) if self.role.tools else "none"

        return (
            f"You are {self.role.name}, a {self.role.sub_role}.\n\n"
            f"{self.role.description}\n\n"
            f"Your responsibilities:\n{responsibilities}\n\n"
            f"Your constraints:\n{constraints}\n\n"
            f"Available tools: {tools}\n\n"
            "Always respond in character and stay within your area of expertise. "
            "When a query falls outside your domain, clearly state which colleague "
            "should handle it."
        )

    async def process(self, message: AgentMessage) -> AgentResponse:
        """Process an incoming message and return a response."""
        logger.debug(f"[{self.name}] processing message from '{message.sender}'")

        self._history.append(message)
        system_prompt = self._build_system_prompt()

        response_text = await self.provider.complete(
            system=system_prompt,
            messages=[{"role": "user", "content": message.content}],
        )

        return AgentResponse(
            agent_name=self.name,
            content=response_text,
            metadata={"sub_role": self.sub_role},
        )

    def process_sync(self, message: AgentMessage) -> AgentResponse:
        """Synchronous wrapper around process()."""
        return asyncio.run(self.process(message))

    def add_neighbor(self, agent_name: str) -> None:
        if agent_name not in self.neighbors:
            self.neighbors.append(agent_name)

    def clear_history(self) -> None:
        self._history.clear()

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, sub_role={self.sub_role!r})"
