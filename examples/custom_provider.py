"""
Example: Custom LLM Provider

Shows how to plug in any LLM by subclassing LLMProvider,
then passing the instance to ChorusAgents.
"""

import asyncio
from chorusagents import ChorusAgents, LLMProvider


class MockProvider(LLMProvider):
    """A deterministic mock provider — useful for testing and offline demos."""

    @property
    def model(self) -> str:
        return "mock-v1"

    async def complete(self, messages, system="", **kwargs) -> str:
        user_msg = messages[-1]["content"] if messages else ""
        return (
            f"[MockProvider] Received: {user_msg[:80]}\n"
            "This is a mock response for testing purposes."
        )


async def main():
    # Initialize your custom provider
    provider = MockProvider()

    # Pass it to ChorusAgents
    fabric = ChorusAgents(provider)

    # Synthesize and query — same API as any real provider
    network = await fabric.create_async("Software Engineering Team")
    print(network.describe())

    result = await network.query_async("Plan the architecture for a new REST API.")
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
