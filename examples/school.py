"""
Example: High School Operations

Demonstrates async usage and OpenAI provider.
Requires OPENAI_API_KEY in your environment (or swap to AnthropicProvider).
"""

import asyncio
from agentfabric import AgentFabric


async def main():
    print("Creating High School Operations network (async)...")
    network = await AgentFabric.create_async(
        "High School Operations",
        provider="anthropic",   # swap to "openai" if preferred
    )

    print("\n--- Network Structure ---")
    print(network.describe())

    # Target a specific agent directly
    result = await network.query_async(
        "A student has been consistently absent for 2 weeks. "
        "Parents are not responding to emails. What is the protocol?",
        entry_agent=network.agent_names[0],
    )
    print("\n--- Response ---")
    print(result.answer)

    # Show routing path
    print(f"\nHandled by: {' → '.join(result.routed_path)}")


if __name__ == "__main__":
    asyncio.run(main())
