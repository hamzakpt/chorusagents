"""
Example: High School Operations — async API

Demonstrates create_async() and query_async().
"""

import asyncio
import os
from chorusagents import ChorusAgents
from chorusagents.providers import OpenAIProvider


async def main():
    # Initialize provider and ChorusAgents
    provider = OpenAIProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
    )
    fabric = ChorusAgents(provider)

    print("Synthesizing High School Operations network (async)...")
    network = await fabric.create_async("High School Operations")

    print("\n--- Network Structure ---")
    print(network.describe())

    # Target a specific agent
    result = await network.query_async(
        "A student has been absent for 2 weeks and parents aren't responding. "
        "What is the escalation protocol?",
        entry_agent=network.agent_names[0],
    )
    print("\n--- Response ---")
    print(result.answer)
    print(f"\nHandled by: {' → '.join(result.routed_path)}")


if __name__ == "__main__":
    asyncio.run(main())
