"""
Example: Criminal Defense Law Firm

Demonstrates the instance-based API with OpenAI as the provider.
Requires OPENAI_API_KEY in your environment.
"""

import os
from chorusagents import ChorusAgents
from chorusagents.providers import OpenAIProvider


def main():
    # 1. Initialize your chosen LLM provider
    provider = OpenAIProvider(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o",
    )

    # 2. Initialize ChorusAgents with that provider
    fabric = ChorusAgents(provider)

    # 3. Synthesize a network
    print("Synthesizing Criminal Defense Law Firm network...")
    network = fabric.create("Criminal Defense Law Firm")

    # 4. Inspect the structure
    print("\n--- Network Structure ---")
    print(network.describe())

    # 5. Visualize
    print("\n--- Mermaid Diagram ---")
    network.visualize(backend="mermaid")

    # 6. Query
    result = network.query(
        "A client was arrested for drug possession. "
        "The police conducted a search without a warrant. "
        "What legal options do we have?"
    )
    print("\n--- Answer ---")
    print(result.answer)

    print("\n--- Full Multi-Agent Report ---")
    print(result.full_report())


if __name__ == "__main__":
    main()
