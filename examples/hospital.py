"""
Example: Hospital Emergency Department

Demonstrates broadcast mode and reusing one fabric instance.
Uses Anthropic Claude — swap to any other provider freely.
"""

import os
from chorusagents import ChorusAgents
from chorusagents.providers import AnthropicProvider


def main():
    # Initialize the LLM provider
    provider = AnthropicProvider(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model="claude-sonnet-4-6",
    )

    # Initialize ChorusAgents — reuse for multiple networks
    fabric = ChorusAgents(provider)

    print("Synthesizing Hospital Emergency Department network...")
    network = fabric.create("Hospital Emergency Department")

    print("\n--- Network Structure ---")
    print(network.describe())
    print(f"Topology: {network.topology.value}")
    print(f"Agents:   {network.agent_names}")

    # Broadcast: all agents respond simultaneously
    print("\n--- Broadcast Query ---")
    result = network.query(
        "A 45-year-old patient arrives with chest pain and shortness of breath. "
        "What immediate steps should each department take?",
        broadcast=True,
    )
    print(result.full_report())

    # Save Mermaid diagram
    network.visualize(backend="mermaid", output_path="hospital_network.md")


if __name__ == "__main__":
    main()
