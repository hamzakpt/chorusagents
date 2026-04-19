"""
Example: Hospital Emergency Department

Demonstrates broadcast mode and graphviz visualization.
Requires ANTHROPIC_API_KEY in your environment.
"""

from agentfabric import AgentFabric

def main():
    print("Creating Hospital Emergency Department network...")
    network = AgentFabric.create("Hospital Emergency Department")

    print("\n--- Network Structure ---")
    print(network.describe())
    print(f"\nTopology: {network.topology.value}")
    print(f"Agents: {network.agent_names}")

    # Broadcast mode: all agents answer simultaneously
    print("\n--- Broadcast Query ---")
    result = network.query(
        "A 45-year-old patient arrives with chest pain and shortness of breath. "
        "What immediate steps should each department take?",
        broadcast=True,
    )
    print(result.full_report())

    # Save Mermaid diagram to file
    network.visualize(backend="mermaid", output_path="hospital_network.md")


if __name__ == "__main__":
    main()
