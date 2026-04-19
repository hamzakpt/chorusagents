"""
Example: Criminal Defense Law Firm

Demonstrates the quickstart API and Mermaid visualization.
Requires ANTHROPIC_API_KEY in your environment.
"""

from agentfabric import AgentFabric

def main():
    print("Creating Criminal Defense Law Firm network...")
    network = AgentFabric.create("Criminal Defense Law Firm")

    print("\n--- Network Structure ---")
    print(network.describe())

    print("\n--- Mermaid Diagram ---")
    network.visualize(backend="mermaid")

    print("\n--- Query ---")
    result = network.query(
        "A client was arrested for drug possession. "
        "The police conducted a search without a warrant. "
        "What legal options do we have?"
    )
    print(result.answer)

    print("\n--- Full Multi-Agent Report ---")
    print(result.full_report())


if __name__ == "__main__":
    main()
