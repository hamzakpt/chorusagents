# AgentFabric

**Autonomously synthesize a complete multi-agent network from a single role description.**

[![PyPI version](https://badge.fury.io/py/agentfabric.svg)](https://pypi.org/project/agentfabric/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

AgentFabric flips traditional multi-agent system design: instead of manually defining roles, communication protocols, and topologies, you provide a single high-level **Meta-Role** and the library synthesizes the entire network for you.

```python
from agentfabric import AgentFabric

network = AgentFabric.create("Criminal Defense Law Firm")
network.visualize()
result = network.query("Draft a motion to suppress evidence from an illegal search.")
print(result.answer)
```

---

## How It Works

```
Meta-Role Input
      │
      ▼
┌─────────────────┐
│  Meta-Architect │  ← LLM performs functional domain decomposition
│   (LLM Brain)  │
└────────┬────────┘
         │  NetworkBlueprint (agents + topology + edges)
         ▼
┌─────────────────┐
│  Agent Factory  │  ← Instantiates live agents with tailored system prompts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Agent Network  │  ← Routes queries, manages agent-to-agent communication
└────────┬────────┘
         │
         ▼
   Visualization + Query API
```

### Architecture Components

| Component | Description |
|-----------|-------------|
| **Meta-Architect** | LLM-powered orchestrator that decomposes a Meta-Role into sub-agents and determines the optimal communication topology |
| **Network Synthesis Engine** | Selects topology (star, pipeline, mesh, hierarchical, custom) based on domain structure and maps data-dependency edges |
| **Agent Factory** | Dynamically creates agent instances with specialized system prompts, tools, and constraints |
| **Agent Network** | The live runtime that routes queries through the synthesized graph |
| **Visualizer** | Exports the network as Mermaid diagrams or Graphviz SVG/PNG |

### Topology Types

| Topology | Best For |
|----------|----------|
| `star` | One clear coordinator/hub (law firm, hospital with chief) |
| `pipeline` | Sequential handoff workflows (document processing, CI/CD) |
| `mesh` | Fully collaborative peer networks (research teams) |
| `hierarchical` | Layered authority structures (military, corporate) |
| `custom` | Mixed or irregular patterns |

---

## Installation

```bash
pip install agentfabric
```

Install extras for additional providers:

```bash
pip install agentfabric[azure]        # Azure OpenAI
pip install agentfabric[gemini]       # Google Gemini
pip install agentfabric[bedrock]      # AWS Bedrock
pip install agentfabric[ollama]       # Ollama (local)
pip install agentfabric[huggingface]  # HuggingFace Inference API
pip install agentfabric[visualization]  # Graphviz SVG/PNG diagrams

# Everything at once:
pip install agentfabric[all]
```

### API Keys

AgentFabric uses Claude (Anthropic) by default. Set the key for your chosen provider:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Anthropic
export OPENAI_API_KEY="sk-..."          # OpenAI
export AZURE_OPENAI_API_KEY="..."       # Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://my.openai.azure.com/"
export GOOGLE_API_KEY="AIza..."         # Google Gemini
export HF_TOKEN="hf_..."               # HuggingFace
# AWS Bedrock uses the standard AWS credential chain (aws configure)
# Ollama needs no API key — just run: ollama serve
```

---

## Quickstart

### Basic Usage

```python
from agentfabric import AgentFabric

# Synthesize a network
network = AgentFabric.create("Hospital Emergency Department")

# Inspect the structure
print(network.describe())

# Visualize (prints Mermaid diagram)
network.visualize()

# Run a query
result = network.query(
    "A 45-year-old patient arrives with chest pain and shortness of breath. "
    "What immediate steps should each department take?"
)
print(result.answer)
```

### Broadcast Mode

Query all agents simultaneously and get a unified report:

```python
result = network.query("Status check — what are each team's priorities?", broadcast=True)
print(result.full_report())
```

### Target a Specific Agent

```python
result = network.query(
    "Review this contract clause for liability issues.",
    entry_agent="ContractReviewer",
)
print(result.answer)
print(f"Handled by: {' → '.join(result.routed_path)}")
```

### Async API

```python
import asyncio
from agentfabric import AgentFabric

async def main():
    network = await AgentFabric.create_async("Software Engineering Team")
    result = await network.query_async("Plan the architecture for our new microservice.")
    print(result.answer)

asyncio.run(main())
```

---

## Providers

AgentFabric supports every major LLM platform. All providers share the same API — just swap the provider object.

| Provider | Install extra | Env var | Default model |
|----------|--------------|---------|---------------|
| `AnthropicProvider` | _(included)_ | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| `OpenAIProvider` | _(included)_ | `OPENAI_API_KEY` | `gpt-4o` |
| `AzureOpenAIProvider` | `agentfabric[azure]` | `AZURE_OPENAI_API_KEY` | _(your deployment)_ |
| `GeminiProvider` | `agentfabric[gemini]` | `GOOGLE_API_KEY` | `gemini-1.5-flash` |
| `BedrockProvider` | `agentfabric[bedrock]` | AWS credentials | `claude-3-5-sonnet-v2` |
| `OllamaProvider` | `agentfabric[ollama]` | _(local server)_ | `llama3.1` |
| `HuggingFaceProvider` | `agentfabric[huggingface]` | `HF_TOKEN` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `LangChainProvider` | `langchain-core` + integration | _(depends on model)_ | any `BaseChatModel` |

### Anthropic / Claude (default)

```python
from agentfabric.providers import AnthropicProvider

network = AgentFabric.create(
    "Law Firm",
    provider=AnthropicProvider(api_key="sk-ant-...", model="claude-opus-4-7"),
)
# or use the string shorthand:
network = AgentFabric.create("Law Firm", provider="anthropic", model="claude-opus-4-7")
```

### OpenAI / GPT

```python
from agentfabric.providers import OpenAIProvider

network = AgentFabric.create(
    "Law Firm",
    provider=OpenAIProvider(api_key="sk-...", model="gpt-4o"),
)
```

### Azure OpenAI

```python
pip install agentfabric[azure]
```

```python
from agentfabric.providers import AzureOpenAIProvider

network = AgentFabric.create(
    "Law Firm",
    provider=AzureOpenAIProvider(
        azure_endpoint="https://my-resource.openai.azure.com/",
        azure_deployment="gpt-4o-prod",   # your deployment name
        api_key="your-azure-key",
        api_version="2024-02-01",
    ),
)
# CLI:
# agentfabric create "Law Firm" --provider azure \
#   --azure-endpoint https://my.openai.azure.com/ \
#   --azure-deployment gpt-4o-prod --api-key <key>
```

### Google Gemini

```python
pip install agentfabric[gemini]
```

```python
from agentfabric.providers import GeminiProvider

network = AgentFabric.create(
    "Research Lab",
    provider=GeminiProvider(api_key="AIza...", model="gemini-1.5-pro"),
)
# or: provider="gemini", api_key="AIza..."
```

### AWS Bedrock

Supports Anthropic Claude, Meta Llama, Mistral, Amazon Titan, Cohere, and more.

```bash
pip install agentfabric[bedrock]
aws configure   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
```

```python
from agentfabric.providers import BedrockProvider

network = AgentFabric.create(
    "Healthcare Network",
    provider=BedrockProvider(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1",
    ),
)
```

### Ollama (local, no API key)

Run any open-source model locally — Llama 3, Mistral, Phi-3, Qwen, Gemma, DeepSeek, and more.

```bash
pip install agentfabric[ollama]
ollama serve
ollama pull llama3.1
```

```python
from agentfabric.providers import OllamaProvider

network = AgentFabric.create(
    "Software Team",
    provider=OllamaProvider(model="llama3.1"),  # no API key needed
)
```

### HuggingFace Inference API

```bash
pip install agentfabric[huggingface]
```

```python
from agentfabric.providers import HuggingFaceProvider

network = AgentFabric.create(
    "Research Lab",
    provider=HuggingFaceProvider(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key="hf_...",
    ),
)
```

### Any LangChain Model

Use any of the 100+ LLMs supported by LangChain (Mistral, Cohere, Together AI, Groq, etc.):

```bash
pip install langchain-core langchain-mistralai  # (or your integration)
```

```python
from langchain_mistralai import ChatMistralAI
from agentfabric.providers import LangChainProvider

network = AgentFabric.create(
    "Research Team",
    provider=LangChainProvider(
        ChatMistralAI(api_key="...", model="mistral-large-latest")
    ),
)
```

### Custom Provider

Implement `LLMProvider` to use any backend not listed above:

```python
from agentfabric import AgentFabric, LLMProvider

class MyProvider(LLMProvider):
    @property
    def model(self) -> str:
        return "my-model-v1"

    async def complete(self, messages, system="", **kwargs) -> str:
        # Call your LLM API here
        return "response text"

network = AgentFabric.create("Research Team", provider=MyProvider())
```

---

## Visualization

### Mermaid (built-in, no extra deps)

```python
# Print to stdout
network.visualize(backend="mermaid")

# Save to file
network.visualize(backend="mermaid", output_path="network.md")

# Get the diagram string directly
diagram = network.mermaid()
```

Paste the output into [mermaid.live](https://mermaid.live) for interactive viewing.

### Graphviz (requires `pip install agentfabric[visualization]`)

```python
# Save as SVG
network.visualize(backend="graphviz", output_path="network", fmt="svg")

# Save as PNG and open immediately
network.visualize(backend="graphviz", fmt="png", view=True)
```

---

## CLI

AgentFabric ships with a command-line interface:

```bash
# Describe a network (uses ANTHROPIC_API_KEY by default)
agentfabric create "High School Operations"

# Create with Mermaid visualization
agentfabric create "Hospital" --visualize mermaid

# Run a query
agentfabric query "Law Firm" "Draft a motion to dismiss." --full-report

# Broadcast to all agents
agentfabric query "Law Firm" "Team status?" --broadcast --full-report

# Different providers
agentfabric create "Startup" --provider openai --model gpt-4o --api-key sk-...
agentfabric create "Lab" --provider gemini --api-key AIza...
agentfabric create "Dev Team" --provider ollama --model llama3.1   # local, free

# Azure OpenAI
agentfabric create "Law Firm" --provider azure \
  --azure-endpoint https://my.openai.azure.com/ \
  --azure-deployment gpt-4o-prod --api-key <azure-key>

# AWS Bedrock (uses ~/.aws credentials)
agentfabric create "Healthcare" --provider bedrock --region us-east-1

# List all supported providers
agentfabric providers
```

---

## Examples

See the [`examples/`](examples/) directory:

| File | Description |
|------|-------------|
| [`law_firm.py`](examples/law_firm.py) | Criminal defense firm with Mermaid visualization |
| [`hospital.py`](examples/hospital.py) | Emergency department with broadcast mode |
| [`school.py`](examples/school.py) | High school operations with async API |
| [`custom_provider.py`](examples/custom_provider.py) | Custom LLM provider integration |

---

## Development

```bash
git clone https://github.com/yourusername/agentfabric
cd agentfabric
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=agentfabric --cov-report=term-missing

# Lint
ruff check agentfabric/
black agentfabric/
```

---

## Project Structure

```
agentfabric/
├── agentfabric/
│   ├── __init__.py          # Public API surface
│   ├── fabric.py            # AgentFabric + FabricNetwork (main entry point)
│   ├── cli.py               # CLI (agentfabric command)
│   ├── core/
│   │   ├── agent.py         # Agent class, AgentRole, AgentMessage
│   │   ├── architect.py     # MetaArchitect (LLM decomposer)
│   │   ├── factory.py       # AgentFactory (instantiates agents)
│   │   ├── network.py       # AgentNetwork (runtime router)
│   │   └── topology.py      # TopologyType, TopologyEdge
│   ├── providers/
│   │   ├── base.py          # LLMProvider ABC
│   │   ├── anthropic.py     # Claude / Anthropic
│   │   └── openai.py        # OpenAI / GPT
│   ├── visualization/
│   │   ├── mermaid.py       # Mermaid diagram renderer
│   │   └── graphviz.py      # Graphviz SVG/PNG renderer
│   └── utils/
│       └── logger.py        # Logging configuration
├── examples/                # Ready-to-run usage examples
├── tests/                   # Full pytest test suite
└── pyproject.toml           # Package metadata + dependencies
```

---

## Real-World Use Cases

- **Rapid Prototyping** — Test complex business logic without manual agent configuration
- **Education** — Simulate how organizations (hospitals, schools, law firms) function
- **Automation** — Dynamically scale agent "teams" based on task complexity
- **Research** — Explore emergent behaviors in synthesized multi-agent systems

---

## License

MIT — see [LICENSE](LICENSE).

---

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Add tests for your changes
4. Run the test suite: `pytest`
5. Submit a pull request
