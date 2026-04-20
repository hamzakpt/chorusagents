# ChorusAgents

**Autonomously synthesize a complete multi-agent network from a single role description.**

[![PyPI version](https://badge.fury.io/py/chorusagents.svg)](https://pypi.org/project/chorusagents/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

ChorusAgents flips traditional multi-agent system design: instead of manually defining roles, communication protocols, and topologies, you provide a single high-level **Meta-Role** and the library synthesizes the entire network for you.

```python
from chorusagents import ChorusAgents
from chorusagents.providers import OpenAIProvider

# 1. Initialize your LLM provider
provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

# 2. Initialize ChorusAgents
fabric = ChorusAgents(provider)

# 3. Synthesize a network
network = fabric.create("Criminal Defense Law Firm")

# 4. Visualize and query
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
pip install chorusagents
```

Install extras for additional providers:

```bash
pip install chorusagents[azure]        # Azure OpenAI
pip install chorusagents[gemini]       # Google Gemini
pip install chorusagents[bedrock]      # AWS Bedrock
pip install chorusagents[ollama]       # Ollama (local)
pip install chorusagents[huggingface]  # HuggingFace Inference API
pip install chorusagents[visualization]  # Graphviz SVG/PNG diagrams

# Everything at once:
pip install chorusagents[all]
```

### API Keys

ChorusAgents uses OpenAI by default. Set the key for your chosen provider:

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI (default)
export ANTHROPIC_API_KEY="sk-ant-..."   # Anthropic
export AZURE_OPENAI_API_KEY="..."       # Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://my.openai.azure.com/"
export GOOGLE_API_KEY="AIza..."         # Google Gemini
export HF_TOKEN="hf_..."               # HuggingFace
# AWS Bedrock uses the standard AWS credential chain (aws configure)
# Ollama needs no API key — just run: ollama serve
```

---

## Quickstart

### Three-step pattern

Every usage follows the same pattern — regardless of which LLM you choose:

```python
from chorusagents import ChorusAgents
from chorusagents.providers import OpenAIProvider   # ← swap to any provider

# Step 1: initialize your LLM provider
provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")

# Step 2: initialize ChorusAgents
fabric = ChorusAgents(provider)

# Step 3: synthesize a network
network = fabric.create("Hospital Emergency Department")
```

### Inspect and visualize

```python
print(network.describe())     # agents + topology summary
network.visualize()            # prints Mermaid diagram to stdout
network.visualize(backend="mermaid", output_path="network.md")  # save to file
```

### Query the network

```python
result = network.query(
    "A 45-year-old patient arrives with chest pain. "
    "What should each department do immediately?"
)
print(result.answer)            # primary response
print(result.full_report())     # every agent's individual response
print(result.routed_path)       # which agents handled it
```

### Broadcast mode — all agents respond in parallel

```python
result = network.query("Status check — priorities for each team?", broadcast=True)
print(result.full_report())
```

### Target a specific agent

```python
result = network.query(
    "Review this contract clause for liability issues.",
    entry_agent="ContractReviewer",
)
print(result.answer)
```

### Reuse one fabric for multiple networks

```python
fabric = ChorusAgents(OpenAIProvider(api_key="sk-..."))

law_firm  = fabric.create("Criminal Defense Law Firm")
hospital  = fabric.create("Hospital Emergency Department")
school    = fabric.create("High School Operations")
```

### Async API

```python
import asyncio
from chorusagents import ChorusAgents
from chorusagents.providers import OpenAIProvider

async def main():
    provider = OpenAIProvider(api_key="sk-...")
    fabric   = ChorusAgents(provider)
    network  = await fabric.create_async("Software Engineering Team")
    result   = await network.query_async("Plan the architecture for our new microservice.")
    print(result.answer)

asyncio.run(main())
```

---

## Providers

ChorusAgents supports every major LLM platform. All providers share the same API — just swap the provider object.

| Provider | Install extra | Env var | Default model |
|----------|--------------|---------|---------------|
| `OpenAIProvider` | _(included)_ | `OPENAI_API_KEY` | `gpt-4o` ✦ default |
| `AnthropicProvider` | _(included)_ | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| `AzureOpenAIProvider` | `chorusagents[azure]` | `AZURE_OPENAI_API_KEY` | _(your deployment)_ |
| `GeminiProvider` | `chorusagents[gemini]` | `GOOGLE_API_KEY` | `gemini-1.5-flash` |
| `BedrockProvider` | `chorusagents[bedrock]` | AWS credentials | `claude-3-5-sonnet-v2` |
| `OllamaProvider` | `chorusagents[ollama]` | _(local server)_ | `llama3.1` |
| `HuggingFaceProvider` | `chorusagents[huggingface]` | `HF_TOKEN` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `LangChainProvider` | `langchain-core` + integration | _(depends on model)_ | any `BaseChatModel` |

### OpenAI / GPT (default)

```python
from chorusagents import ChorusAgents
from chorusagents.providers import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...", model="gpt-4o")
fabric   = ChorusAgents(provider)
network  = fabric.create("Law Firm")
```

### Anthropic / Claude

```python
from chorusagents import ChorusAgents
from chorusagents.providers import AnthropicProvider

provider = AnthropicProvider(api_key="sk-ant-...", model="claude-opus-4-7")
fabric   = ChorusAgents(provider)
network  = fabric.create("Law Firm")
```

### Azure OpenAI

```bash
pip install chorusagents[azure]
```

```python
from chorusagents import ChorusAgents
from chorusagents.providers import AzureOpenAIProvider

provider = AzureOpenAIProvider(
    azure_endpoint="https://my-resource.openai.azure.com/",
    azure_deployment="gpt-4o-prod",   # your Azure deployment name
    api_key="your-azure-key",
    api_version="2024-02-01",
)
fabric  = ChorusAgents(provider)
network = fabric.create("Law Firm")
```

### Google Gemini

```bash
pip install chorusagents[gemini]
```

```python
from chorusagents import ChorusAgents
from chorusagents.providers import GeminiProvider

provider = GeminiProvider(api_key="AIza...", model="gemini-1.5-pro")
fabric   = ChorusAgents(provider)
network  = fabric.create("Research Lab")
```

### AWS Bedrock

Supports Anthropic Claude, Meta Llama, Mistral, Amazon Titan, Cohere, and more.

```bash
pip install chorusagents[bedrock]
aws configure   # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
```

```python
from chorusagents import ChorusAgents
from chorusagents.providers import BedrockProvider

provider = BedrockProvider(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region_name="us-east-1",
)
fabric  = ChorusAgents(provider)
network = fabric.create("Healthcare Network")
```

### Ollama (local, no API key)

Run any open-source model locally — Llama 3, Mistral, Phi-3, Qwen, Gemma, DeepSeek, and more.

```bash
pip install chorusagents[ollama]
ollama serve && ollama pull llama3.1
```

```python
from chorusagents import ChorusAgents
from chorusagents.providers import OllamaProvider

provider = OllamaProvider(model="llama3.1")   # no API key needed
fabric   = ChorusAgents(provider)
network  = fabric.create("Software Team")
```

### HuggingFace Inference API

```bash
pip install chorusagents[huggingface]
```

```python
from chorusagents import ChorusAgents
from chorusagents.providers import HuggingFaceProvider

provider = HuggingFaceProvider(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key="hf_...",
)
fabric  = ChorusAgents(provider)
network = fabric.create("Research Lab")
```

### Any LangChain Model

Use any of the 100+ LLMs supported by LangChain (Mistral, Cohere, Together AI, Groq, etc.):

```bash
pip install langchain-core langchain-mistralai  # (or your integration)
```

```python
from langchain_mistralai import ChatMistralAI
from chorusagents import ChorusAgents
from chorusagents.providers import LangChainProvider

provider = LangChainProvider(ChatMistralAI(api_key="...", model="mistral-large-latest"))
fabric   = ChorusAgents(provider)
network  = fabric.create("Research Team")
```

### Custom Provider

Implement `LLMProvider` to use any backend not listed above:

```python
from chorusagents import ChorusAgents, LLMProvider

class MyProvider(LLMProvider):
    @property
    def model(self) -> str:
        return "my-model-v1"

    async def complete(self, messages, system="", **kwargs) -> str:
        # Call your LLM API here
        return "response text"

provider = MyProvider()
fabric   = ChorusAgents(provider)
network  = fabric.create("Research Team")
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

### Graphviz (requires `pip install chorusagents[visualization]`)

```python
# Save as SVG
network.visualize(backend="graphviz", output_path="network", fmt="svg")

# Save as PNG and open immediately
network.visualize(backend="graphviz", fmt="png", view=True)
```

---

## CLI

ChorusAgents ships with a command-line interface:

```bash
# Describe a network (uses OPENAI_API_KEY by default)
chorusagents create "High School Operations" --api-key sk-...

# Create with Mermaid visualization
chorusagents create "Hospital" --visualize mermaid

# Run a query
chorusagents query "Law Firm" "Draft a motion to dismiss." --full-report

# Broadcast to all agents
chorusagents query "Law Firm" "Team status?" --broadcast --full-report

# Different providers
chorusagents create "Startup" --provider openai --model gpt-4o --api-key sk-...
chorusagents create "Lab" --provider gemini --api-key AIza...
chorusagents create "Dev Team" --provider ollama --model llama3.1   # local, free

# Azure OpenAI
chorusagents create "Law Firm" --provider azure \
  --azure-endpoint https://my.openai.azure.com/ \
  --azure-deployment gpt-4o-prod --api-key <azure-key>

# AWS Bedrock (uses ~/.aws credentials)
chorusagents create "Healthcare" --provider bedrock --region us-east-1

# List all supported providers
chorusagents providers
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
git clone https://github.com/hamzakpt/chorusagents
cd chorusagents
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=chorusagents --cov-report=term-missing

# Lint
ruff check chorusagents/
black chorusagents/
```

---

## Project Structure

```
chorusagents/
├── chorusagents/
│   ├── __init__.py          # Public API surface
│   ├── fabric.py            # ChorusAgents + ChorusNetwork (main entry point)
│   ├── cli.py               # CLI (chorusagents command)
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
