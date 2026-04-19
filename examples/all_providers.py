"""
Example: All Supported LLM Providers

Shows how to initialize AgentFabric with every supported LLM provider.
Uncomment the provider you want to use and set the required credentials.
"""

from agentfabric import AgentFabric

# ─────────────────────────────────────────────────────────────────────────────
# 1. Anthropic / Claude (default)
#    pip install agentfabric
#    export ANTHROPIC_API_KEY="sk-ant-..."
# ─────────────────────────────────────────────────────────────────────────────
from agentfabric.providers import AnthropicProvider

provider = AnthropicProvider(
    api_key="sk-ant-...",           # or set ANTHROPIC_API_KEY env var
    model="claude-sonnet-4-6",      # or "claude-opus-4-7", "claude-haiku-4-5-20251001"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. OpenAI / GPT
#    pip install agentfabric
#    export OPENAI_API_KEY="sk-..."
# ─────────────────────────────────────────────────────────────────────────────
# from agentfabric.providers import OpenAIProvider
# provider = OpenAIProvider(
#     api_key="sk-...",
#     model="gpt-4o",               # or "gpt-4o-mini", "o1", "o3-mini"
# )

# ─────────────────────────────────────────────────────────────────────────────
# 3. Azure OpenAI
#    pip install agentfabric[azure]
#    export AZURE_OPENAI_API_KEY="..."
# ─────────────────────────────────────────────────────────────────────────────
# from agentfabric.providers import AzureOpenAIProvider
# provider = AzureOpenAIProvider(
#     azure_endpoint="https://my-resource.openai.azure.com/",
#     azure_deployment="gpt-4o-prod",   # your deployment name
#     api_key="your-azure-key",
#     api_version="2024-02-01",
# )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Google Gemini
#    pip install agentfabric[gemini]
#    export GOOGLE_API_KEY="AIza..."
# ─────────────────────────────────────────────────────────────────────────────
# from agentfabric.providers import GeminiProvider
# provider = GeminiProvider(
#     api_key="AIza...",
#     model="gemini-1.5-pro",       # or "gemini-2.0-flash", "gemini-2.5-pro"
# )

# ─────────────────────────────────────────────────────────────────────────────
# 5. AWS Bedrock (Claude, Llama, Mistral, Titan, and more)
#    pip install agentfabric[bedrock]
#    Configure AWS credentials: aws configure  (or use IAM role)
# ─────────────────────────────────────────────────────────────────────────────
# from agentfabric.providers import BedrockProvider
# provider = BedrockProvider(
#     model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
#     region_name="us-east-1",
#     # aws_access_key_id="...",    # optional — uses credential chain if omitted
#     # aws_secret_access_key="...",
# )

# ─────────────────────────────────────────────────────────────────────────────
# 6. Ollama — local inference, no API key needed
#    pip install agentfabric[ollama]
#    ollama serve && ollama pull llama3.1
# ─────────────────────────────────────────────────────────────────────────────
# from agentfabric.providers import OllamaProvider
# provider = OllamaProvider(
#     model="llama3.1",             # or "mistral", "phi3", "qwen2.5", "gemma2"
#     host="http://localhost:11434",
# )

# ─────────────────────────────────────────────────────────────────────────────
# 7. HuggingFace Inference API
#    pip install agentfabric[huggingface]
#    export HF_TOKEN="hf_..."
# ─────────────────────────────────────────────────────────────────────────────
# from agentfabric.providers import HuggingFaceProvider
# provider = HuggingFaceProvider(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     api_key="hf_...",
# )

# ─────────────────────────────────────────────────────────────────────────────
# 8. Any LangChain BaseChatModel (Mistral, Cohere, Together AI, etc.)
#    pip install langchain-core langchain-mistralai
# ─────────────────────────────────────────────────────────────────────────────
# from langchain_mistralai import ChatMistralAI
# from agentfabric.providers import LangChainProvider
# provider = LangChainProvider(
#     ChatMistralAI(api_key="...", model="mistral-large-latest")
# )

# ─────────────────────────────────────────────────────────────────────────────
# String shorthand — uses get_provider() internally
# ─────────────────────────────────────────────────────────────────────────────
# network = AgentFabric.create("Law Firm", provider="openai", model="gpt-4o", api_key="sk-...")
# network = AgentFabric.create("Law Firm", provider="gemini", api_key="AIza...")
# network = AgentFabric.create("Law Firm", provider="ollama", model="llama3.1")
# network = AgentFabric.create(
#     "Law Firm", provider="azure",
#     azure_endpoint="https://my.openai.azure.com/",
#     azure_deployment="gpt-4o",
#     api_key="..."
# )

# ─────────────────────────────────────────────────────────────────────────────
# Create network and run a query (same API regardless of provider)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    network = AgentFabric.create("Criminal Defense Law Firm", provider=provider)

    print(network.describe())
    network.visualize(backend="mermaid")

    result = network.query(
        "A client is charged with drug possession. The evidence was seized without a warrant. "
        "What are the strongest legal defenses available?"
    )
    print(result.answer)


if __name__ == "__main__":
    main()
