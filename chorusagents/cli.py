"""Command-line interface for ChorusAgents."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from chorusagents.providers import SUPPORTED_PROVIDERS, get_provider

_PROVIDER_HELP = (
    "LLM provider. Supported: openai (default), anthropic, azure, gemini, "
    "bedrock, ollama, huggingface."
)


def _add_common_provider_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="openai", help=_PROVIDER_HELP)
    p.add_argument("--model", default=None, help="Model ID (provider-specific)")
    p.add_argument("--api-key", default=None, dest="api_key", help="API key / token")
    p.add_argument("--azure-endpoint", default=None, dest="azure_endpoint",
                   help="[Azure] Endpoint URL")
    p.add_argument("--azure-deployment", default=None, dest="azure_deployment",
                   help="[Azure] Deployment name")
    p.add_argument("--api-version", default=None, dest="api_version",
                   help="[Azure] API version")
    p.add_argument("--region", default=None, dest="region_name",
                   help="[Bedrock] AWS region")
    p.add_argument("--ollama-host", default=None, dest="ollama_host",
                   help="[Ollama] Server URL (default: http://localhost:11434)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chorusagents",
        description="Synthesize a multi-agent network from a single role description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OpenAI (default)
  chorusagents create "Criminal Defense Law Firm" --api-key sk-...

  # Anthropic
  chorusagents create "Hospital" --provider anthropic --api-key sk-ant-...

  # Azure OpenAI
  chorusagents create "Law Firm" --provider azure \\
      --azure-endpoint https://my.openai.azure.com/ \\
      --azure-deployment gpt-4o-prod --api-key <azure-key>

  # Google Gemini
  chorusagents create "Research Lab" --provider gemini --api-key AIza...

  # AWS Bedrock (uses ~/.aws credentials)
  chorusagents create "Healthcare" --provider bedrock --region us-east-1

  # Ollama (local, no key needed)
  chorusagents create "Dev Team" --provider ollama --model llama3.1

  # HuggingFace
  chorusagents create "Research" --provider huggingface \\
      --model meta-llama/Meta-Llama-3.1-8B-Instruct --api-key hf_...

  # Query a network
  chorusagents query "Law Firm" "Draft a motion to suppress evidence." \\
      --api-key sk-... --full-report
""",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── create ───────────────────────────────────────────────────────────────
    create_parser = subparsers.add_parser("create", help="Synthesize and describe a network")
    create_parser.add_argument("role", help='Meta-role, e.g. "Criminal Defense Law Firm"')
    _add_common_provider_args(create_parser)
    create_parser.add_argument(
        "--visualize", choices=["mermaid", "graphviz"], default=None,
        help="Render the network diagram",
    )
    create_parser.add_argument("--output", default=None, help="Output file path for diagram")

    # ── query ────────────────────────────────────────────────────────────────
    query_parser = subparsers.add_parser("query", help="Synthesize a network and run a query")
    query_parser.add_argument("role", help="Meta-role")
    query_parser.add_argument("question", help="Question or task")
    _add_common_provider_args(query_parser)
    query_parser.add_argument("--entry-agent", default=None, dest="entry_agent",
                              help="Agent to handle the query first")
    query_parser.add_argument("--broadcast", action="store_true",
                              help="Send query to all agents in parallel")
    query_parser.add_argument("--full-report", action="store_true", dest="full_report",
                              help="Print each agent's response individually")

    # ── providers ─────────────────────────────────────────────────────────────
    subparsers.add_parser("providers", help="List all supported LLM providers")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "providers":
        _list_providers()
        return

    _run_command(args)


def _list_providers() -> None:
    rows = [
        ("openai",       "OpenAIProvider",       "OPENAI_API_KEY",              "gpt-4o"),
        ("anthropic",    "AnthropicProvider",    "ANTHROPIC_API_KEY",           "claude-sonnet-4-6"),
        ("azure",        "AzureOpenAIProvider",  "AZURE_OPENAI_API_KEY+ENDPOINT", "your deployment"),
        ("gemini",       "GeminiProvider",       "GOOGLE_API_KEY",              "gemini-1.5-flash"),
        ("bedrock",      "BedrockProvider",      "AWS credentials",             "claude-3-5-sonnet-v2"),
        ("ollama",       "OllamaProvider",       "(none — local server)",       "llama3.1"),
        ("huggingface",  "HuggingFaceProvider",  "HF_TOKEN",                    "meta-llama/..."),
        ("langchain",    "LangChainProvider",    "(depends on model)",          "any BaseChatModel"),
    ]
    print("\nSupported ChorusAgents Providers\n" + "=" * 62)
    print(f"{'Shorthand':<14} {'Class':<24} {'Credential':<30} {'Default Model'}")
    print("-" * 92)
    for r in rows:
        print(f"{r[0]:<14} {r[1]:<24} {r[2]:<30} {r[3]}")
    print(
        "\nInstall extras as needed:\n"
        "  pip install chorusagents[gemini]       # Google Gemini\n"
        "  pip install chorusagents[bedrock]      # AWS Bedrock\n"
        "  pip install chorusagents[ollama]       # Ollama local\n"
        "  pip install chorusagents[huggingface]  # HuggingFace\n"
        "  pip install langchain-core langchain-openai  # LangChain\n"
    )


def _build_provider_kwargs(args) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if getattr(args, "azure_endpoint", None):
        kwargs["azure_endpoint"] = args.azure_endpoint
    if getattr(args, "azure_deployment", None):
        kwargs["azure_deployment"] = args.azure_deployment
    if getattr(args, "api_version", None):
        kwargs["api_version"] = args.api_version
    if getattr(args, "region_name", None):
        kwargs["region_name"] = args.region_name
    if getattr(args, "ollama_host", None):
        kwargs["host"] = args.ollama_host
    return kwargs


def _run_command(args) -> None:
    try:
        from rich.console import Console
        console = Console()
        def print_info(msg: str) -> None:
            console.print(f"[bold cyan]{msg}[/bold cyan]")
        def print_success(msg: str) -> None:
            console.print(f"[bold green]{msg}[/bold green]")
    except ImportError:
        def print_info(msg: str) -> None:   # type: ignore[misc]
            print(msg)
        def print_success(msg: str) -> None:  # type: ignore[misc]
            print(msg)

    from chorusagents import ChorusAgents

    provider_kwargs = _build_provider_kwargs(args)

    try:
        llm = get_provider(
            provider=args.provider,
            model=getattr(args, "model", None),
            api_key=getattr(args, "api_key", None),
            **provider_kwargs,
        )
        fabric = ChorusAgents(llm)
    except (ValueError, TypeError) as e:
        print(f"Error: Could not initialize provider — {e}", file=sys.stderr)
        print("Run 'chorusagents providers' to see all supported providers.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print_info(f"Synthesizing network for: {args.role!r}  [{args.provider}]")
        network = fabric.create(args.role)
    except (ValueError, TypeError) as e:
        print(f"Error: Failed to parse network blueprint — {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_success(network.describe())

    try:
        if args.command == "create" and getattr(args, "visualize", None):
            network.visualize(backend=args.visualize, output_path=args.output)

        if args.command == "query":
            print_info(f"\nRunning query: {args.question!r}")
            result = network.query(
                args.question,
                entry_agent=getattr(args, "entry_agent", None),
                broadcast=getattr(args, "broadcast", False),
            )
            if getattr(args, "full_report", False):
                print(result.full_report())
            else:
                print_success("\nAnswer:")
                print(result.answer)
    except KeyError as e:
        print(f"Error: Agent not found — {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
