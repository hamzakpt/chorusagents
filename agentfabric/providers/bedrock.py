"""AWS Bedrock provider (Claude, Llama, Mistral, Titan, and more)."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from agentfabric.providers.base import LLMProvider

DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REGION = "us-east-1"


class BedrockProvider(LLMProvider):
    """
    LLM provider backed by AWS Bedrock.

    Supports all Bedrock-hosted models: Anthropic Claude, Meta Llama,
    Mistral, Amazon Titan, Cohere, AI21 Jurassic, and more.

    Requires::

        pip install agentfabric[bedrock]
        # or: pip install boto3

    Authentication uses the standard AWS credential chain:
    environment variables, ``~/.aws/credentials``, IAM roles, etc.

    Parameters
    ----------
    model_id:
        Bedrock model ARN or ID, e.g.
        ``"anthropic.claude-3-5-sonnet-20241022-v2:0"`` or
        ``"meta.llama3-70b-instruct-v1:0"``.
    region_name:
        AWS region. Falls back to ``AWS_DEFAULT_REGION`` env var,
        then ``"us-east-1"``.
    aws_access_key_id:
        AWS access key. Falls back to standard AWS credential chain.
    aws_secret_access_key:
        AWS secret key. Falls back to standard AWS credential chain.
    aws_session_token:
        Optional session token for temporary credentials.
    max_tokens:
        Maximum tokens in the response.

    Example::

        from agentfabric.providers import BedrockProvider
        from agentfabric import AgentFabric

        # Uses default AWS credentials from environment/profile
        provider = BedrockProvider(
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1",
        )
        network = AgentFabric.create("Healthcare Network", provider=provider)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        try:
            import boto3
        except ImportError as e:
            raise ImportError(
                "BedrockProvider requires 'boto3'. "
                "Install it with: pip install agentfabric[bedrock]"
            ) from e

        self._model_id = model_id or DEFAULT_MODEL
        self._max_tokens = max_tokens
        self._region = (
            region_name
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_REGION
        )

        session_kwargs: dict[str, Any] = {}
        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        session = boto3.Session(**session_kwargs)
        self._client = session.client(
            "bedrock-runtime",
            region_name=self._region,
        )

    @property
    def model(self) -> str:
        return self._model_id

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        **kwargs: Any,
    ) -> str:
        import asyncio

        return await asyncio.get_event_loop().run_in_executor(
            None, self._complete_sync, messages, system, kwargs
        )

    def _complete_sync(
        self,
        messages: list[dict[str, str]],
        system: str,
        extra_kwargs: dict,
    ) -> str:
        # Bedrock uses the Converse API (unified across all models)
        converse_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
        ]

        kwargs: dict[str, Any] = {
            "modelId": self._model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": extra_kwargs.pop("max_tokens", self._max_tokens),
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        response = self._client.converse(**kwargs)
        output = response.get("output", {}).get("message", {})
        content_blocks = output.get("content", [])
        if not content_blocks:
            raise RuntimeError(
                f"AWS Bedrock returned empty content for model {self._model_id!r}."
            )
        return content_blocks[0].get("text", "")
