"""AWS Bedrock adapter — invokes Anthropic Claude models via Amazon Bedrock."""

from __future__ import annotations

import json

import boto3

from utils.base import BaseLLMAdapter
from utils import config


class AWSClaudeAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic Claude models hosted on AWS Bedrock.

    Configuration (via .env or environment):
        AWS_ACCESS_KEY_ID       — required (or use an IAM role / AWS profile)
        AWS_SECRET_ACCESS_KEY   — required (or use an IAM role / AWS profile)
        AWS_DEFAULT_REGION      — optional, default: us-east-1
        AWS_BEDROCK_MODEL_ID    — optional, default: anthropic.claude-3-5-sonnet-20241022-v2:0

    Example:
        >>> adapter = AWSClaudeAdapter()
        >>> print(adapter.complete("What is 2 + 2?"))
    """

    DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    DEFAULT_REGION = "us-east-1"
    ANTHROPIC_VERSION = "bedrock-2023-05-31"

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        boto_client=None,
    ):
        self._model_id = model_id or config.get("AWS_BEDROCK_MODEL_ID", self.DEFAULT_MODEL)
        self._region = region or config.get("AWS_DEFAULT_REGION", self.DEFAULT_REGION)
        # Allow injecting a pre-built client (useful in tests / LocalStack).
        self._client = boto_client or boto3.client(
            "bedrock-runtime", region_name=self._region
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Invoke a Claude model on Bedrock and return its reply.

        System messages are extracted from *messages* and forwarded in the
        dedicated ``system`` field required by the Bedrock Messages API.

        Args:
            messages: Conversation in OpenAI message format.
            **kwargs: Bedrock-specific overrides:
                      ``max_tokens`` (default 1 024), ``temperature``, ``top_p``.

        Returns:
            The assistant message content as a plain string.
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat_messages = [m for m in messages if m["role"] != "system"]

        body: dict = {
            "anthropic_version": self.ANTHROPIC_VERSION,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "messages": chat_messages,
        }
        if system_parts:
            body["system"] = "\n".join(system_parts)
        body.update(kwargs)

        raw = self._client.invoke_model(
            modelId=kwargs.pop("model_id", self._model_id),
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(raw["body"].read())
        return result["content"][0]["text"]

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_id
