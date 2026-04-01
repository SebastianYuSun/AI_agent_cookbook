"""OpenAI adapter — wraps the official openai SDK."""

from __future__ import annotations

from openai import OpenAI

from utils.base import BaseLLMAdapter
from utils import config


class OpenAIAdapter(BaseLLMAdapter):
    """
    Thin adapter around the OpenAI Chat Completions API.

    Configuration (via .env or environment):
        OPENAI_API_KEY   — required
        OPENAI_MODEL     — optional, default: gpt-4o

    Example:
        >>> adapter = OpenAIAdapter()
        >>> print(adapter.complete("What is 2 + 2?"))
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._model = model or config.get("OPENAI_MODEL", self.DEFAULT_MODEL)
        self._client = OpenAI(api_key=api_key or config.require("OPENAI_API_KEY"))

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Call the Chat Completions endpoint.

        Args:
            messages: Conversation in OpenAI message format.
            **kwargs: Any parameter accepted by ``client.chat.completions.create``
                      (e.g. ``temperature``, ``max_tokens``).

        Returns:
            The assistant message content as a plain string.
        """
        response = self._client.chat.completions.create(
            model=kwargs.pop("model", self._model),
            messages=messages,
            **kwargs,
        )
        return response.choices[0].message.content

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        return self._model
