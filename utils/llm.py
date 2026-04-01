"""
Thin wrapper around litellm.

litellm handles all provider differences (OpenAI, AWS Bedrock, Google Gemini,
Anthropic, etc.) under a single interface. This module just loads .env and
exposes two simple functions.

Model string examples:
    "gpt-4o"                                          # OpenAI
    "claude-3-5-sonnet-20241022"                      # Anthropic direct
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"  # AWS Bedrock
    "gemini/gemini-2.0-flash"                         # Google Gemini

Full model list: https://docs.litellm.ai/docs/providers
"""

from __future__ import annotations

import litellm

from utils import config  # noqa: F401 — imported for .env side-effect


def chat(model: str, messages: list[dict], **kwargs) -> str:
    """
    Send a list of messages to any LLM and return the reply as a string.

    Args:
        model:    litellm model string (e.g. "gpt-4o", "gemini/gemini-2.0-flash").
        messages: Conversation in OpenAI message format.
        **kwargs: Any parameter supported by litellm.completion
                  (e.g. temperature, max_tokens, tools).

    Returns:
        The assistant's reply as plain text.
    """
    response = litellm.completion(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


def complete(model: str, prompt: str, **kwargs) -> str:
    """
    Convenience wrapper: send a single user prompt and return the reply.

    Args:
        model:  litellm model string.
        prompt: Plain-text user prompt.

    Returns:
        The assistant's reply as plain text.
    """
    return chat(model, [{"role": "user", "content": prompt}], **kwargs)
