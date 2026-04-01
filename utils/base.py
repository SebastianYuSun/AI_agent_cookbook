"""Abstract base class that all LLM adapters implement."""

from abc import ABC, abstractmethod


class BaseLLMAdapter(ABC):
    """
    Unified interface for every LLM provider.

    All adapters accept messages in the OpenAI Chat format:
        [{"role": "system"|"user"|"assistant", "content": "<text>"}]

    This makes it trivial to swap providers without changing call-site code.
    """

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Send a list of chat messages and return the model's reply as a string.

        Args:
            messages: Conversation history in OpenAI message format.
            **kwargs: Provider-specific overrides (max_tokens, temperature, …).

        Returns:
            The assistant's response text.
        """

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Convenience wrapper: send a single user prompt and return the reply.

        Args:
            prompt: Plain-text user prompt.
            **kwargs: Forwarded to :meth:`chat`.

        Returns:
            The assistant's response text.
        """
        return self.chat([{"role": "user", "content": prompt}], **kwargs)
