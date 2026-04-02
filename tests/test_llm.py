"""Unit tests for utils.llm.

Tests mock litellm.completion — no real API keys required.
We are testing that our thin wrapper correctly extracts the response text
and passes arguments through to litellm unchanged.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _mock_response(content: str) -> MagicMock:
    """Build a fake litellm ModelResponse object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class TestChat(unittest.TestCase):
    @patch("utils.llm.litellm.completion")
    def test_returns_content(self, mock_completion):
        """chat() should return the first choice's message content."""
        mock_completion.return_value = _mock_response("hi")
        from utils.llm import chat

        result = chat("gpt-5.4", [{"role": "user", "content": "hi"}])
        self.assertEqual(result, "hi")

    @patch("utils.llm.litellm.completion")
    def test_passes_model_and_messages(self, mock_completion):
        """chat() should forward model and messages to litellm unchanged."""
        mock_completion.return_value = _mock_response("ok")
        from utils.llm import chat

        messages = [{"role": "user", "content": "hi"}]
        chat("gemini/gemini-3.1-flash-lite-preview", messages)

        mock_completion.assert_called_once_with(
            model="gemini/gemini-3.1-flash-lite-preview",
            messages=messages,
        )

    @patch("utils.llm.litellm.completion")
    def test_passes_extra_kwargs(self, mock_completion):
        """Extra kwargs (temperature, max_tokens) should be forwarded to litellm."""
        mock_completion.return_value = _mock_response("ok")
        from utils.llm import chat

        chat("bedrock/anthropic.claude-opus-4-6-v1", [{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=5)

        _, call_kwargs = mock_completion.call_args
        self.assertEqual(call_kwargs["temperature"], 0.0)
        self.assertEqual(call_kwargs["max_tokens"], 5)


class TestComplete(unittest.TestCase):
    @patch("utils.llm.litellm.completion")
    def test_wraps_prompt_as_user_message(self, mock_completion):
        """complete() should wrap the prompt in a user message."""
        mock_completion.return_value = _mock_response("ok")
        from utils.llm import complete

        complete("gpt-5.4", "hi")

        _, call_kwargs = mock_completion.call_args
        self.assertEqual(call_kwargs["messages"][0]["role"], "user")
        self.assertEqual(call_kwargs["messages"][0]["content"], "hi")

    @patch("utils.llm.litellm.completion")
    def test_returns_content(self, mock_completion):
        """complete() should return the reply text."""
        mock_completion.return_value = _mock_response("ok")
        from utils.llm import complete

        result = complete("gpt-5.4", "hi")
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
