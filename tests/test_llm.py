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
        mock_completion.return_value = _mock_response("Paris")
        from utils.llm import chat

        result = chat("gpt-4o", [{"role": "user", "content": "Capital of France?"}])
        self.assertEqual(result, "Paris")

    @patch("utils.llm.litellm.completion")
    def test_passes_model_and_messages(self, mock_completion):
        """chat() should forward model and messages to litellm unchanged."""
        mock_completion.return_value = _mock_response("ok")
        from utils.llm import chat

        messages = [{"role": "user", "content": "hi"}]
        chat("gemini/gemini-2.0-flash", messages)

        mock_completion.assert_called_once_with(
            model="gemini/gemini-2.0-flash",
            messages=messages,
        )

    @patch("utils.llm.litellm.completion")
    def test_passes_extra_kwargs(self, mock_completion):
        """Extra kwargs (temperature, max_tokens) should be forwarded to litellm."""
        mock_completion.return_value = _mock_response("ok")
        from utils.llm import chat

        chat("gpt-4o", [{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=10)

        _, call_kwargs = mock_completion.call_args
        self.assertEqual(call_kwargs["temperature"], 0.0)
        self.assertEqual(call_kwargs["max_tokens"], 10)


class TestComplete(unittest.TestCase):
    @patch("utils.llm.litellm.completion")
    def test_wraps_prompt_as_user_message(self, mock_completion):
        """complete() should wrap the prompt in a user message."""
        mock_completion.return_value = _mock_response("42")
        from utils.llm import complete

        complete("gpt-4o", "What is 6x7?")

        _, call_kwargs = mock_completion.call_args
        self.assertEqual(call_kwargs["messages"][0]["role"], "user")
        self.assertIn("6x7", call_kwargs["messages"][0]["content"])

    @patch("utils.llm.litellm.completion")
    def test_returns_content(self, mock_completion):
        """complete() should return the reply text."""
        mock_completion.return_value = _mock_response("42")
        from utils.llm import complete

        result = complete("gpt-4o", "What is 6x7?")
        self.assertEqual(result, "42")


if __name__ == "__main__":
    unittest.main()
