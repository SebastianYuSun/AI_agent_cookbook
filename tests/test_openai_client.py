"""Unit tests for OpenAIAdapter.

All tests mock the OpenAI SDK — no real API key is required.
"""

import unittest
from unittest.mock import MagicMock, patch


class TestOpenAIAdapter(unittest.TestCase):
    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_mock_response(self, content: str) -> MagicMock:
        """Build a fake openai ChatCompletion response object."""
        msg = MagicMock()
        msg.content = content
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    # ── Tests ─────────────────────────────────────────────────────────────────

    @patch("utils.openai_client.OpenAI")
    @patch("utils.openai_client.config.require", return_value="fake-key")
    def test_chat_returns_content(self, _mock_require, mock_openai_cls):
        """chat() should return the first choice's message content."""
        from utils.openai_client import OpenAIAdapter

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "4"
        )

        adapter = OpenAIAdapter()
        result = adapter.chat([{"role": "user", "content": "What is 2+2?"}])

        self.assertEqual(result, "4")

    @patch("utils.openai_client.OpenAI")
    @patch("utils.openai_client.config.require", return_value="fake-key")
    def test_complete_wraps_prompt(self, _mock_require, mock_openai_cls):
        """complete() should wrap the prompt in a user message and call chat()."""
        from utils.openai_client import OpenAIAdapter

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Paris"
        )

        adapter = OpenAIAdapter()
        adapter.complete("Capital of France?")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages_sent = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        self.assertEqual(messages_sent[0]["role"], "user")
        self.assertIn("France", messages_sent[0]["content"])

    @patch("utils.openai_client.OpenAI")
    @patch("utils.openai_client.config.require", return_value="fake-key")
    def test_default_model(self, _mock_require, mock_openai_cls):
        """Adapter should default to gpt-4o when no model is specified."""
        from utils.openai_client import OpenAIAdapter

        mock_openai_cls.return_value = MagicMock()
        adapter = OpenAIAdapter()
        self.assertEqual(adapter.model, "gpt-4o")

    @patch("utils.openai_client.OpenAI")
    @patch("utils.openai_client.config.require", return_value="fake-key")
    def test_custom_model_override(self, _mock_require, mock_openai_cls):
        """Model passed to constructor should override the default."""
        from utils.openai_client import OpenAIAdapter

        mock_openai_cls.return_value = MagicMock()
        adapter = OpenAIAdapter(model="gpt-4-turbo")
        self.assertEqual(adapter.model, "gpt-4-turbo")

    @patch("utils.openai_client.OpenAI")
    @patch("utils.openai_client.config.require", return_value="fake-key")
    def test_chat_passes_extra_kwargs(self, _mock_require, mock_openai_cls):
        """Extra kwargs (temperature, max_tokens) should be forwarded to the SDK."""
        from utils.openai_client import OpenAIAdapter

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "ok"
        )

        adapter = OpenAIAdapter()
        adapter.chat([{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=10)

        _, call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs["temperature"], 0.0)
        self.assertEqual(call_kwargs["max_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
