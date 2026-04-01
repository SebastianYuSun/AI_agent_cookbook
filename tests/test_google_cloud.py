"""Unit tests for GoogleCloudAdapter.

All tests inject a mock genai client — no real Google API key is required.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


def _make_mock_client(response_text: str = "mock response") -> MagicMock:
    """Build a minimal fake google.genai.Client."""
    mock_response = MagicMock()
    mock_response.text = response_text

    # models.generate_content path (single-turn)
    mock_models = MagicMock()
    mock_models.generate_content.return_value = mock_response

    # chats.create(...).send_message(...) path (multi-turn)
    mock_chat_session = MagicMock()
    mock_chat_session.send_message.return_value = mock_response
    mock_chats = MagicMock()
    mock_chats.create.return_value = mock_chat_session

    mock_client = MagicMock()
    mock_client.models = mock_models
    mock_client.chats = mock_chats

    return mock_client


class TestGoogleCloudAdapter(unittest.TestCase):
    def _make_adapter(self, response_text: str = "mock response"):
        """Return a GoogleCloudAdapter backed by a mock genai client."""
        from utils.google_cloud import GoogleCloudAdapter

        mock_client = _make_mock_client(response_text)
        with patch("utils.google_cloud.config.require", return_value="fake-key"):
            adapter = GoogleCloudAdapter(genai_client=mock_client)
        return adapter, mock_client

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_chat_single_user_message(self):
        """A single user message should call generate_content and return text."""
        adapter, _ = self._make_adapter("Paris")
        result = adapter.chat([{"role": "user", "content": "Capital of France?"}])
        self.assertEqual(result, "Paris")

    def test_complete_returns_text(self):
        """complete() should wrap the prompt and return the model reply."""
        adapter, _ = self._make_adapter("42")
        result = adapter.complete("Meaning of life?")
        self.assertEqual(result, "42")

    def test_single_turn_uses_generate_content(self):
        """Single-turn chat (no history) should use models.generate_content."""
        adapter, mock_client = self._make_adapter("ok")

        adapter.chat([{"role": "user", "content": "Hello"}])

        mock_client.models.generate_content.assert_called_once()
        mock_client.chats.create.assert_not_called()

    def test_multi_turn_uses_chat_session(self):
        """When history exists, chats.create + send_message should be used."""
        adapter, mock_client = self._make_adapter("Sure!")

        adapter.chat(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]
        )

        mock_client.chats.create.assert_called_once()
        mock_client.chats.create.return_value.send_message.assert_called_once_with(
            "How are you?"
        )
        mock_client.models.generate_content.assert_not_called()

    def test_no_user_message_raises(self):
        """Passing only system messages (no user turn) should raise ValueError."""
        adapter, _ = self._make_adapter()

        with self.assertRaises(ValueError):
            adapter.chat([{"role": "system", "content": "Be helpful."}])

    def test_default_model_name(self):
        """Default model should be gemini-2.0-flash."""
        adapter, _ = self._make_adapter()
        self.assertEqual(adapter.model_name, "gemini-2.0-flash")

    def test_custom_model_override(self):
        """Model passed to the constructor should override the default."""
        from utils.google_cloud import GoogleCloudAdapter

        mock_client = _make_mock_client()
        with patch("utils.google_cloud.config.require", return_value="fake-key"):
            adapter = GoogleCloudAdapter(model="gemini-2.0-pro", genai_client=mock_client)
        self.assertEqual(adapter.model_name, "gemini-2.0-pro")


if __name__ == "__main__":
    unittest.main()
