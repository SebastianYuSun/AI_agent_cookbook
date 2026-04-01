"""Unit tests for AWSClaudeAdapter.

All tests inject a mock boto3 client — no real AWS credentials are required.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch


def _make_bedrock_response(text: str) -> dict:
    """Construct a fake Bedrock invoke_model response."""
    body_bytes = json.dumps({"content": [{"type": "text", "text": text}]}).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_bytes
    return {"body": mock_body}


class TestAWSClaudeAdapter(unittest.TestCase):
    def _make_adapter(self, mock_response: dict | None = None):
        """Return an AWSClaudeAdapter with an injected mock boto3 client."""
        from utils.aws_claude import AWSClaudeAdapter

        mock_client = MagicMock()
        if mock_response:
            mock_client.invoke_model.return_value = mock_response
        return AWSClaudeAdapter(boto_client=mock_client), mock_client

    # ── Tests ─────────────────────────────────────────────────────────────────

    def test_chat_returns_text(self):
        """chat() should return the first content block's text."""
        adapter, _ = self._make_adapter(_make_bedrock_response("Hello!"))
        result = adapter.chat([{"role": "user", "content": "Hi"}])
        self.assertEqual(result, "Hello!")

    def test_system_message_extracted(self):
        """System messages must be sent in the top-level 'system' field, not in messages."""
        adapter, mock_client = self._make_adapter(_make_bedrock_response("ok"))
        mock_client.invoke_model.return_value = _make_bedrock_response("ok")

        adapter.chat(
            [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "Ahoy!"},
            ]
        )

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        body = json.loads(call_kwargs["body"])
        self.assertEqual(body["system"], "You are a pirate.")
        # The system message should NOT appear inside messages[].
        roles_sent = [m["role"] for m in body["messages"]]
        self.assertNotIn("system", roles_sent)

    def test_no_system_message(self):
        """chat() without a system message should not include a 'system' key."""
        adapter, mock_client = self._make_adapter(_make_bedrock_response("ok"))

        adapter.chat([{"role": "user", "content": "Hi"}])

        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        self.assertNotIn("system", body)

    def test_default_max_tokens(self):
        """Default max_tokens should be 1024."""
        adapter, mock_client = self._make_adapter(_make_bedrock_response("ok"))

        adapter.chat([{"role": "user", "content": "Hi"}])

        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        self.assertEqual(body["max_tokens"], 1024)

    def test_max_tokens_override(self):
        """max_tokens kwarg should override the default."""
        adapter, mock_client = self._make_adapter(_make_bedrock_response("ok"))

        adapter.chat([{"role": "user", "content": "Hi"}], max_tokens=512)

        body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        self.assertEqual(body["max_tokens"], 512)

    def test_default_model_id(self):
        """Adapter should default to the Claude 3.5 Sonnet model."""
        from utils.aws_claude import AWSClaudeAdapter

        adapter = AWSClaudeAdapter(boto_client=MagicMock())
        self.assertIn("claude", adapter.model_id)

    def test_complete_convenience(self):
        """complete() should wrap the prompt and return the model reply."""
        adapter, _ = self._make_adapter(_make_bedrock_response("42"))
        result = adapter.complete("What is 6 × 7?")
        self.assertEqual(result, "42")


if __name__ == "__main__":
    unittest.main()
