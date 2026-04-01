"""
Integration tests — these call real APIs and require valid credentials in .env.

Run locally with:
    make test-integration

These are intentionally excluded from CI. They verify that your API keys are
configured correctly and that the providers are reachable.
"""

import pytest
from utils.llm import chat, complete

# A prompt designed to produce a short, deterministic-ish response.
PING = [{"role": "user", "content": "Reply with exactly one word: hello"}]


class TestOpenAI:
    def test_chat(self):
        result = chat("gpt-4o", PING)
        assert isinstance(result, str) and len(result) > 0

    def test_complete(self):
        result = complete("gpt-4o", "Reply with exactly one word: hello")
        assert isinstance(result, str) and len(result) > 0


class TestAWSBedrock:
    def test_chat(self):
        result = chat("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", PING)
        assert isinstance(result, str) and len(result) > 0


class TestGoogleGemini:
    def test_chat(self):
        result = chat("gemini/gemini-2.0-flash", PING)
        assert isinstance(result, str) and len(result) > 0
