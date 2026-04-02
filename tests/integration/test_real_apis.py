"""
Integration tests — these call real APIs and require valid credentials in .env.

Run locally with:
    make test-integration

These are intentionally excluded from CI. They verify that your API keys are
configured correctly and that the providers are reachable.
"""

from utils.llm import chat, complete

PING = [{"role": "user", "content": "Say hi"}]


class TestOpenAI:
    def test_chat(self):
        result = chat("gpt-5.4", PING)
        assert isinstance(result, str) and len(result) > 0

    def test_complete(self):
        result = complete("gpt-5.4", "Say hi")
        assert isinstance(result, str) and len(result) > 0


class TestAWSBedrock:
    def test_chat(self):
        result = chat("bedrock/anthropic.claude-opus-4-6-v1", PING)
        assert isinstance(result, str) and len(result) > 0


class TestGoogleGemini:
    def test_chat(self):
        result = chat("gemini/gemini-3.1-pro-preview", PING)
        assert isinstance(result, str) and len(result) > 0
