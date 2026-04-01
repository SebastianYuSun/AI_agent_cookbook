"""Google Cloud adapter — wraps the google-genai SDK (Gemini API)."""

from __future__ import annotations

from google import genai
from google.genai import types

from utils.base import BaseLLMAdapter
from utils import config


class GoogleCloudAdapter(BaseLLMAdapter):
    """
    Adapter for Google Gemini models via the Gemini Developer API.

    Configuration (via .env or environment):
        GOOGLE_API_KEY  — required
        GOOGLE_MODEL    — optional, default: gemini-2.0-flash

    Example:
        >>> adapter = GoogleCloudAdapter()
        >>> print(adapter.complete("What is 2 + 2?"))
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        genai_client=None,
    ):
        self._model_name = model or config.get("GOOGLE_MODEL", self.DEFAULT_MODEL)
        resolved_key = api_key or config.require("GOOGLE_API_KEY")
        # Allow injecting a pre-built client (useful in tests).
        self._client = genai_client or genai.Client(api_key=resolved_key)

    # ── Public interface ──────────────────────────────────────────────────────

    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Send a conversation to Gemini and return the reply.

        Converts the OpenAI message format to Gemini's Content format
        internally, so call-site code stays provider-agnostic.

        Args:
            messages: Conversation in OpenAI message format.
            **kwargs: Passed to ``client.models.generate_content``
                      (e.g. ``config=types.GenerateContentConfig(...)``).

        Returns:
            The model's reply as a plain string.
        """
        system_prompt, history, last_user_message = self._convert_messages(messages)

        generate_config = kwargs.pop("config", None)
        if system_prompt:
            generate_config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                **(generate_config.__dict__ if generate_config else {}),
            )

        if history:
            chat_session = self._client.chats.create(
                model=self._model_name,
                history=history,
                config=generate_config,
            )
            response = chat_session.send_message(last_user_message)
        else:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=last_user_message,
                config=generate_config,
                **kwargs,
            )

        return response.text

    # ── Private helpers ───────────────────────────────────────────────────────

    def _convert_messages(
        self, messages: list[dict]
    ) -> tuple[str | None, list[types.Content], str]:
        """
        Split OpenAI-format messages into the three pieces Gemini needs:
          - system_prompt : combined text of all system messages
          - history       : prior user/model turns (all but the last user msg)
          - last_message  : the final user message to send now
        """
        system_parts: list[str] = []
        turns: list[types.Content] = []

        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                turns.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                turns.append(types.Content(role="model", parts=[types.Part(text=content)]))

        if not turns:
            raise ValueError("At least one user message is required.")

        last_user_text = turns[-1].parts[0].text
        history = turns[:-1]
        system_prompt = "\n".join(system_parts) if system_parts else None
        return system_prompt, history, last_user_text

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name
