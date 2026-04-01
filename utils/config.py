"""
Load .env into the environment at import time.

litellm reads API keys directly from environment variables
(OPENAI_API_KEY, AWS_ACCESS_KEY_ID, GOOGLE_API_KEY, etc.),
so this module just ensures .env is loaded before any litellm call.
"""

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)
