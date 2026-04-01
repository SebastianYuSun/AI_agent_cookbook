"""
Load configuration from environment variables.

Priority order (highest → lowest):
  1. Values already set in the shell environment
  2. .env file in the project root
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up from this file's location to find the project root (.env lives there).
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def require(key: str) -> str:
    """Return the value of *key* or raise a clear error if it is missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set.\n"
            f"Copy .env.example → .env and fill in your credentials."
        )
    return value


def get(key: str, default: str | None = None) -> str | None:
    """Return the value of *key*, or *default* if it is not set."""
    return os.getenv(key, default)
