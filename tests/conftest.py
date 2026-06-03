"""Shared pytest fixtures."""

from __future__ import annotations

import os

# Ensure tests never try to hit the real Gemini API.
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("SQLITE_PATH", ":memory:")
