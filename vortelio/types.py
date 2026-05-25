"""Type definitions for the Vortelio Python SDK."""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

# ── Simple dicts returned by API methods ──────────────────────────────────────

ModelInfo = Dict[str, Any]
Message = Dict[str, Any]
EmbeddingResult = Dict[str, Any]
GenerateResult = Dict[str, Any]
ChatResult = Dict[str, Any]


class StreamToken:
    """Single streamed token from generate/chat."""
    __slots__ = ("token", "done", "raw")

    def __init__(self, token: str, done: bool, raw: Dict[str, Any]) -> None:
        self.token = token
        self.done = done
        self.raw = raw

    def __str__(self) -> str:
        return self.token

    def __repr__(self) -> str:
        return f"StreamToken({self.token!r}, done={self.done})"
