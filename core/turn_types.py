# Shared result types used across core/service helpers.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ContinueRewriteResult:
    user_text_for_model: str
    detected_language: Optional[str] = None
    rewritten: bool = False


@dataclass(frozen=True)
class GenerationRunResult:
    assistant_text: str
    gen_tokens: int
    turns_limit: Optional[int]
    last_err: Optional[Exception]
    succeeded: bool
    non_ctx_error: bool = False
