from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ContinueRewriteResult:
    user_text_for_model: str
    detected_language: Optional[str] = None
    rewritten: bool = False
