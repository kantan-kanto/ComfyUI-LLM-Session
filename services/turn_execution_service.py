# Service skeleton for shared single-turn execution flow.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class TurnExecutionRequest:
    user_text: str
    session_id: str
    model: str
    mmproj: str
    system_prompt: str
    turn_params: Dict[str, Any] = field(default_factory=dict)
    runtime_options: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Callable[..., Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnExecutionResult:
    assistant_text: str
    history_path: Optional[str] = None
    generation_succeeded: bool = False
    error: Optional[Exception] = None


class TurnExecutionService:
    def execute_turn(self, request: TurnExecutionRequest) -> TurnExecutionResult:
        # Commit 1 intentionally adds only the service shape (not yet wired).
        raise NotImplementedError("Turn execution flow is not wired yet.")
