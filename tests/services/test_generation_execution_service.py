from __future__ import annotations

from services.generation_execution_service import GenerationExecutionService


def test_is_ctx_error_true_for_context_window_overflow_message() -> None:
    err = RuntimeError("Requested tokens (913) exceed context window of 512")
    assert GenerationExecutionService._is_ctx_error(err) is True


def test_is_ctx_error_false_for_unrelated_error_message() -> None:
    err = RuntimeError("backend failed")
    assert GenerationExecutionService._is_ctx_error(err) is False