from __future__ import annotations

from services.generation_execution_service import GenerationExecutionService


def test_is_ctx_error_true_for_context_window_overflow_message() -> None:
    err = RuntimeError("Requested tokens (913) exceed context window of 512")
    assert GenerationExecutionService._is_ctx_error(err) is True


def test_is_ctx_error_false_for_unrelated_error_message() -> None:
    err = RuntimeError("backend failed")
    assert GenerationExecutionService._is_ctx_error(err) is False


def _raise_sample_error() -> None:
    raise RuntimeError("sample boom")


def test_format_current_exception_summary_is_single_line() -> None:
    try:
        _raise_sample_error()
    except RuntimeError:
        summary = GenerationExecutionService._format_current_exception_summary()

    assert "RuntimeError" in summary
    assert "sample boom" in summary
    assert "\n" not in summary


def test_retry_error_logger_prints_one_line(capsys) -> None:
    service = GenerationExecutionService()
    request = type("Req", (), {"log_prefix": "[LLM Session Chat]"})()
    logger = service._build_retry_error_logger(request)

    try:
        _raise_sample_error()
    except RuntimeError:
        logger()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "Generation exception summary:" in lines[0]
    assert "sample boom" in lines[0]
