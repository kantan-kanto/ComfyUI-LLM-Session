from __future__ import annotations

from core.turn_types import GenerationRunResult
from services.generation_execution_service import GenerationExecutionService
from services.turn_execution_service import (
    SessionChatNodeExecutionDependencies,
    SessionChatNodeExecutionRequest,
    SessionChatNodeExecutionService,
    TurnExecutionRequest,
    TurnExecutionResult,
    TurnExecutionService,
)
from turn_execution_helpers import (
    DummyManager,
    DummyLogRequest,
    _base_deps,
    _capture_generation_kwargs,
    _make_request,
)
import pytest


def test_execute_turn_success_updates_history_and_writes_file() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )

    result = service.execute_turn(_make_request(deps, mgr))

    assert result.generation_succeeded is True
    assert result.assistant_text == "assistant reply"
    assert len(history["turns"]) == 1
    assert writes and writes[0][0] == "hist.json"


def test_execute_turn_does_not_enable_heartbeat_for_timing_log_level() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )
    observed = _capture_generation_kwargs(deps)

    result = service.execute_turn(_make_request(deps, mgr, log_level="timing"))

    assert result.generation_succeeded is True
    assert observed["heartbeat_logger"] is None


def test_execute_turn_enables_heartbeat_for_debug_log_level() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )
    observed = _capture_generation_kwargs(deps)

    result = service.execute_turn(_make_request(deps, mgr, log_level="debug"))

    assert result.generation_succeeded is True
    assert callable(observed["heartbeat_logger"])


def test_execute_turn_disables_heartbeat_for_debug_stream_to_console() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )
    observed = _capture_generation_kwargs(deps)
    request = TurnExecutionRequest(
        **{
            **_make_request(deps, mgr, log_level="debug").__dict__,
            "stream_to_console": True,
        }
    )

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    assert observed["heartbeat_logger"] is None


def test_attempt_logger_reports_token_limit_and_exact_completion_tokens(capsys) -> None:
    logger = GenerationExecutionService()._build_attempt_logger(DummyLogRequest())
    assert logger is not None

    logger(True, 1, 2.0, 64, 12, None, 10, False)

    out = capsys.readouterr().out
    assert "max_tokens=" not in out
    assert "token_limit=64" in out
    assert "completion_tokens=10" in out
    assert "tokens_per_second=5.00" in out
    assert "turns_limit=12" in out


def test_attempt_logger_reports_estimated_completion_tokens(capsys) -> None:
    logger = GenerationExecutionService()._build_attempt_logger(DummyLogRequest())
    assert logger is not None

    logger(True, 1, 4.0, 64, 12, None, 10, True)

    out = capsys.readouterr().out
    assert "completion_tokens_est=10" in out
    assert "tokens_per_second_est=2.50" in out


def test_execute_turn_passes_advanced_generation_seed_kwargs() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    observed: dict[str, object] = {}
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    generation_result = GenerationRunResult(
        assistant_text="assistant reply",
        gen_tokens=64,
        turns_limit=12,
        last_err=None,
        succeeded=True,
        non_ctx_error=False,
    )
    deps, _writes = _base_deps(history, run_generation_result=generation_result)
    deps["run_generation_with_adaptive_retry"] = lambda **kwargs: observed.update(kwargs) or generation_result
    request = TurnExecutionRequest(
        **{
            **_make_request(deps, mgr).__dict__,
            "advanced_generation_kwargs": {"seed": 123},
        }
    )

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    assert observed["advanced_generation_kwargs"] == {"seed": 123}


def test_execute_turn_passes_advanced_summary_seed_kwargs() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    observed: dict[str, object] = {}
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    generation_result = GenerationRunResult(
        assistant_text="assistant reply",
        gen_tokens=64,
        turns_limit=12,
        last_err=None,
        succeeded=True,
        non_ctx_error=False,
    )
    deps, _writes = _base_deps(history, run_generation_result=generation_result)
    deps["maybe_summarize_history"] = lambda **kwargs: observed.update(kwargs) or history
    request = TurnExecutionRequest(
        **{
            **_make_request(deps, mgr).__dict__,
            "advanced_summary_generation_kwargs": {"seed": 456},
        }
    )

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    assert observed["advanced_generation_kwargs"] == {"seed": 456}


def test_execute_turn_records_advanced_seed_kwargs_in_history_params() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )
    request = TurnExecutionRequest(
        **{
            **_make_request(deps, mgr).__dict__,
            "advanced_generation_kwargs": {"seed": 123},
            "advanced_summary_generation_kwargs": {"seed": 456},
        }
    )

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    params = history["turns"][0]["params"]
    assert params["advanced_generation_kwargs"] == {"seed": 123}
    assert params["advanced_summary_generation_kwargs"] == {"seed": 456}


def test_execute_turn_returns_failure_on_non_ctx_error() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="",
            gen_tokens=32,
            turns_limit=8,
            last_err=RuntimeError("backend failed"),
            succeeded=False,
            non_ctx_error=True,
        ),
    )

    result = service.execute_turn(_make_request(deps, mgr))

    assert result.generation_succeeded is False
    assert result.assistant_text == ""
    assert isinstance(result.error, RuntimeError)
