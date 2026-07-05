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


def test_execute_turn_returns_failure_when_model_placeholder() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="unused",
            gen_tokens=1,
            turns_limit=1,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )
    deps["is_no_models_placeholder"] = lambda _model: True

    result = service.execute_turn(_make_request(deps, mgr))

    assert result.generation_succeeded is False
    assert result.assistant_text == ""


def test_execute_turn_requires_vision_when_media_is_present() -> None:
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
    request = _make_request(deps, mgr)
    request = TurnExecutionRequest(**{**request.__dict__, "media": object(), "mmproj": "(Auto-detect)"})

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    assert mgr.last_load_kwargs["vision_required"] is True


def test_execute_turn_validates_media_before_model_load() -> None:
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
    deps["validate_chat_media"] = lambda **_kwargs: (_ for _ in ()).throw(
        ValueError("AUDIO media is currently supported only for Gemma 4 models.")
    )
    request = _make_request(deps, mgr)
    request = TurnExecutionRequest(**{**request.__dict__, "media": object(), "mmproj": "(Auto-detect)"})

    result = service.execute_turn(request)

    assert result.generation_succeeded is False
    assert isinstance(result.error, ValueError)
    assert "Gemma 4" in str(result.error)
    assert mgr.loaded is False


def test_execute_turn_wraps_message_build_errors() -> None:
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
    deps["build_chat_messages"] = lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad media shape"))

    result = service.execute_turn(_make_request(deps, mgr))

    assert result.generation_succeeded is False
    assert isinstance(result.error, ValueError)
    assert "bad media shape" in str(result.error)


def test_execute_turn_requires_vision_when_mmproj_is_explicit() -> None:
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
    request = _make_request(deps, mgr)
    request = TurnExecutionRequest(**{**request.__dict__, "mmproj": "mmproj-gemma4.gguf"})

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    assert mgr.last_load_kwargs["vision_required"] is True


def test_execute_turn_does_not_require_vision_for_auto_mmproj_without_media() -> None:
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
    request = _make_request(deps, mgr)
    request = TurnExecutionRequest(**{**request.__dict__, "mmproj": "(Auto-detect)"})

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    assert mgr.last_load_kwargs["vision_required"] is False
