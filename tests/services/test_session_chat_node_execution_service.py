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


def test_session_chat_node_execution_service_success_path() -> None:
    service = SessionChatNodeExecutionService()
    calls: dict[str, object] = {"logged": None, "executed": None}

    request = SessionChatNodeExecutionRequest(
        model="model.gguf",
        turn_kwargs={"user_text": "hello"},
    )
    deps = SessionChatNodeExecutionDependencies(
        require_llama_cpp_available=lambda: None,
        resolve_valid_model_path=lambda _model, _start_time: "/models/model.gguf",
        get_or_create_model_manager=lambda: "mgr",
        execute_session_chat_turn=lambda **kwargs: (
            calls.update({"executed": kwargs})
            or TurnExecutionResult(assistant_text="ok", generation_succeeded=True)
        ),
        session_chat_error_return=lambda _start, _msg=None: ("",),
        log_session_chat_total=lambda _start, status: calls.update({"logged": status}),
    )

    result = service.run(request=request, dependencies=deps)

    assert result == ("ok",)
    assert isinstance(calls["executed"], dict)
    assert calls["executed"]["model_manager"] == "mgr"
    assert calls["logged"] == "Finished"


def test_session_chat_node_execution_service_failure_with_error_uses_error_return() -> None:
    service = SessionChatNodeExecutionService()
    error_calls: list[tuple[float, object]] = []

    request = SessionChatNodeExecutionRequest(
        model="model.gguf",
        turn_kwargs={"user_text": "hello"},
    )
    deps = SessionChatNodeExecutionDependencies(
        require_llama_cpp_available=lambda: None,
        resolve_valid_model_path=lambda _model, _start_time: "/models/model.gguf",
        get_or_create_model_manager=lambda: "mgr",
        execute_session_chat_turn=lambda **_kwargs: TurnExecutionResult(
            assistant_text="",
            generation_succeeded=False,
            error=RuntimeError("boom"),
        ),
        session_chat_error_return=lambda start, message=None: (
            error_calls.append((start, message)),
            ("",),
        )[1],
        log_session_chat_total=lambda _start, _status: None,
    )

    result = service.run(request=request, dependencies=deps)

    assert result == ("",)
    assert error_calls
    assert "boom" in str(error_calls[0][1])


def test_session_chat_node_execution_service_warns_on_persistence_failure(capsys) -> None:
    service = SessionChatNodeExecutionService()

    request = SessionChatNodeExecutionRequest(
        model="model.gguf",
        turn_kwargs={"user_text": "hello"},
    )
    deps = SessionChatNodeExecutionDependencies(
        require_llama_cpp_available=lambda: None,
        resolve_valid_model_path=lambda _model, _start_time: "/models/model.gguf",
        get_or_create_model_manager=lambda: "mgr",
        execute_session_chat_turn=lambda **_kwargs: TurnExecutionResult(
            assistant_text="ok",
            generation_succeeded=True,
            persistence_succeeded=False,
            persistence_error=RuntimeError("disk full"),
        ),
        session_chat_error_return=lambda _start, _msg=None: ("",),
        log_session_chat_total=lambda _start, _status: None,
    )

    result = service.run(request=request, dependencies=deps)

    assert result == ("ok",)
    captured = capsys.readouterr()
    assert "response generated but history was not saved" in captured.out
    assert "disk full" in captured.out


def test_session_chat_node_execution_service_keeps_existing_model_manager() -> None:
    service = SessionChatNodeExecutionService()
    calls: dict[str, object] = {"factory_calls": 0, "executed": None}

    request = SessionChatNodeExecutionRequest(
        model="model.gguf",
        turn_kwargs={"user_text": "hello", "model_manager": "existing"},
    )

    def _factory():
        calls["factory_calls"] = int(calls["factory_calls"]) + 1
        return "new"

    deps = SessionChatNodeExecutionDependencies(
        require_llama_cpp_available=lambda: None,
        resolve_valid_model_path=lambda _model, _start_time: "/models/model.gguf",
        get_or_create_model_manager=_factory,
        execute_session_chat_turn=lambda **kwargs: (
            calls.update({"executed": kwargs})
            or TurnExecutionResult(assistant_text="ok", generation_succeeded=True)
        ),
        session_chat_error_return=lambda _start, _msg=None: ("",),
        log_session_chat_total=lambda _start, _status: None,
    )

    result = service.run(request=request, dependencies=deps)

    assert result == ("ok",)
    assert calls["factory_calls"] == 0
    assert calls["executed"]["model_manager"] == "existing"


def test_session_chat_node_execution_service_stops_when_model_is_invalid() -> None:
    service = SessionChatNodeExecutionService()
    executed = {"called": False}

    request = SessionChatNodeExecutionRequest(
        model="model.gguf",
        turn_kwargs={"user_text": "hello"},
    )
    deps = SessionChatNodeExecutionDependencies(
        require_llama_cpp_available=lambda: None,
        resolve_valid_model_path=lambda _model, _start_time: None,
        get_or_create_model_manager=lambda: "mgr",
        execute_session_chat_turn=lambda **_kwargs: (
            executed.update({"called": True})
            or TurnExecutionResult(assistant_text="ok", generation_succeeded=True)
        ),
        session_chat_error_return=lambda _start, _msg=None: ("",),
        log_session_chat_total=lambda _start, _status: None,
    )
    result = service.run(request=request, dependencies=deps)

    assert result == ("",)
    assert executed["called"] is False
