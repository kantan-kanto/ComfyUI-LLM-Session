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


def test_execute_from_node_inputs_matches_execute_turn_path() -> None:
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

    result = service.execute_from_node_inputs(
        user_text="hello",
        session_id="sid",
        model="model.gguf",
        mmproj="(Auto detect)",
        system_prompt="sys",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        n_gpu_layers=0,
        n_ctx=1024,
        media=None,
        max_turns=12,
        summarize_old_history=True,
        summary_chunk_turns=3,
        max_tokens_summary=128,
        summary_max_chars=1500,
        dynamic_max_tokens=True,
        min_generation_tokens=96,
        safety_margin_tokens=64,
        persistent_cache="off",
        repeat_penalty=1.12,
        repeat_last_n=256,
        rewrite_continue=True,
        runtime_cache="off",
        log_level="timing",
        suppress_backend_logs=True,
        history_dir="",
        reset_session=False,
        stream_to_console=False,
        model_manager=mgr,
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
        strip_assistant_before_reasoning_filter=False,
        include_media_and_stream_in_turn_params=True,
        kv_log_saved_when_not_minimal=False,
        kv_log_unsupported_when_not_minimal=False,
        include_error_in_invalidate_message=False,
        enable_attempt_logging=False,
        log_prefix="[LLM Session Chat]",
        dependencies=deps,
    )

    assert result.generation_succeeded is True
    assert result.assistant_text == "assistant reply"
    assert writes and writes[0][0] == "hist.json"


def test_execute_session_chat_turn_adds_media_stream_params() -> None:
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

    result = service.execute_session_chat_turn(
        user_text="hello",
        session_id="sid",
        model="model.gguf",
        mmproj="(Auto detect)",
        system_prompt="sys",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        n_gpu_layers=0,
        n_ctx=1024,
        media=None,
        max_turns=12,
        summarize_old_history=True,
        summary_chunk_turns=3,
        max_tokens_summary=128,
        summary_max_chars=1500,
        dynamic_max_tokens=True,
        min_generation_tokens=96,
        safety_margin_tokens=64,
        persistent_cache="off",
        repeat_penalty=1.12,
        repeat_last_n=256,
        rewrite_continue=True,
        runtime_cache="off",
        log_level="timing",
        suppress_backend_logs=True,
        history_dir="",
        reset_session=False,
        stream_to_console=False,
        model_manager=mgr,
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
        dependencies=deps,
    )

    assert result.generation_succeeded is True
    params = history["turns"][0]["params"]
    assert params["image_used"] is False
    assert params["image_count"] == 0
    assert params["audio_used"] is False
    assert params["audio_format"] == ""
    assert "streamed" in params
    user = history["turns"][0]["user"]
    assert "image_note" in user
    assert "audio_note" in user
    assert "media_note" not in user


def test_execute_session_chat_turn_records_image_media_params() -> None:
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
    media = type("ImageMedia", (), {"shape": (3, 16, 16, 3)})()
    request = TurnExecutionRequest(**{**request.__dict__, "media": media})

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    params = history["turns"][0]["params"]
    assert params["image_used"] is True
    assert params["image_count"] == 3
    assert params["audio_used"] is False
    assert params["audio_format"] == ""


def test_execute_session_chat_turn_records_audio_media_params() -> None:
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
    request = TurnExecutionRequest(**{**request.__dict__, "media": {"waveform": object(), "sample_rate": 16000}})

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    params = history["turns"][0]["params"]
    assert params["image_used"] is False
    assert params["image_count"] == 0
    assert params["audio_used"] is True
    assert params["audio_format"] == "wav"


def test_execute_dialogue_cycle_turn_strips_and_omits_media_stream_params() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="  assistant reply  ",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )

    result = service.execute_dialogue_cycle_turn(
        user_text="hello",
        session_id="sid",
        model="model.gguf",
        mmproj="(Auto detect)",
        system_prompt="sys",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        n_gpu_layers=0,
        n_ctx=1024,
        media=None,
        max_turns=12,
        summarize_old_history=True,
        summary_chunk_turns=3,
        max_tokens_summary=128,
        summary_max_chars=1500,
        dynamic_max_tokens=True,
        min_generation_tokens=96,
        safety_margin_tokens=64,
        persistent_cache="off",
        repeat_penalty=1.12,
        repeat_last_n=256,
        rewrite_continue=True,
        runtime_cache="off",
        log_level="timing",
        suppress_backend_logs=True,
        history_dir="",
        reset_session=False,
        stream_to_console=False,
        model_manager=mgr,
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
        dependencies=deps,
    )

    assert result.generation_succeeded is True
    assert result.assistant_text == "assistant reply"
    params = history["turns"][0]["params"]
    assert "image_used" not in params
    assert "audio_used" not in params
    assert "streamed" not in params


def test_execute_session_chat_turn_sets_profile_flags() -> None:
    service = TurnExecutionService()
    captured: dict[str, object] = {}

    def _spy_execute_from_node_inputs(**kwargs):
        captured.update(kwargs)
        return "ok"

    service.execute_from_node_inputs = _spy_execute_from_node_inputs  # type: ignore[method-assign]
    result = service.execute_session_chat_turn(user_text="hello")

    assert result == "ok"
    assert captured["strip_assistant_before_reasoning_filter"] is False
    assert captured["include_media_and_stream_in_turn_params"] is True
    assert captured["kv_log_saved_when_not_minimal"] is False
    assert captured["kv_log_unsupported_when_not_minimal"] is False
    assert captured["include_error_in_invalidate_message"] is False
    assert captured["enable_attempt_logging"] is True
    assert captured["log_prefix"] == "[LLM Session Chat]"


def test_execute_dialogue_cycle_turn_sets_profile_flags() -> None:
    service = TurnExecutionService()
    captured: dict[str, object] = {}

    def _spy_execute_from_node_inputs(**kwargs):
        captured.update(kwargs)
        return "ok"

    service.execute_from_node_inputs = _spy_execute_from_node_inputs  # type: ignore[method-assign]
    result = service.execute_dialogue_cycle_turn(user_text="hello")

    assert result == "ok"
    assert captured["strip_assistant_before_reasoning_filter"] is True
    assert captured["include_media_and_stream_in_turn_params"] is False
    assert captured["kv_log_saved_when_not_minimal"] is True
    assert captured["kv_log_unsupported_when_not_minimal"] is True
    assert captured["include_error_in_invalidate_message"] is True
    assert captured["enable_attempt_logging"] is True
    assert captured["log_prefix"] == "[LLM Dialogue Cycle]"


def test_execute_dialogue_cycle_turn_accepts_log_prefix_override() -> None:
    service = TurnExecutionService()
    captured: dict[str, object] = {}

    def _spy_execute_from_node_inputs(**kwargs):
        captured.update(kwargs)
        return "ok"

    service.execute_from_node_inputs = _spy_execute_from_node_inputs  # type: ignore[method-assign]
    result = service.execute_dialogue_cycle_turn(
        user_text="hello",
        log_prefix_override="[LLM Dialogue Cycle A/1]",
    )

    assert result == "ok"
    assert captured["enable_attempt_logging"] is True
    assert captured["log_prefix"] == "[LLM Dialogue Cycle A/1]"
    assert "log_prefix_override" not in captured
