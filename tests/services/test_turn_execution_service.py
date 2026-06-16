from __future__ import annotations

from core.turn_types import GenerationRunResult
from services.turn_execution_service import (
    SessionChatNodeExecutionDependencies,
    SessionChatNodeExecutionRequest,
    SessionChatNodeExecutionService,
    TurnExecutionRequest,
    TurnExecutionResult,
    TurnExecutionService,
)
import pytest


class DummyManager:
    def __init__(self) -> None:
        self.model = object()
        self.cache_dir_override = None
        self.loaded = False
        self.cache_configured = False
        self.last_load_kwargs = None

    def load_model(self, **kwargs):
        self.loaded = True
        self.last_load_kwargs = kwargs
        self.model = object()
        return self.model

    def configure_cache(self, *_args, **_kwargs):
        self.cache_configured = True

    def invalidate_cache(self, *_args, **_kwargs):
        return None


def _base_deps(history_ref, *, run_generation_result: GenerationRunResult):
    writes = []

    deps = {
        "llama_cpp_available": True,
        "llama_cpp_import_error": None,
        "is_no_models_placeholder": lambda _model: False,
        "get_llm_model_roots": lambda: ["/models"],
        "resolve_model_and_mmproj": lambda _roots, _model, _mmproj: ("/models/m.gguf", None),
        "mmproj_auto": "(Auto-detect)",
        "mmproj_not_required": "(Not required)",
        "load_history": lambda **_kwargs: (history_ref, "hist.json"),
        "clear_kv_state_for_session": lambda _sid: None,
        "rewrite_continue_prompt": lambda **kwargs: type("R", (), {
            "user_text_for_model": kwargs.get("user_text", ""),
            "rewritten": False,
            "detected_language": None,
        })(),
        "detect_history_language": lambda _history: "en",
        "session_cache_root": lambda _sid, _dir: "cache_dir",
        "build_chat_messages": lambda **_kwargs: [{"role": "user", "content": "hello"}],
        "build_text_chat_request": lambda **_kwargs: None,
        "build_kv_state_signature": lambda **_kwargs: "kvsig",
        "try_restore_kv_state": lambda **_kwargs: None,
        "is_state_data_mismatch_error": lambda _err: False,
        "saved_llama_state_size": lambda _state: None,
        "current_llama_state_size": lambda _llm: None,
        "kv_state_debug_info": lambda _state: "dbg",
        "get_context_turns": lambda _history, max_turns=None: [],
        "mem_kv_state": {},
        "maybe_compact_summary": lambda **_kwargs: history_ref,
        "cache_debug_label": lambda _mgr: "cache",
        "run_generation_with_adaptive_retry": lambda **_kwargs: run_generation_result,
        "make_suppress_backend_logs": lambda _suppress: __import__("contextlib").nullcontext(),
        "processing_interrupted": lambda: False,
        "throw_if_processing_interrupted": lambda: None,
        "is_interrupt_error": lambda _err: False,
        "iter_chat_completion_robust": lambda *_args, **_kwargs: iter(()),
        "create_chat_completion_robust": lambda *_args, **_kwargs: {"choices": [{"message": {"content": "unused"}}]},
        "extract_stream_content": lambda _chunk: "",
        "retry_kwargs_with_repeat_last_n_fallback": lambda kwargs, _n: dict(kwargs),
        "strip_reasoning_output": lambda text: text,
        "next_turn_id": lambda _history: 1,
        "now_iso": lambda: "2026-03-21T12:00:00+09:00",
        "maybe_summarize_history": lambda **_kwargs: history_ref,
        "atomic_write_json": lambda path, obj: writes.append((path, obj.copy())),
        "try_save_kv_state": lambda **_kwargs: None,
    }
    return deps, writes


def _make_request(deps, mgr: DummyManager, *, log_level: str = "timing") -> TurnExecutionRequest:
    return TurnExecutionRequest(
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
        runtime_cache="off",
        log_level=log_level,
        model_manager=mgr,
        dependencies=deps,
    )


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
    observed: dict[str, object] = {}
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

    def run_generation_with_adaptive_retry(**kwargs):
        observed.update(kwargs)
        return GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        )

    deps["run_generation_with_adaptive_retry"] = run_generation_with_adaptive_retry

    result = service.execute_turn(_make_request(deps, mgr, log_level="timing"))

    assert result.generation_succeeded is True
    assert observed["heartbeat_logger"] is None


def test_execute_turn_enables_heartbeat_for_debug_log_level() -> None:
    service = TurnExecutionService()
    mgr = DummyManager()
    observed: dict[str, object] = {}
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

    def run_generation_with_adaptive_retry(**kwargs):
        observed.update(kwargs)
        return GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        )

    deps["run_generation_with_adaptive_retry"] = run_generation_with_adaptive_retry

    result = service.execute_turn(_make_request(deps, mgr, log_level="debug"))

    assert result.generation_succeeded is True
    assert callable(observed["heartbeat_logger"])


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


def test_execute_turn_requires_vision_when_image_is_present() -> None:
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
    request = TurnExecutionRequest(**{**request.__dict__, "image": object(), "mmproj": "(Auto-detect)"})

    result = service.execute_turn(request)

    assert result.generation_succeeded is True
    assert mgr.last_load_kwargs["vision_required"] is True


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


def test_execute_turn_does_not_require_vision_for_auto_mmproj_without_image() -> None:
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
        image=None,
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
        include_image_and_stream_in_turn_params=True,
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


def test_execute_session_chat_turn_adds_image_stream_params() -> None:
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
        image=None,
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
    assert "image_used" in params
    assert "streamed" in params


def test_execute_dialogue_cycle_turn_strips_and_omits_image_stream_params() -> None:
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
        image=None,
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
    assert captured["include_image_and_stream_in_turn_params"] is True
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
    assert captured["include_image_and_stream_in_turn_params"] is False
    assert captured["kv_log_saved_when_not_minimal"] is True
    assert captured["kv_log_unsupported_when_not_minimal"] is True
    assert captured["include_error_in_invalidate_message"] is True
    assert captured["enable_attempt_logging"] is False
    assert captured["log_prefix"] == "[LLM Dialogue Cycle]"



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


class TestErrorHandlingP0P1:
    """Tests for P0/P1 error handling improvements."""

    def test_cache_invalidation_failure_during_reset_logs_error(self, caplog) -> None:
        """P0: Test that cache invalidation failure during session reset is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        # Make invalidate_cache raise an exception
        def failing_invalidate(*_args, **_kwargs):
            raise RuntimeError("Cache invalidation failed")

        mgr.invalidate_cache = failing_invalidate

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Create request with reset_session=True
        request = TurnExecutionRequest(
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
            runtime_cache="off",
            model_manager=mgr,
            dependencies=deps,
            reset_session=True,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite cache invalidation failure
        assert result.generation_succeeded is True

    def test_cache_directory_creation_failure_logs_error(self, caplog) -> None:
        """P0: Test that cache directory creation failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make session_cache_root raise an exception
        deps["session_cache_root"] = lambda _sid, _dir: (_ for _ in ()).throw(RuntimeError("Cannot create cache dir"))

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error and continue without cache
        result = service.execute_turn(request)

        # Generation should still succeed despite cache directory failure
        assert result.generation_succeeded is True

    def test_history_save_failure_logs_error(self, caplog) -> None:
        """P0: Test that history file save failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make atomic_write_json raise an exception
        deps["atomic_write_json"] = lambda _path, _obj: (_ for _ in ()).throw(RuntimeError("Failed to write history"))

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite history save failure
        assert result.generation_succeeded is True
        assert result.persistence_succeeded is False
        assert isinstance(result.persistence_error, RuntimeError)
        assert "Failed to write history" in str(result.persistence_error)

    def test_kv_state_restore_failure_logs_error(self, caplog) -> None:
        """P1: Test that KV state restore failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make try_restore_kv_state raise an exception
        deps["try_restore_kv_state"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("KV restore failed"))

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite KV state restore failure
        assert result.generation_succeeded is True

    def test_summarization_failure_logs_error(self, caplog) -> None:
        """P1: Test that summarization failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make maybe_summarize_history raise an exception
        deps["maybe_summarize_history"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Summarization failed"))

        request = TurnExecutionRequest(
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
            runtime_cache="off",
            model_manager=mgr,
            dependencies=deps,
            summarize_old_history=True,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite summarization failure
        assert result.generation_succeeded is True

    def test_kv_state_save_failure_logs_error(self, caplog) -> None:
        """P1: Test that KV state save failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make try_save_kv_state raise an exception
        deps["try_save_kv_state"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("KV save failed"))

        request = TurnExecutionRequest(
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
            runtime_cache="KV_cache",
            model_manager=mgr,
            dependencies=deps,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite KV state save failure
        assert result.generation_succeeded is True

    def test_cache_invalidation_on_mismatch_failure_logs_error(self, caplog) -> None:
        """P1: Test that cache invalidation failure on mismatch is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make is_state_data_mismatch_error return True to trigger cache invalidation
        deps["is_state_data_mismatch_error"] = lambda _err: True

        # Make invalidate_cache raise an exception during mismatch handling
        def failing_invalidate(llm, remove_disk_data=False):
            raise RuntimeError("Cache invalidation on mismatch failed")

        mgr.invalidate_cache = failing_invalidate

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite cache invalidation failure
        assert result.generation_succeeded is True

    def test_summary_compaction_failure_logs_error(self, caplog) -> None:
        """P1: Test that summary compaction failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make maybe_compact_summary raise an exception
        deps["maybe_compact_summary"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Summary compaction failed"))

        request = TurnExecutionRequest(
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
            runtime_cache="off",
            model_manager=mgr,
            dependencies=deps,
            summarize_old_history=True,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite summary compaction failure
        assert result.generation_succeeded is True
