from __future__ import annotations

from core.turn_types import GenerationRunResult
from services.turn_execution_service import TurnExecutionRequest, TurnExecutionService


class DummyManager:
    def __init__(self) -> None:
        self.model = object()
        self.cache_dir_override = None
        self.loaded = False
        self.cache_configured = False

    def load_model(self, **_kwargs):
        self.loaded = True
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


def _make_request(deps, mgr: DummyManager) -> TurnExecutionRequest:
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
