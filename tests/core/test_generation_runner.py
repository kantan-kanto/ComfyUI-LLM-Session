from __future__ import annotations

import contextlib

from core.generation_runner import run_generation_with_adaptive_retry


def _noop_attempt_logger(*_args, **_kwargs):
    return None


def test_generation_success_first_attempt() -> None:
    def create_chat_completion_robust(_llm, _messages, **_kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}

    result = run_generation_with_adaptive_retry(
        llm=object(),
        messages=[{"role": "user", "content": "hello"}],
        text_chat_request=None,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        repeat_last_n=64,
        dynamic_max_tokens=True,
        min_generation_tokens=32,
        safety_margin_tokens=8,
        initial_turns_limit=5,
        stream_to_console=False,
        max_attempts=3,
        is_ctx_error=lambda _e: False,
        is_state_data_mismatch_error=lambda _e: False,
        on_state_cache_mismatch=lambda _e: None,
        on_compact_summary=lambda: None,
        rebuild_messages_for_turns_limit=lambda _t: ([{"role": "user", "content": "hello"}], None),
        attempt_logger=_noop_attempt_logger,
        debug_traceback=False,
        traceback_print_exc=lambda: None,
        suppress_backend_logs_ctx_factory=contextlib.nullcontext,
        iter_chat_completion_robust=lambda *_a, **_k: iter(()),
        create_chat_completion_robust=create_chat_completion_robust,
        extract_stream_content=lambda _chunk: "",
        retry_kwargs_with_repeat_last_n_fallback=lambda kwargs, _n: dict(kwargs),
    )

    assert result.succeeded is True
    assert result.assistant_text == "ok"
    assert result.non_ctx_error is False


def test_generation_retries_on_ctx_error_with_reduced_tokens() -> None:
    call_count = {"n": 0}
    observed_max_tokens = []

    def create_chat_completion_robust(_llm, _messages, **kwargs):
        call_count["n"] += 1
        observed_max_tokens.append(kwargs.get("max_tokens"))
        if call_count["n"] == 1:
            raise RuntimeError("Prompt exceeds n_ctx")
        return {"choices": [{"message": {"content": "recovered"}}]}

    rebuilt_turns = []

    result = run_generation_with_adaptive_retry(
        llm=object(),
        messages=[{"role": "user", "content": "hello"}],
        text_chat_request=None,
        max_tokens=120,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        repeat_last_n=64,
        dynamic_max_tokens=True,
        min_generation_tokens=32,
        safety_margin_tokens=8,
        initial_turns_limit=6,
        stream_to_console=False,
        max_attempts=4,
        is_ctx_error=lambda e: "n_ctx" in str(e),
        is_state_data_mismatch_error=lambda _e: False,
        on_state_cache_mismatch=lambda _e: None,
        on_compact_summary=lambda: None,
        rebuild_messages_for_turns_limit=lambda turns: (rebuilt_turns.append(turns) or [{"role": "user", "content": "hello"}], None),
        attempt_logger=None,
        debug_traceback=False,
        traceback_print_exc=lambda: None,
        suppress_backend_logs_ctx_factory=contextlib.nullcontext,
        iter_chat_completion_robust=lambda *_a, **_k: iter(()),
        create_chat_completion_robust=create_chat_completion_robust,
        extract_stream_content=lambda _chunk: "",
        retry_kwargs_with_repeat_last_n_fallback=lambda kwargs, _n: dict(kwargs),
    )

    assert result.succeeded is True
    assert result.assistant_text == "recovered"
    assert observed_max_tokens[0] == 112
    assert observed_max_tokens[1] == 56
    assert rebuilt_turns == [6]


def test_generation_non_ctx_error_returns_failure_immediately() -> None:
    def create_chat_completion_robust(_llm, _messages, **_kwargs):
        raise RuntimeError("backend failed")

    result = run_generation_with_adaptive_retry(
        llm=object(),
        messages=[{"role": "user", "content": "hello"}],
        text_chat_request=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        repeat_last_n=64,
        dynamic_max_tokens=True,
        min_generation_tokens=32,
        safety_margin_tokens=8,
        initial_turns_limit=6,
        stream_to_console=False,
        max_attempts=4,
        is_ctx_error=lambda _e: False,
        is_state_data_mismatch_error=lambda _e: False,
        on_state_cache_mismatch=lambda _e: None,
        on_compact_summary=lambda: None,
        rebuild_messages_for_turns_limit=lambda _turns: ([{"role": "user", "content": "hello"}], None),
        attempt_logger=None,
        debug_traceback=False,
        traceback_print_exc=lambda: None,
        suppress_backend_logs_ctx_factory=contextlib.nullcontext,
        iter_chat_completion_robust=lambda *_a, **_k: iter(()),
        create_chat_completion_robust=create_chat_completion_robust,
        extract_stream_content=lambda _chunk: "",
        retry_kwargs_with_repeat_last_n_fallback=lambda kwargs, _n: dict(kwargs),
    )

    assert result.succeeded is False
    assert result.non_ctx_error is True
    assert isinstance(result.last_err, RuntimeError)
