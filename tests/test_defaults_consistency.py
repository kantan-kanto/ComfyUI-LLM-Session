from __future__ import annotations

from core.defaults import (
    FULL_UI_DEFAULTS,
    LOG_LEVEL_OPTIONS,
    PERSISTENT_CACHE_OPTIONS,
    RUNTIME_CACHE_OPTIONS,
)


def test_full_ui_shared_defaults_consistency():
    session = FULL_UI_DEFAULTS["session_chat"]
    cycle = FULL_UI_DEFAULTS["dialogue_cycle"]

    shared_keys = [
        "system_prompt",
        "max_tokens",
        "temperature",
        "top_p",
        "n_gpu_layers",
        "n_ctx",
        "max_turns",
        "summarize_old_history",
        "summary_chunk_turns",
        "max_tokens_summary",
        "summary_max_chars",
        "dynamic_max_tokens",
        "min_generation_tokens",
        "safety_margin_tokens",
        "persistent_cache",
        "runtime_cache",
        "repeat_penalty",
        "repeat_last_n",
        "rewrite_continue",
        "log_level",
        "suppress_backend_logs",
        "reset_session",
        "stream_to_console",
        "enable_thinking",
    ]

    for key in shared_keys:
        assert session[key] == cycle[key]


def test_full_ui_option_defaults_are_valid_choices():
    session = FULL_UI_DEFAULTS["session_chat"]
    cycle = FULL_UI_DEFAULTS["dialogue_cycle"]

    for target in (session, cycle):
        assert target["persistent_cache"] in PERSISTENT_CACHE_OPTIONS
        assert target["runtime_cache"] in RUNTIME_CACHE_OPTIONS
        assert target["log_level"] in LOG_LEVEL_OPTIONS
