# core/defaults.py: centralized behavior-preserving defaults for node config and UI wiring.

from __future__ import annotations

from typing import Any, Dict

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
LOG_LEVEL_OPTIONS = ("minimal", "timing", "debug")
PERSISTENT_CACHE_OPTIONS = ("LlamaDiskCache", "off")
RUNTIME_CACHE_OPTIONS = ("KV_cache", "LlamaRAMCache", "LlamaTrieCache", "off")

SIMPLE_DEFAULTS: Dict[str, Any] = {
    "schema_version": 1,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "n_gpu_layers": 0,
    "n_ctx": 4096,
    "max_turns": 6,
    "summarize_old_history": True,
    "summary_chunk_turns": 6,
    "max_tokens_summary": 128,
    "summary_max_chars": 1500,
    "dynamic_max_tokens": True,
    "min_generation_tokens": 96,
    "safety_margin_tokens": 64,
    "persistent_cache": "off",
    "runtime_cache": "LlamaTrieCache",
    "log_level": "timing",
    "suppress_backend_logs": True,
    "repeat_penalty": 1.12,
    "repeat_last_n": 256,
    "rewrite_continue": True,
    "reset_session": False,
    "stream_to_console": False,
    "system_prompt_A": "",
    "system_prompt_B": "",
}

FULL_UI_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "session_chat": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": 0,
        "n_ctx": 4096,
        "persistent_cache": "off",
        "runtime_cache": "LlamaTrieCache",
        "log_level": "timing",
        "suppress_backend_logs": True,
        "repeat_penalty": 1.12,
        "repeat_last_n": 256,
        "rewrite_continue": True,
        "max_turns": 12,
        "summarize_old_history": True,
        "summary_chunk_turns": 3,
        "max_tokens_summary": 128,
        "summary_max_chars": 1500,
        "dynamic_max_tokens": True,
        "min_generation_tokens": 96,
        "safety_margin_tokens": 64,
        "reset_session": False,
        "stream_to_console": False,
    },
    "dialogue_cycle": {
        "cycles": 1,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_prompt_A": "",
        "system_prompt_B": "",
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "n_gpu_layers": 0,
        "n_ctx": 4096,
        "max_turns": 12,
        "summarize_old_history": True,
        "summary_chunk_turns": 3,
        "max_tokens_summary": 128,
        "summary_max_chars": 1500,
        "dynamic_max_tokens": True,
        "min_generation_tokens": 96,
        "safety_margin_tokens": 64,
        "persistent_cache": "off",
        "runtime_cache": "LlamaTrieCache",
        "repeat_penalty": 1.12,
        "repeat_last_n": 256,
        "rewrite_continue": True,
        "log_level": "timing",
        "suppress_backend_logs": True,
        "reset_session": False,
        "stream_to_console": False,
    },
}

SUMMARY_HELPER_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.2,
    "suppress_logs": False,
    "min_summary_max_chars": 200,
}
