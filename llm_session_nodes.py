# ComfyUI-LLM-Session
# Copyright (C) 2026 kantan-kanto (https://github.com/kantan-kanto)
#
# Local LLM session nodes for ComfyUI (GGUF / llama.cpp via llama-cpp-python).
# - LLM Session Chat: persistent multi-turn chat with file-based history and optional summarization
# - LLM Dialogue Cycle: two-model turn-based dialogue runner without graph cycles
from __future__ import annotations
import os
import io
import sys
import contextlib
import base64
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
import shutil
import folder_paths
import hashlib
import traceback
from importlib import import_module


def _import_layer_module(module_name: str):
    if __package__:
        try:
            return import_module(f".{module_name}", package=__package__)
        except ImportError:
            pass
    return import_module(module_name)


_core_defaults = _import_layer_module("core.defaults")
DEFAULT_SYSTEM_PROMPT = _core_defaults.DEFAULT_SYSTEM_PROMPT
FULL_UI_DEFAULTS = _core_defaults.FULL_UI_DEFAULTS
LOG_LEVEL_OPTIONS = _core_defaults.LOG_LEVEL_OPTIONS
PERSISTENT_CACHE_OPTIONS = _core_defaults.PERSISTENT_CACHE_OPTIONS
RUNTIME_CACHE_OPTIONS = _core_defaults.RUNTIME_CACHE_OPTIONS
SIMPLE_DEFAULTS = _core_defaults.SIMPLE_DEFAULTS
SIMPLE_WRAPPER_DEFAULTS = _core_defaults.SIMPLE_WRAPPER_DEFAULTS
SUMMARY_HELPER_DEFAULTS = _core_defaults.SUMMARY_HELPER_DEFAULTS

_core_continue_rewrite = _import_layer_module("core.continue_rewrite")
rewrite_continue_prompt = _core_continue_rewrite.rewrite_continue_prompt

_core_generation_runner = _import_layer_module("core.generation_runner")
run_generation_with_adaptive_retry = _core_generation_runner.run_generation_with_adaptive_retry
run_with_typeerror_fallback = _core_generation_runner.run_with_typeerror_fallback

_core_kv_state = _import_layer_module("core.kv_state")
build_kv_state_signature = _core_kv_state.build_kv_state_signature
try_restore_kv_state = _core_kv_state.try_restore_kv_state
try_save_kv_state = _core_kv_state.try_save_kv_state

_core_runtime_container = _import_layer_module("core.runtime_container")
RuntimeContainer = _core_runtime_container.RuntimeContainer

history_store = _import_layer_module("infra.history_store")

_services_chat_turn = _import_layer_module("services.chat_turn_service")
ChatTurnService = _services_chat_turn.ChatTurnService
DialogueCycleDependencies = _services_chat_turn.DialogueCycleDependencies
DialogueCycleNodeExecutionDependencies = _services_chat_turn.DialogueCycleNodeExecutionDependencies
DialogueCycleNodeExecutionRequest = _services_chat_turn.DialogueCycleNodeExecutionRequest
DialogueCycleNodeExecutionService = _services_chat_turn.DialogueCycleNodeExecutionService
DialogueCycleRequest = _services_chat_turn.DialogueCycleRequest

_services_turn_execution = _import_layer_module("services.turn_execution_service")
SessionChatNodeExecutionDependencies = _services_turn_execution.SessionChatNodeExecutionDependencies
SessionChatNodeExecutionRequest = _services_turn_execution.SessionChatNodeExecutionRequest
SessionChatNodeExecutionService = _services_turn_execution.SessionChatNodeExecutionService
TurnExecutionDependencies = _services_turn_execution.TurnExecutionDependencies
TurnExecutionResult = _services_turn_execution.TurnExecutionResult
TurnExecutionService = _services_turn_execution.TurnExecutionService

_core_logging_utils = _import_layer_module("core.logging_utils")
log_error_safely = _core_logging_utils.log_error_safely
get_module_logger = _core_logging_utils.get_module_logger
LOG_LEVEL_MINIMAL = _core_logging_utils.LOG_LEVEL_MINIMAL

# ============================================================================
# Module Layout (for future file split)
# ============================================================================
# - Simple defaults (config-driven)
# - Chat handler configuration (llama-cpp)
# - Text/chat prompt builders
# - Model discovery and path resolution
# - Image + language utilities
# - History I/O and session storage
# - Message building + generation wrappers
# - Summarization helpers
# - Runtime container + cache + model manager
# - UI definition helpers
# - ComfyUI node implementations
# - Simple wrappers
# - Node registration + cleanup

# ============================================================================
# Simple Defaults (config-driven)
# ============================================================================
# LLM Session Chat (Simple) loads optional defaults from:
#   <this_repo>/config/simple_defaults.json
#
# - If the file is missing or invalid, built-in safe defaults are used.
# - Unknown keys are ignored.
#
# This keeps the Simple node easy to use while still allowing customization.

_DEFAULT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
_SIMPLE_DEFAULTS_BUILTIN: Dict[str, Any] = dict(SIMPLE_DEFAULTS)
_FULL_UI_SESSION_CHAT_DEFAULTS = FULL_UI_DEFAULTS["session_chat"]
_FULL_UI_DIALOGUE_CYCLE_DEFAULTS = FULL_UI_DEFAULTS["dialogue_cycle"]
_SUMMARY_HELPER_DEFAULTS: Dict[str, Any] = dict(SUMMARY_HELPER_DEFAULTS)
_SIMPLE_WRAPPER_DEFAULTS: Dict[str, Any] = dict(SIMPLE_WRAPPER_DEFAULTS)

_SIMPLE_ALLOWED_KEYS = set(_SIMPLE_DEFAULTS_BUILTIN.keys()) - {"schema_version"}
_ADVANCED_GENERATION_ALLOWED_KEYS = {"seed"}
_ADVANCED_SUMMARY_GENERATION_ALLOWED_KEYS = {"seed"}

def _simple_config_log(message: str, log_level: str) -> None:
    try:
        if str(log_level).strip().lower() != "minimal":
            print(f"[LLM Session Simple] {message}")
    except Exception:
        pass

def _normalize_config_path(config_path: Optional[str]) -> str:
    raw = (config_path or "").strip()
    if not raw:
        return ""
    # Strip wrapping quotes if the UI includes them.
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1].strip()
    try:
        raw = os.path.expandvars(raw)
        raw = os.path.expanduser(raw)
    except Exception:
        pass
    try:
        return str(Path(raw))
    except Exception:
        return raw

def _normalize_tensor_split(value: Any) -> Optional[List[float]]:
    """Return a llama.cpp tensor_split list, or None when unset/invalid."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None

    split: List[float] = []
    for item in value:
        try:
            f = float(item)
        except Exception:
            return None
        if f < 0:
            return None
        split.append(f)

    if not split or not any(x > 0 for x in split):
        return None
    return split


def _advanced_seed_kwargs(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict) or value.get("seed") is None:
        return {}
    try:
        return {"seed": int(value.get("seed"))}
    except Exception:
        return {}


def _warn_unsupported_advanced_keys(
    *,
    section_name: str,
    value: Any,
    allowed_keys: set[str],
    log_level: str,
) -> None:
    if not isinstance(value, dict):
        return
    unsupported = sorted(str(k) for k in value.keys() if k not in allowed_keys)
    if not unsupported:
        return
    _simple_config_log(
        f"Warning: Ignoring unsupported {section_name} keys: {', '.join(unsupported)}",
        log_level,
    )

def _simple_config_path() -> str:
    try:
        base = Path(__file__).parent
        return str(base / "config" / "simple_defaults.json")
    except Exception:
        return ""

def _load_simple_defaults(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load Simple defaults from JSON, falling back to built-in safe defaults."""
    cfg_path = _normalize_config_path(config_path) or _simple_config_path()
    defaults = dict(_SIMPLE_DEFAULTS_BUILTIN)
    log_level_hint = defaults.get("log_level", _SIMPLE_DEFAULTS_BUILTIN["log_level"])

    if not cfg_path:
        _simple_config_log("No config path provided; using built-in defaults.", log_level_hint)
        return defaults
    if not os.path.exists(cfg_path):
        _simple_config_log(f"Config not found: {cfg_path} (using built-in defaults)", log_level_hint)
        return defaults

    try:
        # Use utf-8-sig to tolerate BOM without failing JSON parsing.
        with open(cfg_path, "r", encoding="utf-8-sig") as f:
            config_obj = json.load(f)
        if not isinstance(config_obj, dict):
            _simple_config_log(f"Config JSON is not an object: {cfg_path}", log_level_hint)
            return defaults
        if int(config_obj.get("schema_version", 1)) != 1:
            # Future-proof: unknown schema -> ignore
            _simple_config_log(f"Config schema_version unsupported: {config_obj.get('schema_version')} in {cfg_path}", log_level_hint)
            return defaults

        if "log_level" in config_obj:
            log_level_hint = str(config_obj.get("log_level") or log_level_hint)

        for k, v in config_obj.items():
            if k not in _SIMPLE_ALLOWED_KEYS:
                continue
            defaults[k] = v
    except Exception:
        _simple_config_log(f"Failed to load config: {cfg_path}", log_level_hint)
        return defaults

    chat_handler_overrides: Dict[str, Dict[str, Any]] = {}
    text_chat_builder_overrides: Dict[str, Dict[str, Any]] = {}
    for chat_format in sorted(
        set(_ENABLE_THINKING_CHAT_HANDLER_FORMATS)
        | set(_ENABLE_THINKING_TEXT_CHAT_BUILDER_FORMATS)
    ):
        raw_overrides = config_obj.get(chat_format)
        if not isinstance(raw_overrides, dict) or "enable_thinking" not in raw_overrides:
            continue
        if chat_format in _ENABLE_THINKING_CHAT_HANDLER_FORMATS:
            chat_handler_overrides.setdefault(chat_format, {})["enable_thinking"] = raw_overrides.get(
                "enable_thinking"
            )
        if chat_format in _ENABLE_THINKING_TEXT_CHAT_BUILDER_FORMATS:
            text_chat_builder_overrides.setdefault(chat_format, {})["enable_thinking"] = raw_overrides.get(
                "enable_thinking"
            )

    # Best-effort type coercion + clamping (never raise)
    def _as_int(x, d):
        try:
            return int(x)
        except Exception:
            return d

    def _as_float(x, d):
        try:
            return float(x)
        except Exception:
            return d

    def _as_bool(x, d):
        try:
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return bool(x)
            s = str(x).strip().lower()
            if s in ("1", "true", "yes", "y", "on"):
                return True
            if s in ("0", "false", "no", "n", "off"):
                return False
        except Exception:
            pass
        return d

    # Numeric clamps (match UI ranges loosely)
    defaults["max_tokens"] = max(1, _as_int(defaults.get("max_tokens"), _SIMPLE_DEFAULTS_BUILTIN["max_tokens"]))
    defaults["n_ctx"] = max(512, _as_int(defaults.get("n_ctx"), _SIMPLE_DEFAULTS_BUILTIN["n_ctx"]))
    defaults["n_gpu_layers"] = _as_int(defaults.get("n_gpu_layers"), _SIMPLE_DEFAULTS_BUILTIN["n_gpu_layers"])
    defaults["tensor_split"] = _normalize_tensor_split(defaults.get("tensor_split"))
    defaults["max_turns"] = max(0, _as_int(defaults.get("max_turns"), _SIMPLE_DEFAULTS_BUILTIN["max_turns"]))
    defaults["summary_chunk_turns"] = max(1, _as_int(defaults.get("summary_chunk_turns"), _SIMPLE_DEFAULTS_BUILTIN["summary_chunk_turns"]))
    defaults["max_tokens_summary"] = max(16, _as_int(defaults.get("max_tokens_summary"), _SIMPLE_DEFAULTS_BUILTIN["max_tokens_summary"]))
    defaults["summary_max_chars"] = max(200, _as_int(defaults.get("summary_max_chars"), _SIMPLE_DEFAULTS_BUILTIN["summary_max_chars"]))
    defaults["min_generation_tokens"] = max(1, _as_int(defaults.get("min_generation_tokens"), _SIMPLE_DEFAULTS_BUILTIN["min_generation_tokens"]))
    defaults["safety_margin_tokens"] = max(0, _as_int(defaults.get("safety_margin_tokens"), _SIMPLE_DEFAULTS_BUILTIN["safety_margin_tokens"]))
    defaults["repeat_last_n"] = max(0, _as_int(defaults.get("repeat_last_n"), _SIMPLE_DEFAULTS_BUILTIN["repeat_last_n"]))

    defaults["temperature"] = min(2.0, max(0.0, _as_float(defaults.get("temperature"), _SIMPLE_DEFAULTS_BUILTIN["temperature"])))
    defaults["top_p"] = min(1.0, max(0.05, _as_float(defaults.get("top_p"), _SIMPLE_DEFAULTS_BUILTIN["top_p"])))
    defaults["repeat_penalty"] = min(2.0, max(1.0, _as_float(defaults.get("repeat_penalty"), _SIMPLE_DEFAULTS_BUILTIN["repeat_penalty"])))

    # Enums / strings
    defaults["persistent_cache"] = str(defaults.get("persistent_cache", _SIMPLE_DEFAULTS_BUILTIN["persistent_cache"]))
    if defaults["persistent_cache"] not in _PERSISTENT_CACHE_OPTIONS:
        defaults["persistent_cache"] = _SIMPLE_DEFAULTS_BUILTIN["persistent_cache"]

    defaults["runtime_cache"] = str(defaults.get("runtime_cache", _SIMPLE_DEFAULTS_BUILTIN["runtime_cache"]))
    if defaults["runtime_cache"] not in _RUNTIME_CACHE_OPTIONS:
        defaults["runtime_cache"] = _SIMPLE_DEFAULTS_BUILTIN["runtime_cache"]

    defaults["log_level"] = str(defaults.get("log_level", _SIMPLE_DEFAULTS_BUILTIN["log_level"])).lower()
    if defaults["log_level"] not in _LOG_LEVEL_OPTIONS:
        defaults["log_level"] = _SIMPLE_DEFAULTS_BUILTIN["log_level"]

    _warn_unsupported_advanced_keys(
        section_name="advanced_generation_kwargs",
        value=config_obj.get("advanced_generation_kwargs"),
        allowed_keys=_ADVANCED_GENERATION_ALLOWED_KEYS,
        log_level=defaults["log_level"],
    )
    _warn_unsupported_advanced_keys(
        section_name="advanced_summary_generation_kwargs",
        value=config_obj.get("advanced_summary_generation_kwargs"),
        allowed_keys=_ADVANCED_SUMMARY_GENERATION_ALLOWED_KEYS,
        log_level=defaults["log_level"],
    )

    # Booleans
    defaults["summarize_old_history"] = _as_bool(defaults.get("summarize_old_history"), _SIMPLE_DEFAULTS_BUILTIN["summarize_old_history"])
    defaults["dynamic_max_tokens"] = _as_bool(defaults.get("dynamic_max_tokens"), _SIMPLE_DEFAULTS_BUILTIN["dynamic_max_tokens"])
    defaults["suppress_backend_logs"] = _as_bool(defaults.get("suppress_backend_logs"), _SIMPLE_DEFAULTS_BUILTIN["suppress_backend_logs"])
    defaults["rewrite_continue"] = _as_bool(defaults.get("rewrite_continue"), _SIMPLE_DEFAULTS_BUILTIN["rewrite_continue"])
    defaults["reset_session"] = _as_bool(defaults.get("reset_session"), _SIMPLE_DEFAULTS_BUILTIN["reset_session"])
    defaults["stream_to_console"] = _as_bool(defaults.get("stream_to_console"), _SIMPLE_DEFAULTS_BUILTIN["stream_to_console"])

    for chat_format, overrides in list(chat_handler_overrides.items()):
        if "enable_thinking" in overrides:
            overrides["enable_thinking"] = _as_bool(
                overrides.get("enable_thinking"),
                CHAT_HANDLER_KWARGS_MAP.get(chat_format, {}).get("enable_thinking", True),
            )
    for chat_format, overrides in list(text_chat_builder_overrides.items()):
        if "enable_thinking" in overrides:
            overrides["enable_thinking"] = _as_bool(
                overrides.get("enable_thinking"),
                TEXT_CHAT_BUILDER_CONFIG_MAP.get(chat_format, {}).get("enable_thinking", False),
            )

    # System prompt(s)
    sp = defaults.get("system_prompt")
    defaults["system_prompt"] = str(sp) if sp is not None else _SIMPLE_DEFAULTS_BUILTIN["system_prompt"]
    sp_a = defaults.get("system_prompt_A")
    defaults["system_prompt_A"] = str(sp_a) if sp_a is not None else _SIMPLE_DEFAULTS_BUILTIN["system_prompt_A"]
    sp_b = defaults.get("system_prompt_B")
    defaults["system_prompt_B"] = str(sp_b) if sp_b is not None else _SIMPLE_DEFAULTS_BUILTIN["system_prompt_B"]
    defaults["chat_handler_overrides"] = chat_handler_overrides
    defaults["text_chat_builder_overrides"] = text_chat_builder_overrides
    defaults["advanced_generation_kwargs"] = _advanced_seed_kwargs(
        config_obj.get("advanced_generation_kwargs")
    )
    defaults["advanced_summary_generation_kwargs"] = _advanced_seed_kwargs(
        config_obj.get("advanced_summary_generation_kwargs")
    )

    return defaults


def _load_simple_defaults_bundle(
    config_path: Optional[str] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]], Optional[Dict[str, Dict[str, Any]]]]:
    defaults = _load_simple_defaults(config_path=config_path)
    return (
        defaults,
        defaults.get("chat_handler_overrides"),
        defaults.get("text_chat_builder_overrides"),
    )


def _resolve_simple_system_prompts(
    defaults: Dict[str, Any],
    system: str,
    systemA: str,
    systemB: str,
) -> tuple[str, str, str]:
    system_prompt = (system or "").strip() or str(defaults["system_prompt"] or _DEFAULT_SYSTEM_PROMPT)
    system_prompt_A = (systemA or "").strip() or str(defaults["system_prompt_A"] or "")
    system_prompt_B = (systemB or "").strip() or str(defaults["system_prompt_B"] or "")
    return system_prompt, system_prompt_A, system_prompt_B


def _build_dialogue_cycle_simple_chat_kwargs(
    *,
    defaults: Dict[str, Any],
    force_text_only: bool,
    history_dir: str,
    reset_session: bool,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    mmproj_a = _MMPROJ_NOT_REQUIRED if force_text_only else _MMPROJ_AUTO
    mmproj_b = _MMPROJ_NOT_REQUIRED if force_text_only else _MMPROJ_AUTO
    return {
        "mmprojA": mmproj_a,
        "mmprojB": mmproj_b,
        "max_tokens": int(defaults["max_tokens"]),
        "temperature": float(defaults["temperature"]),
        "top_p": float(defaults["top_p"]),
        "n_gpu_layers": int(defaults["n_gpu_layers"]),
        "tensor_split": defaults.get("tensor_split"),
        "n_ctx": int(defaults["n_ctx"]),
        "max_turns": int(defaults["max_turns"]),
        "summarize_old_history": bool(defaults["summarize_old_history"]),
        "summary_chunk_turns": int(defaults["summary_chunk_turns"]),
        "max_tokens_summary": int(defaults["max_tokens_summary"]),
        "summary_max_chars": int(defaults["summary_max_chars"]),
        "dynamic_max_tokens": bool(defaults["dynamic_max_tokens"]),
        "min_generation_tokens": int(defaults["min_generation_tokens"]),
        "safety_margin_tokens": int(defaults["safety_margin_tokens"]),
        "persistent_cache": str(defaults["persistent_cache"]),
        "runtime_cache": str(defaults["runtime_cache"]),
        "repeat_penalty": float(defaults["repeat_penalty"]),
        "repeat_last_n": int(defaults["repeat_last_n"]),
        "rewrite_continue": bool(defaults["rewrite_continue"]),
        "log_level": str(defaults["log_level"]),
        "suppress_backend_logs": bool(defaults["suppress_backend_logs"]),
        "history_dir": history_dir or "",
        "reset_session": bool(reset_session),
        "stream_to_console": bool(defaults["stream_to_console"]),
        "chat_handler_overrides": chat_handler_overrides,
        "text_chat_builder_overrides": text_chat_builder_overrides,
        "advanced_generation_kwargs": dict(defaults.get("advanced_generation_kwargs") or {}),
        "advanced_summary_generation_kwargs": dict(defaults.get("advanced_summary_generation_kwargs") or {}),
    }


def _build_session_chat_simple_chat_kwargs(
    *,
    defaults: Dict[str, Any],
    history_dir: str,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    return {
        "system_prompt": defaults["system_prompt"],
        "max_tokens": int(defaults["max_tokens"]),
        "temperature": float(defaults["temperature"]),
        "top_p": float(defaults["top_p"]),
        "n_gpu_layers": int(defaults["n_gpu_layers"]),
        "tensor_split": defaults.get("tensor_split"),
        "n_ctx": int(defaults["n_ctx"]),
        "max_turns": int(defaults["max_turns"]),
        "summarize_old_history": bool(defaults["summarize_old_history"]),
        "summary_chunk_turns": int(defaults["summary_chunk_turns"]),
        "max_tokens_summary": int(defaults["max_tokens_summary"]),
        "summary_max_chars": int(defaults["summary_max_chars"]),
        "dynamic_max_tokens": bool(defaults["dynamic_max_tokens"]),
        "min_generation_tokens": int(defaults["min_generation_tokens"]),
        "safety_margin_tokens": int(defaults["safety_margin_tokens"]),
        "persistent_cache": str(defaults["persistent_cache"]),
        "repeat_penalty": float(defaults["repeat_penalty"]),
        "repeat_last_n": int(defaults["repeat_last_n"]),
        "rewrite_continue": bool(defaults["rewrite_continue"]),
        "runtime_cache": str(defaults["runtime_cache"]),
        "log_level": str(defaults["log_level"]),
        "suppress_backend_logs": bool(defaults["suppress_backend_logs"]),
        "history_dir": history_dir or "",
        "reset_session": bool(defaults["reset_session"]),
        "stream_to_console": bool(defaults["stream_to_console"]),
        "chat_handler_overrides": chat_handler_overrides,
        "text_chat_builder_overrides": text_chat_builder_overrides,
        "advanced_generation_kwargs": dict(defaults.get("advanced_generation_kwargs") or {}),
        "advanced_summary_generation_kwargs": dict(defaults.get("advanced_summary_generation_kwargs") or {}),
    }


# llama-cpp-python imports
from collections import defaultdict
from typing import Any, Callable

# ============================================================================
# Chat Handler Configuration (llama-cpp)
# ============================================================================

# chat_format -> ChatHandler class name
chat_handler_map = {
    "llava-1-5": "Llava15ChatHandler",
    "llava-1-6": "Llava16ChatHandler",
    "moondream2": "MoondreamChatHandler",
    "nanollava": "NanollavaChatHandler",
    "llama-3-vision-alpha": "Llama3VisionAlphaChatHandler",
    "minicpm-v-2.6": "MiniCPMv26ChatHandler",
    "minicpm-v-4.0": "MiniCPMv26ChatHandler",
    "minicpm-v-4.5": "MiniCPMv45ChatHandler",
    "minicpm-v-4.6": "MiniCPMV46ChatHandler",
    "gemma3": "Gemma3ChatHandler",
    "gemma4": "Gemma4ChatHandler",
    "glm4.1v": "GLM41VChatHandler",
    "glm4.6v": "GLM46VChatHandler",
    "granite-docling": "GraniteDoclingChatHandler",
    "lfm2-vl": "LFM2VLChatHandler",
    "lfm2.5-vl": "LFM25VLChatHandler",
    "paddleocr": "PaddleOCRChatHandler",
    "qwen2.5-vl": "Qwen25VLChatHandler",
    "qwen3-vl": "Qwen3VLChatHandler",
    "qwen3.5": "Qwen35ChatHandler",
    "step3-vl": "Step3VLChatHandler",
}
DECLARED_CHAT_HANDLER_MAP = dict(chat_handler_map)
JAMEPENG_LLAMA_CPP_URL = "https://github.com/JamePeng/llama-cpp-python"

# chat_format ごとの追加 kwargs
CHAT_HANDLER_KWARGS_MAP = {
    "llava-1-5": {},
    "llava-1-6": {},
    "moondream2": {},
    "nanollava": {},
    "llama-3-vision-alpha": {},
    "minicpm-v-2.6": {},
    "minicpm-v-4.0": {},
    "minicpm-v-4.5": {},
    "minicpm-v-4.6": {"enable_thinking": False},
    "gemma3": {},
    "gemma4": {"enable_thinking": False},
    "glm4.1v": {},
    "glm4.6v": {},
    "granite-docling": {},
    "lfm2-vl": {},
    "lfm2.5-vl": {},
    "paddleocr": {},
    "qwen2.5-vl": {"image_min_tokens": 1024},
    "qwen3-vl": {"image_min_tokens": 1024},
    "qwen3.5": {"enable_thinking": False, "image_min_tokens": 1024},
    "step3-vl": {},
}

TEXT_CHAT_BUILDER_CONFIG_MAP = {
    "gemma4": {"enable_thinking": False},
    "minicpm-v-4.6": {"enable_thinking": False},
    "qwen3.5": {"enable_thinking": False},
}

SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP = {
    "gemma4": {"enable_thinking": False},
    "minicpm-v-4.6": {"enable_thinking": False},
    "qwen3.5": {"enable_thinking": False},
}

def _chat_formats_with_config_key(
    config_map: Dict[str, Dict[str, Any]],
    key: str,
) -> tuple[str, ...]:
    return tuple(sorted({
        chat_format
        for chat_format, config in config_map.items()
        if isinstance(config, dict) and key in config
    }))


_ENABLE_THINKING_CHAT_HANDLER_FORMATS = _chat_formats_with_config_key(
    CHAT_HANDLER_KWARGS_MAP,
    "enable_thinking",
)
_ENABLE_THINKING_TEXT_CHAT_BUILDER_FORMATS = _chat_formats_with_config_key(
    TEXT_CHAT_BUILDER_CONFIG_MAP,
    "enable_thinking",
)


def _merge_chat_format_bool_ui_default_preserving_explicit_overrides(
    overrides: Optional[Dict[str, Dict[str, Any]]],
    chat_formats: tuple[str, ...],
    key: str,
    value: bool,
) -> Dict[str, Dict[str, Any]]:
    """Merge UI defaults without overwriting explicit config/model overrides."""
    merged: Dict[str, Dict[str, Any]] = {}
    if isinstance(overrides, dict):
        for chat_format, values in overrides.items():
            if isinstance(values, dict):
                merged[chat_format] = dict(values)
    for chat_format in chat_formats:
        merged.setdefault(chat_format, {}).setdefault(key, bool(value))
    return merged


def _merge_enable_thinking_chat_handler_overrides(
    overrides: Optional[Dict[str, Dict[str, Any]]],
    enable_thinking: bool,
) -> Dict[str, Dict[str, Any]]:
    return _merge_chat_format_bool_ui_default_preserving_explicit_overrides(
        overrides,
        _ENABLE_THINKING_CHAT_HANDLER_FORMATS,
        "enable_thinking",
        enable_thinking,
    )


def _merge_enable_thinking_text_chat_builder_overrides(
    overrides: Optional[Dict[str, Dict[str, Any]]],
    enable_thinking: bool,
) -> Dict[str, Dict[str, Any]]:
    return _merge_chat_format_bool_ui_default_preserving_explicit_overrides(
        overrides,
        _ENABLE_THINKING_TEXT_CHAT_BUILDER_FORMATS,
        "enable_thinking",
        enable_thinking,
    )


normalized_chat_format_map = {
    "llava-1-5": "llava-1-5",
    "llava15": "llava-1-5",
    "llava-v1.5": "llava-1-5",
    "llava-1-6": "llava-1-6",
    "llava16": "llava-1-6",
    "llava-v1.6": "llava-1-6",
    "moondream2": "moondream2",
    "nanollava": "nanollava",
    "llama-3": "llama-3-vision-alpha",
    "llama3": "llama-3-vision-alpha",
    "minicpm-v-2.6": "minicpm-v-2.6",
    "minicpm-v-2_6": "minicpm-v-2.6",
    "minicpmv26": "minicpm-v-2.6",
    "minicpm-v-4.0": "minicpm-v-4.0",
    "minicpm-v-4_0": "minicpm-v-4.0",
    "minicpmv40": "minicpm-v-4.0",
    "minicpm-v-4.5": "minicpm-v-4.5",
    "minicpm-v-4_5": "minicpm-v-4.5", 
    "minicpmv45": "minicpm-v-4.5",
    "minicpm-v-4.6": "minicpm-v-4.6",
    "minicpm-v-4_6": "minicpm-v-4.6",
    "minicpmv46": "minicpm-v-4.6",
    "gemma3": "gemma3",
    "gemma-3": "gemma3",
    "gemma_3": "gemma3",
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "gemma_4": "gemma4",
    "glm4.1v": "glm4.1v",
    "glm4_1v": "glm4.1v",
    "glm41v": "glm4.1v",
    "glm-4.1v": "glm4.1v",
    "glm4.6v": "glm4.6v",
    "glm4_6v": "glm4.6v",
    "glm46v": "glm4.6v",
    "glm-4.6v": "glm4.6v",
    "granitedocling": "granite-docling",
    "granite-docling": "granite-docling",
    "lfm2-vl": "lfm2-vl",
    "lfm2vl": "lfm2-vl",
    "lfm2.5-vl": "lfm2.5-vl",
    "lfm2.5vl": "lfm2.5-vl",
    "lfm2_5-vl": "lfm2.5-vl",
    "lfm2_5vl": "lfm2.5-vl",
    "paddleocr": "paddleocr",
    "qwen2.5-vl": "qwen2.5-vl",
    "qwen2_5-vl": "qwen2.5-vl",
    "qwen25vl": "qwen2.5-vl",
    "qwen3-vl": "qwen3-vl",
    "qwen3vl": "qwen3-vl",
    "qwen3.5": "qwen3.5",
    "qwen3_5": "qwen3.5",
    "qwen35": "qwen3.5",
    "qwen3.6": "qwen3.5",
    "qwen3_6": "qwen3.5",
    "qwen36": "qwen3.5",
    "step3-vl": "step3-vl",
    "step3vl": "step3-vl",
}

_LLAMA_CPP_IMPORT_ERROR = None
chat_handler_factory_map: dict[str, Callable[[str], Any]] = {}


def _group_formats_by_handler(handler_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for chat_format, handler_name in handler_map.items():
        grouped[handler_name].append(chat_format)
    return dict(grouped)


def _make_chat_handler_factory(handler_cls: type, extra_kwargs: dict[str, Any]) -> Callable[[str], Any]:
    def factory(mmproj_path: str) -> Any:
        return handler_cls(clip_model_path=mmproj_path, **extra_kwargs)

    return factory


def _get_chat_handler_kwargs(
    chat_format: str,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> dict[str, Any]:
    extra_kwargs = dict(CHAT_HANDLER_KWARGS_MAP.get(chat_format, {}))
    if isinstance(chat_handler_overrides, dict):
        override_values = chat_handler_overrides.get(chat_format)
        if isinstance(override_values, dict):
            extra_kwargs.update(override_values)
    return extra_kwargs


def _get_text_chat_builder_config(
    chat_format: str,
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> dict[str, Any]:
    config = dict(TEXT_CHAT_BUILDER_CONFIG_MAP.get(chat_format, {}))
    if isinstance(text_chat_builder_overrides, dict):
        override_values = text_chat_builder_overrides.get(chat_format)
        if isinstance(override_values, dict):
            config.update(override_values)
    return config


def _detect_model_family(model_path: str) -> Optional[str]:
    model_name_lower = os.path.basename(model_path).lower()
    for key, family in normalized_chat_format_map.items():
        if key in model_name_lower:
            return family
    return None


def _detect_gemma4_variant(model_path: str) -> Optional[str]:
    model_name_lower = os.path.basename(model_path).lower()
    compact_name = re.sub(r"[^a-z0-9]+", "", model_name_lower)
    if "e2b" in compact_name:
        return "e2b"
    if "e4b" in compact_name:
        return "e4b"
    if "26ba4b" in compact_name:
        return "26ba4b"
    if "31b" in compact_name:
        return "31b"
    return None


def _warn_if_gemma4_vision_thinking_required(
    model_path: str,
    model_family: str,
    active_chat_handler_kwargs: Dict[str, Any],
) -> None:
    if model_family != "gemma4":
        return
    if active_chat_handler_kwargs.get("enable_thinking") is not False:
        return
    variant = _detect_gemma4_variant(model_path)
    if variant not in ("e2b", "e4b"):
        return
    print(
        "[GGUFModelManager] Warning: Gemma4 E2B/E4B vision models appear to "
        "require enable_thinking=True in the JamePeng Gemma4ChatHandler. "
        "The current override is enable_thinking=False, so the model may "
        "ignore it or behave unexpectedly."
    )


# ============================================================================
# Text/Chat Prompt Builders
# ============================================================================

def _merge_text_chat_builder_overrides(
    model_path: Optional[str],
    base_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    forced_overrides_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Merge model-family-specific builder overrides without touching unrelated models."""
    merged: Dict[str, Dict[str, Any]] = {}
    if isinstance(base_overrides, dict):
        for chat_format, override_values in base_overrides.items():
            if isinstance(override_values, dict):
                merged[str(chat_format)] = dict(override_values)

    if not forced_overrides_map:
        return merged or None

    model_family = _detect_model_family(model_path or "")
    forced_values = forced_overrides_map.get(model_family or "")
    if not isinstance(forced_values, dict):
        return merged or None

    current = dict(merged.get(model_family) or {})
    current.update(forced_values)
    merged[model_family] = current
    return merged


def _load_available_chat_handlers(
    handler_map: dict[str, str],
    handler_kwargs_map: dict[str, dict[str, Any]],
    handler_module: Any,
) -> tuple[dict[str, str], dict[str, Callable[[str], Any]], dict[str, Any]]:
    handler_to_formats = _group_formats_by_handler(handler_map)

    available_handler_map: dict[str, str] = {}
    available_factory_map: dict[str, Callable[[str], Any]] = {}
    available_class_registry: dict[str, Any] = {}

    for handler_name, chat_formats in handler_to_formats.items():
        try:
            handler_cls = getattr(handler_module, handler_name)
        except AttributeError:
            continue
        available_class_registry[handler_name] = handler_cls

        for chat_format in chat_formats:
            available_handler_map[chat_format] = handler_name
            extra_kwargs = handler_kwargs_map.get(chat_format, {})
            available_factory_map[chat_format] = _make_chat_handler_factory(
                handler_cls,
                extra_kwargs,
            )

    return available_handler_map, available_factory_map, available_class_registry


def _empty_chat_handler_registry() -> dict[str, Any]:
    return {}


def _llama_cpp_version_report() -> tuple[str, str]:
    pkg_ver = "unknown"
    backend_ver = "unknown"
    try:
        import llama_cpp  # type: ignore
        pkg_ver = str(getattr(llama_cpp, "__version__", "unknown"))
        try:
            backend_fn = getattr(getattr(llama_cpp, "llama_cpp", None), "llama_cpp_version", None)
            if callable(backend_fn):
                backend_ver = str(backend_fn())
        except Exception:
            pass
    except Exception:
        pass
    return pkg_ver, backend_ver


def _format_multimodal_handler_unavailable_error(
    *,
    model_path: str,
    model_family: Optional[str],
    required_handler: Optional[str],
) -> str:
    pkg_ver, backend_ver = _llama_cpp_version_report()
    model_name = os.path.basename(model_path or "") or "(unknown)"
    family = model_family or "unknown"
    handler = required_handler or "unknown"
    unknown_family_note = (
        "Note: The model filename did not match any known multimodal family aliases.\n"
        if model_family is None
        else ""
    )
    return (
        "Vision is required for this request, but the installed llama-cpp-python build\n"
        "does not provide the required multimodal chat handler.\n\n"
        f"Model: {model_name}\n"
        f"Detected model family: {family}\n"
        f"Required handler: {handler}\n"
        f"{unknown_family_note}"
        "\n"
        f"Installed llama-cpp-python: {pkg_ver}\n"
        f"llama.cpp backend: {backend_ver}\n"
        "\n"
        "This can happen when the model family is newer than the installed\n"
        "llama-cpp-python build, or when the build does not include that multimodal\n"
        "chat handler yet.\n\n"
        "Please check the upstream JamePeng llama-cpp-python:\n"
        f"{JAMEPENG_LLAMA_CPP_URL}"
    )


# ============================================================================
# llama-cpp-python Integration
# ============================================================================

try:
    from llama_cpp import Llama

    llama_chat_format = import_module("llama_cpp.llama_chat_format")

    chat_handler_map, chat_handler_factory_map, chat_handler_class_registry = _load_available_chat_handlers(
        handler_map=chat_handler_map,
        handler_kwargs_map=CHAT_HANDLER_KWARGS_MAP,
        handler_module=llama_chat_format,
    )

    LLAMA_CPP_AVAILABLE = True

except Exception as e:
    LLAMA_CPP_AVAILABLE = False
    _LLAMA_CPP_IMPORT_ERROR = repr(e)

    chat_handler_map = {}
    chat_handler_factory_map = {}
    chat_handler_class_registry = _empty_chat_handler_registry()

    print(f"[LLM Session] Warning: llama-cpp-python not available: {_LLAMA_CPP_IMPORT_ERROR}")

# ============================================================================
# Model Discovery & Path Resolution
# ============================================================================

_LLM_MODELS_DIR_NAME = "LLM"
_LLM_SESSION_CATEGORY = "LLM/Session"
_NO_GGUF_MODELS_PLACEHOLDER = f"(No GGUF models found in models/{_LLM_MODELS_DIR_NAME}/)"
_MMPROJ_AUTO = "(Auto-detect)"
_MMPROJ_NOT_REQUIRED = "(Not required)"
_LOG_LEVEL_OPTIONS = list(LOG_LEVEL_OPTIONS)
_PERSISTENT_CACHE_OPTIONS = list(PERSISTENT_CACHE_OPTIONS)
_RUNTIME_CACHE_OPTIONS = list(RUNTIME_CACHE_OPTIONS)

def _list_gguf_recursive(models_dir: str) -> tuple[list[str], list[str]]:
    """
    models_dir 配下を再帰的に探索し、.gguf を相対パスで返す。
    - model: mmproj 以外
    - mmproj: ファイル名が mmproj で始まるもの
    """
    base = Path(models_dir)
    if not base.exists():
        return [], []

    models: list[str] = []
    mmprojs: list[str] = []

    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".gguf":
            continue
        rel = p.relative_to(base).as_posix()  # サブフォルダを含む相対パス
        if p.name.lower().startswith("mmproj"):
            mmprojs.append(rel)
        else:
            models.append(rel)

    models.sort(key=str.lower)
    mmprojs.sort(key=str.lower)
    return models, mmprojs

def _get_llm_model_roots() -> list[str]:
    """
    Collect LLM model roots from:
    - default ComfyUI models/LLM
    - extra_model_paths.yaml entries registered as folder key "LLM" / "llm"
    """
    roots: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if not path:
            return
        norm = os.path.normpath(os.path.abspath(path))
        key = norm.lower()
        if key in seen:
            return
        if not os.path.isdir(norm):
            return
        seen.add(key)
        roots.append(norm)

    # Primary ComfyUI models/LLM
    _add(os.path.join(folder_paths.models_dir, _LLM_MODELS_DIR_NAME))

    # Extra paths loaded via extra_model_paths.yaml
    if hasattr(folder_paths, "get_folder_paths"):
        for folder_key in ("LLM", "llm"):
            try:
                for p in folder_paths.get_folder_paths(folder_key):
                    _add(p)
            except Exception:
                pass

    return roots

def _list_gguf_recursive_multi(roots: list[str]) -> tuple[list[str], list[str]]:
    """
    Search multiple roots and return merged relative paths.
    Relative paths are deduplicated (first root wins if names collide).
    """
    models: list[str] = []
    mmprojs: list[str] = []
    seen_models: set[str] = set()
    seen_mmprojs: set[str] = set()

    for root in roots or []:
        m_list, mp_list = _list_gguf_recursive(root)
        for rel in m_list:
            k = rel.lower()
            if k in seen_models:
                continue
            seen_models.add(k)
            models.append(rel)
        for rel in mp_list:
            k = rel.lower()
            if k in seen_mmprojs:
                continue
            seen_mmprojs.add(k)
            mmprojs.append(rel)

    models.sort(key=str.lower)
    mmprojs.sort(key=str.lower)
    return models, mmprojs

def _is_no_models_placeholder(model: Optional[str]) -> bool:
    return _NO_GGUF_MODELS_PLACEHOLDER in (model or "")

def _safe_join_under(base_dir: str, rel_path: str) -> str:
    base = Path(base_dir).resolve()
    p = (base / rel_path).resolve()
    # パストラバーサル対策：base 配下に解決されることを保証
    if base != p and base not in p.parents:
        raise ValueError(f"Invalid path (outside models_dir): {rel_path}")
    return str(p)

def _resolve_llm_relpath(rel_path: str, roots: Optional[list[str]] = None) -> str:
    """
    Resolve a user-selected relative model path under known LLM roots.
    Returns the first existing match; falls back to the first root for validation.
    """
    roots = roots or _get_llm_model_roots()
    if not roots:
        default_root = os.path.join(folder_paths.models_dir, _LLM_MODELS_DIR_NAME)
        return _safe_join_under(default_root, rel_path)

    for root in roots:
        try:
            candidate = _safe_join_under(root, rel_path)
        except Exception:
            continue
        if os.path.exists(candidate):
            return candidate

    return _safe_join_under(roots[0], rel_path)


def _resolve_model_and_mmproj(roots: list[str], model: str, mmproj: str) -> tuple[str, Optional[str]]:
    model_path = _resolve_llm_relpath(model, roots=roots)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    mmproj_path = None
    if mmproj == _MMPROJ_NOT_REQUIRED:
        mmproj_path = _MMPROJ_NOT_REQUIRED  # sentinel
    elif mmproj != _MMPROJ_AUTO:
        mmproj_path = _resolve_llm_relpath(mmproj, roots=roots)
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"mmproj not found: {mmproj_path}")

    return model_path, mmproj_path


# ============================================================================
# Image + Language Utilities
# ============================================================================

def encode_image_base64(pil_image: Image.Image, max_pixels: int = 262144) -> str:
    """
    Convert PIL image to base64 encoded string
    
    Args:
        pil_image: PIL Image object
        max_pixels: Maximum pixels for token saving
    
    Returns:
        Base64 encoded image string
    """
    # Resize image
    width, height = pil_image.size
    total_pixels = width * height
    
    if total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # JPEG compress and base64 encode
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", optimize=True, quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def tensor2pil(image_tensor) -> List[Image.Image]:
    """
    Convert ComfyUI tensor to PIL Image list
    
    Args:
        image_tensor: ComfyUI image tensor (B, H, W, C)
    
    Returns:
        List of PIL Image objects
    """
    batch_count = image_tensor.size(0) if len(image_tensor.shape) > 3 else 1
    
    if batch_count > 1:
        out = []
        for i in range(batch_count):
            out.extend(tensor2pil(image_tensor[i]))
        return out
    
    # Convert to numpy array (0-255)
    np_image = np.clip(255.0 * image_tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return [Image.fromarray(np_image)]

def detect_language(text: str) -> str:
    """
    Detect language from text (simple)
    
    Args:
        text: Input text
    
    Returns:
        'zh', 'ja', or 'en'
    """
    # Check for CJK character ranges
    has_chinese = False
    has_japanese = False
    
    for char in text:
        # Japanese-specific characters (Hiragana/Katakana)
        if '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':
            has_japanese = True
        # Chinese characters (Hanzi/Kanji - shared but check for Japanese context)
        elif '\u4e00' <= char <= '\u9fff':
            has_chinese = True
    
    # Japanese has priority if Hiragana/Katakana detected
    if has_japanese:
        return 'ja'
    if has_chinese:
        return 'zh'
    
    return 'en'

def _detect_history_language(history: Dict[str, Any]) -> str:
    """
    Detect primary language from conversation history
    
    Args:
        history: Conversation history dictionary
    
    Returns:
        'zh', 'ja', or 'en'
    """
    turns = history.get("turns", [])
    if not turns:
        return "en"
    
    # Check last few turns (up to 3)
    recent_text = ""
    for turn in turns[-3:]:
        recent_text += turn.get("user", {}).get("text", "") + " "
        recent_text += turn.get("assistant", {}).get("text", "") + " "
    
    if not recent_text.strip():
        return "en"
    
    return detect_language(recent_text)

# ============================================================================
# History I/O and Session Storage (file-based, ComfyUI hidden)
# ============================================================================

def _now_iso_jst() -> str:
    """Return current time in ISO8601 with JST offset."""
    # Avoid depending on system tz; format explicitly as +09:00
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).isoformat(timespec="seconds")

def _safe_output_dir() -> str:
    """Best-effort output dir resolution across ComfyUI versions."""
    if hasattr(folder_paths, "get_output_directory"):
        try:
            return folder_paths.get_output_directory()
        except Exception:
            pass
    for attr in ("output_directory", "output_dir"):
        if hasattr(folder_paths, attr):
            d = getattr(folder_paths, attr)
            if isinstance(d, str) and d:
                return d
    # Fallback: cwd/output
    return os.path.join(os.getcwd(), "output")

def default_sessions_dir() -> str:
    return os.path.join(_safe_output_dir(), "llm_session_sessions")

def _transcript_path(session_id: str, history_dir: Optional[str] = None) -> str:
    """Return transcript file path: {session_id}.txt under the same base as history files."""
    return history_store.transcript_path(session_id, history_dir, default_sessions_dir())


def _append_transcript_lines(path: str, lines: List[str]) -> None:
    history_store.append_transcript_lines(path, lines)


def _session_cache_root(session_id: str, history_dir: Optional[str] = None) -> str:
    return history_store.session_cache_root(session_id, history_dir, default_sessions_dir())


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write JSON atomically-ish with .tmp + .bak."""
    history_store.atomic_write_json(path, obj)


def _next_turn_id(history: Dict[str, Any]) -> int:
    return history_store.next_turn_id(history)


def _get_context_turns(history: Dict[str, Any], max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
    return history_store.get_context_turns(history, max_turns=max_turns)


def load_history(
    session_id: str,
    history_dir: Optional[str],
    system_prompt: str,
    model_sig: Optional[Dict[str, Any]] = None,
    log_level: Optional[str] = None,
    reset_session: bool = _SIMPLE_WRAPPER_DEFAULTS["reset_session"],
) -> tuple[Dict[str, Any], str]:
    return history_store.load_history(
        session_id=session_id,
        history_dir=history_dir,
        system_prompt=system_prompt,
        model_sig=model_sig,
        log_level=log_level,
        reset_session=reset_session,
        default_dir=default_sessions_dir(),
        now_iso=_now_iso_jst,
    )

def _coerce_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


# ============================================================================
# Message Building + Generation Wrappers
# ============================================================================
# Some GGUF models / chat templates do not support the "system" role.
# For those models, we fall back by folding system messages into the first user turn.
# This keeps the node compatible with a broader set of instruction-tuned GGUF models
# without requiring model-specific templates.

def build_chat_messages(history: Dict[str, Any],
                        user_text: str,
                        image_tensor=None,
                        max_turns: Optional[int] = None,
                        summarize_old_history: bool = True,
                        system_prompt: str = "") -> List[Dict[str, Any]]:
    """Build chat-completion messages. Image is included only for this turn."""
    sys = (system_prompt or "").strip() or (history.get("system_prompt") or "").strip()
    summary = ""
    if summarize_old_history:
        summary = (history.get("summary") or {}).get("text") or ""

    turns = _get_context_turns(history, max_turns=max_turns)

    messages: List[Dict[str, Any]] = []
    if sys or summary.strip():
        system_parts: List[str] = []
        if sys:
            system_parts.append(sys)
        if summary.strip():
            # Keep summary in system, but merge into one system message so
            # templates that require a single leading system role still work.
            system_parts.append(f"Conversation summary:\n{summary.strip()}")
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for t in turns:
        u = ((t.get("user") or {}).get("text")) or ""
        a = ((t.get("assistant") or {}).get("text")) or ""
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})

    # Current turn
    if image_tensor is not None:
        try:
            pil_list = tensor2pil(image_tensor)
            if pil_list:
                # Use first image in batch for this turn (keeps it simple in phase 1)
                img_b64 = encode_image_base64(pil_list[0])
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": user_text or ""}
                    ]
                })
                return messages
        except Exception:
            # fall through to text-only
            pass

    messages.append({"role": "user", "content": user_text or ""})
    return messages


def _fold_system_into_user_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a new messages list with all system-role contents folded into the first user message."""
    sys_parts: List[str] = []
    out: List[Dict[str, Any]] = []

    for msg in (messages or []):
        if (msg or {}).get("role") == "system":
            c = (msg or {}).get("content")
            if isinstance(c, str) and c.strip():
                sys_parts.append(c.strip())
            continue
        out.append(msg)

    sys_text = "\n\n".join(sys_parts).strip()
    if not sys_text:
        return out

    # Find first user giving it the system prefix
    for i, msg in enumerate(out):
        if (msg or {}).get("role") != "user":
            continue
        content = (msg or {}).get("content")

        # Text-only user content
        if isinstance(content, str):
            out[i]["content"] = sys_text + "\n\n" + content
            return out

        # Multi-part content (e.g., image_url + text)
        if isinstance(content, list):
            # Try to prepend to the first text part; if none, insert one.
            new_list = []
            injected = False
            for part in content:
                if (not injected) and isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text") or ""
                    part = dict(part)
                    part["text"] = sys_text + "\n\n" + t
                    injected = True
                new_list.append(part)
            if not injected:
                new_list.append({"type": "text", "text": sys_text})
            out[i]["content"] = new_list
            return out

        # Unknown content type; replace with string
        out[i]["content"] = sys_text
        return out

    # No user message exists; prepend one
    return [{"role": "user", "content": sys_text}] + out


# Text-chat builders (for chat formats that prefer completion-style prompts)
def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "\n".join([p for p in parts if p]).strip()
    return str(content or "")


def _build_qwen35_text_prompt(messages: List[Dict[str, Any]], config: dict[str, Any]) -> tuple[str, list[str]]:
    system_message = _DEFAULT_SYSTEM_PROMPT
    prompt_parts: List[str] = []
    enable_thinking = bool(config.get("enable_thinking", False))

    for message in messages:
        role = str((message or {}).get("role") or "")
        content = _message_content_to_text((message or {}).get("content"))
        if role == "system":
            if content.strip():
                system_message = content
            continue
        if role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            continue
        if role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    if prompt_parts:
        prompt += "\n".join(prompt_parts) + "\n"
    prompt += "<|im_start|>assistant\n"
    if enable_thinking:
        prompt += "<think>\n"
    else:
        prompt += "<think>\n\n</think>\n\n"
    return prompt, ["<|im_end|>", "<|endoftext|>"]


def _build_gemma4_text_prompt(messages: List[Dict[str, Any]], config: dict[str, Any]) -> tuple[str, list[str]]:
    system_parts: List[str] = []
    turns: List[tuple[str, str]] = []
    enable_thinking = bool(config.get("enable_thinking", False))

    for message in messages:
        role = str((message or {}).get("role") or "")
        content = _message_content_to_text((message or {}).get("content"))
        if role == "system":
            if content.strip():
                system_parts.append(content.strip())
            continue
        if role == "user":
            turns.append(("user", content))
            continue
        if role == "assistant":
            turns.append(("model", content))

    system_message = "\n\n".join(system_parts).strip()
    if enable_thinking:
        system_message = f"<|think|>\n{system_message}".strip()
    if system_message:
        if turns and turns[0][0] == "user":
            role, content = turns[0]
            turns[0] = (role, f"{system_message}\n\n{content}".strip())
        else:
            turns.insert(0, ("user", system_message))

    prompt = ""
    for role, content in turns:
        prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"

    prompt += "<start_of_turn>model\n"
    return prompt, ["<end_of_turn>", "<eos>"]


def _build_minicpm_v46_text_prompt(messages: List[Dict[str, Any]], config: dict[str, Any]) -> tuple[str, list[str]]:
    prompt_parts: List[str] = []
    enable_thinking = bool(config.get("enable_thinking", False))

    for message in messages:
        role = str((message or {}).get("role") or "")
        content = _message_content_to_text((message or {}).get("content")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    prompt = ""
    if prompt_parts:
        prompt += "\n".join(prompt_parts) + "\n"
    prompt += "<|im_start|>assistant\n"
    if enable_thinking:
        prompt += "<think>\n"
    else:
        prompt += "<think>\n\n</think>\n\n"
    return prompt, ["<|endoftext|>", "<|im_end|>"]


def _build_text_chat_request(
    model_path: str,
    mmproj_path: Optional[str],
    messages: List[Dict[str, Any]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    if mmproj_path not in (None, "", _MMPROJ_NOT_REQUIRED):
        return None
    for message in messages:
        content = (message or {}).get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") != "text":
                    return None
    model_family = _detect_model_family(model_path)
    if not model_family or model_family not in TEXT_CHAT_BUILDER_CONFIG_MAP:
        return None
    config = _get_text_chat_builder_config(
        model_family,
        text_chat_builder_overrides=text_chat_builder_overrides,
    )
    if model_family == "qwen3.5":
        prompt, stop = _build_qwen35_text_prompt(messages, config)
        return {
            "mode": "completion",
            "model_family": model_family,
            "prompt": prompt,
            "stop": stop,
            "config": config,
        }
    if model_family == "gemma4":
        prompt, stop = _build_gemma4_text_prompt(messages, config)
        return {
            "mode": "completion",
            "model_family": model_family,
            "prompt": prompt,
            "stop": stop,
            "config": config,
        }
    if model_family == "minicpm-v-4.6":
        prompt, stop = _build_minicpm_v46_text_prompt(messages, config)
        return {
            "mode": "completion",
            "model_family": model_family,
            "prompt": prompt,
            "stop": stop,
            "config": config,
        }
    return None


# Completion helpers
def _retry_kwargs_with_repeat_last_n_fallback(
    kwargs: Dict[str, Any],
    repeat_last_n: Optional[int],
) -> Dict[str, Any]:
    retried_kwargs = dict(kwargs)
    if "penalty_last_n" in retried_kwargs:
        retried_kwargs.pop("penalty_last_n", None)
        if repeat_last_n and int(repeat_last_n) > 0:
            retried_kwargs["repeat_last_n"] = int(repeat_last_n)
            return retried_kwargs
    retried_kwargs.pop("repeat_last_n", None)
    retried_kwargs.pop("repeat_penalty", None)
    retried_kwargs.pop("top_p", None)
    return retried_kwargs


def _make_suppress_backend_logs(suppress: bool):
    """Return a context manager that suppresses stdout/stderr if suppress is True."""
    if not suppress:
        return contextlib.nullcontext()
    class _Suppress:
        def __enter__(self):
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        def __exit__(self, exc_type, exc, tb):
            sys.stdout = self._stdout
            sys.stderr = self._stderr
            return False
    return _Suppress()

def _create_chat_completion_robust(llm: "Llama", messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """create_chat_completion with a fallback for models that do not support system role."""
    try:
        return llm.create_chat_completion(messages=messages, **kwargs)
    except Exception as e:
        s = str(e)
        if "System role not supported" not in s and "system role not supported" not in s:
            raise
        # Retry with system folded into user
        folded = _fold_system_into_user_messages(messages)
        return llm.create_chat_completion(messages=folded, **kwargs)

def _iter_chat_completion_robust(llm: "Llama", messages: List[Dict[str, Any]], **kwargs):
    """Streaming create_chat_completion with system-role fallback."""
    try:
        return llm.create_chat_completion(messages=messages, stream=True, **kwargs)
    except Exception as e:
        s = str(e)
        if "System role not supported" not in s and "system role not supported" not in s:
            raise
        folded = _fold_system_into_user_messages(messages)
        return llm.create_chat_completion(messages=folded, stream=True, **kwargs)

def _extract_stream_content(chunk: Any) -> str:
    """Best-effort extraction of streamed text from llama-cpp-python chunks."""
    try:
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if not choices:
            return ""
        choice = choices[0] or {}
        delta = choice.get("delta")
        if isinstance(delta, dict):
            return delta.get("content") or ""
        msg = choice.get("message")
        if isinstance(msg, dict):
            return msg.get("content") or ""
        txt = choice.get("text")
        return txt or ""
    except Exception:
        return ""

def _create_text_or_chat_completion(
    llm: "Llama",
    messages: List[Dict[str, Any]],
    *,
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    forced_text_chat_builder_overrides_map: Optional[Dict[str, Dict[str, Any]]] = None,
    repeat_last_n: Optional[int] = None,
    **kwargs,
) -> tuple[Dict[str, Any], bool]:
    """Use the same text-chat-builder path as normal chat when available."""
    text_chat_request = None
    if model_path:
        effective_text_chat_builder_overrides = _merge_text_chat_builder_overrides(
            model_path=model_path,
            base_overrides=text_chat_builder_overrides,
            forced_overrides_map=forced_text_chat_builder_overrides_map,
        )
        text_chat_request = _build_text_chat_request(
            model_path=model_path,
            mmproj_path=mmproj_path,
            messages=messages,
            text_chat_builder_overrides=effective_text_chat_builder_overrides,
        )

    completion_kwargs = dict(kwargs)

    def _execute_completion(active_kwargs: Dict[str, Any]):
        if text_chat_request is not None:
            return llm.create_completion(
                prompt=text_chat_request["prompt"],
                stop=text_chat_request["stop"],
                **active_kwargs,
            )
        return _create_chat_completion_robust(llm, messages, **active_kwargs)

    resp = run_with_typeerror_fallback(
        execute_with_kwargs=_execute_completion,
        completion_kwargs=completion_kwargs,
        retry_kwargs_with_repeat_last_n_fallback=_retry_kwargs_with_repeat_last_n_fallback,
        repeat_last_n=int(repeat_last_n or 0),
    )

    return resp, bool(text_chat_request is not None)

def _strip_reasoning_output(text: str) -> str:
    """Best-effort removal of exposed reasoning sections from model output."""
    s = str(text or "")
    if not s:
        return s

    # Some local models expose internal channel markup. If any channel name contains
    # "final", keep only the last such block's message body.
    channel_blocks = re.findall(
        r"(?is)<\|channel\|>\s*([^\r\n<]*)\s*<\|message\|>\s*(.*?)(?=<\|end\|>|<\|start\|>|<\|channel\|>|$)",
        s,
    )
    for channel_name, body in reversed(channel_blocks):
        if "final" in (channel_name or "").lower():
            cand = (body or "").strip()
            if cand:
                return cand

    # Gemma-family outputs may sometimes use non-standard channel delimiters like:
    #   <|channel>thought ... <channel|>final answer...
    # In that case, treat the last channel delimiter as the handoff point to final output.
    channel_delims = list(
        re.finditer(r"(?is)(<\|channel\|>|<\|channel>|<channel\|>)", s)
    )
    if channel_delims:
        last = channel_delims[-1]
        tail = s[last.end():].strip()
        if tail:
            return tail

    # If a closing think tag exists, keep only the tail after the last one.
    # Some models emit reasoning text without an opening <think> but still output </think>.
    if "</think>" in s.lower():
        idx = s.lower().rfind("</think>")
        s = s[idx + len("</think>"):].strip()
        if not s:
            return s

    # Remove explicit think tags first.
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    if not s:
        return s

    # Prefer explicit final-answer markers when present.
    marker_patterns = [
        r"(?is)\bfinal\s*answer\s*:\s*(.+)$",
        r"(?is)最終回答\s*[:：]\s*(.+)$",
        r"(?is)回答\s*[:：]\s*(.+)$",
    ]
    for pat in marker_patterns:
        m = re.search(pat, s)
        if m:
            cand = (m.group(1) or "").strip()
            if cand:
                return cand

    # Remove common leading reasoning headers if they are at the beginning.
    lead_reasoning = re.match(
        r"(?is)^\s*(thinking\s*process|reasoning|analysis)\s*[:：]\s*",
        s,
    )
    if lead_reasoning:
        tail = s[lead_reasoning.end():].strip()
        # Heuristic: keep last non-empty paragraph as answer candidate.
        parts = [p.strip() for p in re.split(r"\n\s*\n", tail) if p.strip()]
        if parts:
            return parts[-1]
        return ""

    return s


# ============================================================================
# Summarization Helpers
# ============================================================================

def _make_summary_prompt(existing_summary: str, turns_chunk: list) -> list:
    """
    Build messages for summarizing a chunk of turns.
    We keep the summary factual and compact to control n_ctx and speed.
    """
    chunk_lines = []
    for t in turns_chunk:
        u = (t.get("user") or {}).get("text", "")
        a = (t.get("assistant") or {}).get("text", "")
        if u:
            chunk_lines.append(f"USER: {u}")
        if a:
            chunk_lines.append(f"ASSISTANT: {a}")
    chunk_text = "\n".join(chunk_lines).strip()

    user_payload = []
    if existing_summary.strip():
        user_payload.append("Existing summary:\n" + existing_summary.strip())
    if chunk_text:
        user_payload.append("New conversation chunk:\n" + chunk_text)
    user_text = "\n\n".join(user_payload).strip() or "(empty)"

    return [
        {
            "role": "system",
            "content": (
                "Summarize the conversation into a compact rolling memory for later dialogue turns.\n\n"
                "Keep only:\n"
                "- named people and their roles\n"
                "- stable facts already stated in the conversation\n"
                "- important decisions, constraints, preferences, promises, and open TODOs\n"
                "- the current situation if it matters for the next turn\n\n"
                "Rules:\n"
                "- Do not add new information.\n"
                "- Do not infer unstated attributes.\n"
                "- Do not replace names with generic labels if a name is known.\n"
                "- If a detail is uncertain, omit it.\n"
                "- Prefer plain factual statements in Japanese.\n"
                "- Use 2 to 5 short lines.\n"
                "- Keep the wording concrete and minimal.\n"
                "- No headings, no bullet markers, no analysis, no reasoning."
            )
        },
        {"role": "user", "content": user_text}
    ]

def _summarize_with_model(model: "Llama", existing_summary: str, turns_chunk: list,
                         temperature: float, max_tokens: int, suppress_logs: bool = _SUMMARY_HELPER_DEFAULTS["suppress_logs"],
                         model_path: Optional[str] = None,
                         mmproj_path: Optional[str] = None,
                         text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                         advanced_generation_kwargs: Optional[Dict[str, Any]] = None) -> str:
    msgs = _make_summary_prompt(existing_summary, turns_chunk)
    extra_kwargs = dict(advanced_generation_kwargs or {})
    with _make_suppress_backend_logs(suppress_logs):
        resp, used_text_builder = _create_text_or_chat_completion(
            model,
            msgs,
            model_path=model_path,
            mmproj_path=mmproj_path,
            text_chat_builder_overrides=text_chat_builder_overrides,
            forced_text_chat_builder_overrides_map=SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            **extra_kwargs,
        )
    if used_text_builder:
        raw = (resp["choices"][0]["text"] or "").strip()
    else:
        raw = (resp["choices"][0]["message"]["content"] or "").strip()
    cleaned = _strip_reasoning_output(raw).strip()
    return cleaned or raw

def maybe_compact_summary(model: "Llama",
                          history: Dict[str, Any],
                          summary_max_chars: int = _FULL_UI_SESSION_CHAT_DEFAULTS["summary_max_chars"],
                          temperature: float = _SUMMARY_HELPER_DEFAULTS["temperature"],
                          max_tokens_summary: int = _FULL_UI_SESSION_CHAT_DEFAULTS["max_tokens_summary"],
                          suppress_logs: bool = _SUMMARY_HELPER_DEFAULTS["suppress_logs"],
                          model_path: Optional[str] = None,
                          mmproj_path: Optional[str] = None,
                          text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                          advanced_generation_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    If summary grows too large, re-summarize it to keep it compact.
    This is a safety valve to keep prompts small and fast.
    """
    sm = history.get("summary") or {}
    text = (sm.get("text") or "")
    limit = max(_SUMMARY_HELPER_DEFAULTS["min_summary_max_chars"], _coerce_int(summary_max_chars, _FULL_UI_SESSION_CHAT_DEFAULTS["summary_max_chars"]))
    if len(text) <= limit:
        return history

    msgs = [
        {
            "role": "system",
            "content": (
                "Rewrite the summary into a shorter rolling memory for later dialogue turns.\n\n"
                "Preserve:\n"
                "- named people and roles\n"
                "- stable facts\n"
                "- important constraints, preferences, promises, and unresolved points\n\n"
                "Rules:\n"
                "- Do not add or guess information.\n"
                "- Keep names if they are known.\n"
                "- Remove vague labels, analysis, and redundancy.\n"
                "- Write plain factual Japanese only.\n"
                "- Use 2 to 5 short lines.\n"
                "- No headings, no bullet markers, no reasoning."
            )
        },
        {"role": "user", "content": text}
    ]
    extra_kwargs = dict(advanced_generation_kwargs or {})
    with _make_suppress_backend_logs(suppress_logs):
        resp, used_text_builder = _create_text_or_chat_completion(
            model,
            msgs,
            model_path=model_path,
            mmproj_path=mmproj_path,
            text_chat_builder_overrides=text_chat_builder_overrides,
            forced_text_chat_builder_overrides_map=SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP,
            temperature=float(temperature),
            max_tokens=int(max_tokens_summary),
            **extra_kwargs,
        )
    if used_text_builder:
        raw = (resp["choices"][0]["text"] or "").strip()
    else:
        raw = (resp["choices"][0]["message"]["content"] or "").strip()
    compact = _strip_reasoning_output(raw).strip() or raw
    history.setdefault("summary", {})
    history["summary"]["enabled"] = True
    history["summary"]["text"] = compact
    history["summary"]["updated_at"] = _now_iso_jst()
    return history

def maybe_summarize_history(model: "Llama",
                           history: Dict[str, Any],
                           max_turns: int,
                           summarize_old_history: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["summarize_old_history"],
                           summary_chunk_turns: int = _FULL_UI_SESSION_CHAT_DEFAULTS["summary_chunk_turns"],
                           temperature: float = _SUMMARY_HELPER_DEFAULTS["temperature"],
                           max_tokens_summary: int = _FULL_UI_SESSION_CHAT_DEFAULTS["max_tokens_summary"],
                           summary_max_chars: int = _FULL_UI_SESSION_CHAT_DEFAULTS["summary_max_chars"],
                           suppress_logs: bool = _SUMMARY_HELPER_DEFAULTS["suppress_logs"],
                           model_path: Optional[str] = None,
                           mmproj_path: Optional[str] = None,
                           text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
                           advanced_generation_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Phase 1.5:
    - Do NOT summarize every turn. Summarize only when overflow reaches summary_chunk_turns.
    - Summarize a fixed-size chunk from the oldest overflow turns, then trim that chunk.
    - Keep a rolling summary in history['summary'] and keep it compact.
    """
    if not summarize_old_history:
        return history

    pending = _get_context_turns(history, max_turns=None)
    mt = max(0, _coerce_int(max_turns, _FULL_UI_SESSION_CHAT_DEFAULTS["max_turns"]))
    if mt == 0:
        return history

    if len(pending) <= mt:
        return history

    chunk = max(1, _coerce_int(summary_chunk_turns, _FULL_UI_SESSION_CHAT_DEFAULTS["summary_chunk_turns"]))
    overflow_n = len(pending) - mt
    if overflow_n < chunk:
        # Not enough overflow to justify a summary generation (saves time)
        return history

    # Take the oldest chunk from the currently unsummarized window.
    to_summarize = pending[:chunk]

    history.setdefault("summary", {})
    history["summary"]["enabled"] = True
    existing = (history["summary"].get("text") or "").strip()

    new_sum = _summarize_with_model(
        model=model,
        existing_summary=existing,
        turns_chunk=to_summarize,
        temperature=temperature,
        max_tokens=max_tokens_summary,
        suppress_logs=suppress_logs,
        model_path=model_path,
        mmproj_path=mmproj_path,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
    )

    history["summary"]["text"] = new_sum
    history["summary"]["updated_at"] = _now_iso_jst()
    last_chunk_id = 0
    for t in to_summarize:
        if isinstance(t, dict):
            last_chunk_id = max(last_chunk_id, _coerce_int(t.get("id"), 0))
    history["summary"]["covered_until_turn_id"] = max(
        _coerce_int(history["summary"].get("covered_until_turn_id"), 0),
        last_chunk_id,
    )

    # Compact summary if it grows too large
    history = maybe_compact_summary(
        model=model,
        history=history,
        summary_max_chars=summary_max_chars,
        temperature=temperature,
        max_tokens_summary=max_tokens_summary,
        suppress_logs=suppress_logs,
        model_path=model_path,
        mmproj_path=mmproj_path,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
    )

    return history


# ============================================================================
# Cache + Model Manager
# ============================================================================

class _LayeredCache:
    """
    Two-level cache wrapper.
    - primary: fast cache (RAM/Trie)
    - secondary: persistent cache (Disk)
    """

    def __init__(self, primary: Any, secondary: Any):
        self.primary = primary
        self.secondary = secondary

    def __contains__(self, key: Any) -> bool:
        try:
            if key in self.primary:
                return True
        except Exception:
            pass
        try:
            return key in self.secondary
        except Exception:
            return False

    def __getitem__(self, key: Any) -> Any:
        try:
            if key in self.primary:
                return self.primary[key]
        except Exception:
            pass
        value = self.secondary[key]
        # Read-through: populate primary on secondary hit.
        try:
            self.primary[key] = value
        except Exception:
            pass
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        # Write-through to both levels.
        self.primary[key] = value
        self.secondary[key] = value


class GGUFModelManager:

    def __init__(self):
        self.model: Optional[Llama] = None
        self.chat_handler = None
        self._current_cache_info = None

        # Keep a full signature of what's currently loaded
        self._signature: Optional[tuple] = None
        self.current_model_path: Optional[str] = None
        self.current_mmproj_path: Optional[str] = None

        # Optional override for where cache should be stored
        self.cache_dir_override: Optional[str] = None

    def _normalize_path(self, p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        return os.path.normpath(p)

    def _auto_detect_mmproj(self, model_path: str, model_family: str) -> Optional[str]:
        """
        Auto-detect mmproj using normalized_chat_format_map.

        Rule:
        - Find the key in normalized_chat_format_map whose value == model_family
        - In the model directory, scan mmproj-*.gguf
        - Keep those whose name after 'mmproj-' starts with the detected family key
        - If exactly one match -> return it
        - If 0 or multiple -> return None
        """

        model_dir = os.path.dirname(model_path)
        family_aliases = []

        if not os.path.isdir(model_dir):
            print("[GGUFModelManager] mmproj auto-detect failed: model directory does not exist.\n"
            f"dir: {model_dir}")
            return None

        # model_family → family key
        for k, v in normalized_chat_format_map.items():
            if v == model_family:
                family_aliases.append(k)

        if not family_aliases:
            print("mmproj auto-detect failed: no aliases found for model family.\n"
            f"family: {model_family}")
            return None                

        mmproj_files = []
        for f in os.listdir(model_dir):
            fl = f.lower()
            if fl.startswith("mmproj-") and fl.endswith(".gguf"):
                mmproj_files.append(f)

        matches = set()
        for f in mmproj_files:
            mmname = f[len("mmproj-"):-len(".gguf")]
            for k in family_aliases:
                if k in mmname.lower():
                        matches.add(f)

        matches = sorted(matches, key=str.lower)

        if len(matches) == 1:
            fname = matches[0]
            cand = os.path.join(model_dir, fname)
            print(f"Auto-detected mmproj (family={model_family}): {fname}")
            return self._normalize_path(cand)
        elif len(matches) == 0:
            print(
                "mmproj auto-detect failed: no mmproj matched the model family prefix.\n"
                f"family: {model_family}\n"
                f"aliases: {', '.join(sorted(family_aliases, key=str.lower))}\n"
                f"dir: {model_dir}\n"
                f"mmproj candidates: {', '.join(sorted(mmproj_files, key=str.lower)) or '(none)'}"
            )
        else:
            print(
                "mmproj auto-detect failed: multiple mmproj files matched the model family prefix.\n"
                f"family: {model_family}\n"
                f"aliases: {', '.join(sorted(family_aliases, key=str.lower))}\n"
                f"dir: {model_dir}\n"
                f"matched: {', '.join(matches)}\n"
                "Please select mmproj manually."
            )

        return None

    def _make_signature(
        self,
        model_path: str,
        mmproj_path: Optional[str],
        n_ctx: int,
        n_gpu_layers: int,
        tensor_split: Optional[List[float]],
        use_vision: bool,
        chat_handler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        return (
            self._normalize_path(model_path),
            self._normalize_path(mmproj_path),
            int(n_ctx),
            int(n_gpu_layers),
            tuple(float(x) for x in tensor_split) if tensor_split is not None else None,
            bool(use_vision),
            json.dumps(chat_handler_kwargs or {}, sort_keys=True, ensure_ascii=True),
        )

    def load_model(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        tensor_split: Optional[List[float]] = None,
        chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        vision_required: bool = False,
        verbose: bool = False,
    ) -> Llama:
        """Load GGUF model."""
        _require_llama_cpp_available()

        model_path = self._normalize_path(model_path)
        tensor_split = _normalize_tensor_split(tensor_split)

        # If user explicitly selected "(Not required)", force text-only.
        force_no_mmproj = (mmproj_path == _MMPROJ_NOT_REQUIRED)
        if force_no_mmproj and vision_required:
            raise RuntimeError("Vision is required but mmproj was explicitly disabled.")
        if force_no_mmproj:
            mmproj_path = None

        ### Load Chat Handler
        chat_handler = None
        chat_format = None
        use_vision = False
        matched_vision_family = False
        active_chat_handler_kwargs: Dict[str, Any] = {}

        model_name_lower = os.path.basename(model_path).lower()

        for k, v in normalized_chat_format_map.items():

            if k in model_name_lower:
                model_family = v
                matched_vision_family = True
                handler_name = DECLARED_CHAT_HANDLER_MAP.get(model_family)

                if model_family not in chat_handler_factory_map:
                    if vision_required:
                        msg = _format_multimodal_handler_unavailable_error(
                            model_path=model_path,
                            model_family=model_family,
                            required_handler=handler_name,
                        )
                        raise RuntimeError(msg)
                    print(
                        "[GGUFModelManager] Warning: Multimodal chat handler unavailable "
                        f"for {model_family}; falling back to text-only mode."
                    )
                    break

                if not force_no_mmproj:
                    if mmproj_path is None:
                        mmproj_path = self._auto_detect_mmproj(model_path, model_family)
                        if mmproj_path is None:
                            model_dir = os.path.dirname(model_path)
                            msg = (
                                "[GGUFModelManager] Failed to auto-detect mmproj file "
                                f"for model family {model_family} in {model_dir}"
                            )
                            if vision_required:
                                raise RuntimeError(msg)
                            print(f"{msg}; falling back to text-only mode.")
                    else:
                        mmproj_path = self._normalize_path(mmproj_path)
                
                if mmproj_path is not None:
                    print(f"[GGUFModelManager] Using mmproj: {mmproj_path}")
                    try:
                        handler_name = chat_handler_map.get(model_family)
                        handler_cls = chat_handler_class_registry.get(handler_name) if handler_name else None
                        if handler_cls is None:
                            raise RuntimeError(
                                _format_multimodal_handler_unavailable_error(
                                    model_path=model_path,
                                    model_family=model_family,
                                    required_handler=DECLARED_CHAT_HANDLER_MAP.get(model_family),
                                )
                            )
                        active_chat_handler_kwargs = _get_chat_handler_kwargs(
                            model_family,
                            chat_handler_overrides=chat_handler_overrides,
                        )
                        _warn_if_gemma4_vision_thinking_required(
                            model_path,
                            model_family,
                            active_chat_handler_kwargs,
                        )
                        chat_handler = handler_cls(
                            clip_model_path=mmproj_path,
                            **active_chat_handler_kwargs,
                        )
                        chat_format = model_family
                        use_vision = True
                    except Exception as e:
                        if vision_required:
                            raise RuntimeError(f"Vision chat handler initialization failed:\n{e}") from e
                        print(f"[GGUFModelManager] Warning: Failed to initialize chat handler: {e}")
                        chat_handler = None
                        chat_format = None
                        use_vision = False
                else:
                    chat_handler = None
                    chat_format = None
                    use_vision = False

                break

        if vision_required and not matched_vision_family:
            raise RuntimeError(
                _format_multimodal_handler_unavailable_error(
                    model_path=model_path,
                    model_family=None,
                    required_handler=None,
                )
            )

        if not use_vision:
            print("[GGUFModelManager] Using text-only mode")
            chat_handler = None
            use_vision = False

        new_sig = self._make_signature(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            use_vision=use_vision,
            chat_handler_kwargs=active_chat_handler_kwargs,
        )

        # If signature matches, reuse
        if self.model is not None and self._signature == new_sig:
            print(f"[GGUFModelManager] Using cached model: {model_path}")
            return self.model

        # Otherwise, explicitly unload to avoid stale mmproj/handler state
        if self.model is not None:
            print("[GGUFModelManager] Signature changed -> unloading previous model to avoid stale state")
            self.unload_model()

        print(f"[GGUFModelManager] Loading model: {model_path}")
        print(f"[GGUFModelManager] n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}")
        if tensor_split is not None:
            print(f"[GGUFModelManager] tensor_split={tensor_split}")

        # Store handler on manager (used later to decide if images are supported)
        self.chat_handler = chat_handler
        self.chat_format = chat_format

        # Model loading
        llama_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if tensor_split is not None:
            llama_kwargs["tensor_split"] = tensor_split

        if use_vision and self.chat_handler is not None:
            print("[GGUFModelManager] Loading with vision support")
            self.model = Llama(
                **llama_kwargs,
                chat_handler=self.chat_handler,
                chat_format=self.chat_format,
                # Vision models often need this; safe default for vision path.
                logits_all=True,
            )
        else:
            print("[GGUFModelManager] Loading in text-only mode")
            self.model = Llama(
                **llama_kwargs,
                # chat_format=self.chat_format,
            )

        self.current_model_path = model_path
        self.current_mmproj_path = self._normalize_path(mmproj_path)
        self._signature = new_sig

        print("[GGUFModelManager] Model loaded successfully")
        return self.model

    def _runtime_cache_fingerprint(self) -> str:
        """Best-effort runtime fingerprint to avoid reusing incompatible on-disk states."""
        pkg_ver = "unknown"
        backend_ver = "unknown"
        try:
            import llama_cpp  # type: ignore
            pkg_ver = str(getattr(llama_cpp, "__version__", "unknown"))
            try:
                backend_fn = getattr(getattr(llama_cpp, "llama_cpp", None), "llama_cpp_version", None)
                if callable(backend_fn):
                    backend_ver = str(backend_fn())
            except Exception:
                pass
        except Exception:
            pass
        return f"llama_cpp_py={pkg_ver}|llama_cpp_backend={backend_ver}"

    def _default_cache_dir(
        self,
        model_path: str,
        mmproj_path: str,
        n_ctx: int,
        n_gpu_layers: int = 0,
        tensor_split: Optional[List[float]] = None,
    ) -> str:
        """
        Compute a stable cache directory for cache data.
        Cache root can be scoped by session; under that root, we key by model settings
        so incompatible configurations do not share the same disk cache directory.
        """
        base = (self.cache_dir_override or os.path.join(_safe_output_dir(), "llm_session_sessions", "cache"))
        os.makedirs(base, exist_ok=True)
        key_src = (
            f"{os.path.abspath(model_path)}|{os.path.abspath(mmproj_path or '')}|"
            f"n_ctx={int(n_ctx)}|n_gpu_layers={int(n_gpu_layers)}|"
            f"tensor_split={json.dumps(tensor_split or [], sort_keys=True)}|"
            f"chat_format={self.chat_format or ''}|use_vision={bool(self.chat_handler is not None)}|"
            f"{self._runtime_cache_fingerprint()}"
        )
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join(base, key)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def configure_cache(
        self,
        llm,
        model_path: str,
        mmproj_path: str,
        n_ctx: int,
        n_gpu_layers: int = 0,
        tensor_split: Optional[List[float]] = None,
        persistent_cache: str = _FULL_UI_SESSION_CHAT_DEFAULTS["persistent_cache"],
        runtime_cache: str = _FULL_UI_SESSION_CHAT_DEFAULTS["runtime_cache"],
    ) -> None:
        """
        Configure cache backends.
        - persistent_cache: LlamaDiskCache / off
        - runtime_cache: KV_cache / LlamaRAMCache / LlamaTrieCache / off

        Note:
        - KV_cache is handled separately by save_state/load_state logic.
        - cache object attached to llm is composed from runtime/persistent backends.
        """
        persistent_cache = str(persistent_cache or _PERSISTENT_CACHE_OPTIONS[1])
        runtime_cache = str(runtime_cache or _RUNTIME_CACHE_OPTIONS[3])
        try:
            # llama-cpp-python provides cache helpers in many versions (names vary).
            import llama_cpp
        except Exception as e:
            if persistent_cache != _PERSISTENT_CACHE_OPTIONS[1] or runtime_cache not in (
                _RUNTIME_CACHE_OPTIONS[3],
                _RUNTIME_CACHE_OPTIONS[0],
            ):
                print(f"[GGUFModelManager] Cache requested but llama_cpp import failed: {e}")
            return

        # Determine desired cache object
        cache_obj: Any = None
        cache_desc = _PERSISTENT_CACHE_OPTIONS[1]
        disk_dir: Optional[str] = None
        try:
            runtime_obj = None
            runtime_desc = _RUNTIME_CACHE_OPTIONS[3]
            if runtime_cache == _RUNTIME_CACHE_OPTIONS[1]:
                if hasattr(llama_cpp, "LlamaRAMCache"):
                    runtime_obj = llama_cpp.LlamaRAMCache()
                    runtime_desc = "LlamaRAMCache"
                elif hasattr(llama_cpp, "RAMCache"):
                    runtime_obj = llama_cpp.RAMCache()
                    runtime_desc = "RAMCache"
                else:
                    print("[GGUFModelManager] Runtime cache requested but RAM cache class is unavailable")
            elif runtime_cache == _RUNTIME_CACHE_OPTIONS[2]:
                if hasattr(llama_cpp, "LlamaTrieCache"):
                    runtime_obj = llama_cpp.LlamaTrieCache()
                    runtime_desc = "LlamaTrieCache"
                else:
                    print("[GGUFModelManager] Runtime cache requested but LlamaTrieCache is unavailable")

            persistent_obj = None
            persistent_desc = _PERSISTENT_CACHE_OPTIONS[1]
            if persistent_cache == _PERSISTENT_CACHE_OPTIONS[0]:
                cache_dir = self._default_cache_dir(
                    model_path,
                    mmproj_path,
                    n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    tensor_split=tensor_split,
                )
                disk_dir = cache_dir
                if hasattr(llama_cpp, "LlamaDiskCache"):
                    persistent_obj = llama_cpp.LlamaDiskCache(cache_dir)
                    persistent_desc = f"LlamaDiskCache:{cache_dir}"
                elif hasattr(llama_cpp, "DiskCache"):
                    persistent_obj = llama_cpp.DiskCache(cache_dir)
                    persistent_desc = f"DiskCache:{cache_dir}"
                elif hasattr(llama_cpp, "LlamaCache"):
                    try:
                        persistent_obj = llama_cpp.LlamaCache(cache_dir)
                        persistent_desc = f"LlamaCache:{cache_dir}"
                    except Exception as e:
                        # P1: Log persistent cache creation failure
                        log_error_safely("GGUFModelManager", e, "Failed to create LlamaCache", LOG_LEVEL_MINIMAL)
                        persistent_obj = None
                if persistent_obj is None:
                    print("[GGUFModelManager] Persistent cache requested but disk cache class is unavailable")

            if runtime_obj is not None and persistent_obj is not None:
                cache_obj = _LayeredCache(runtime_obj, persistent_obj)
                cache_desc = f"runtime={runtime_desc};persistent={persistent_desc}"
            elif runtime_obj is not None:
                cache_obj = runtime_obj
                cache_desc = f"runtime={runtime_desc}"
            elif persistent_obj is not None:
                cache_obj = persistent_obj
                cache_desc = f"persistent={persistent_desc}"
            else:
                cache_obj = None

            if cache_obj is None:
                # Disable cache when no cache backend is selected/available.
                try:
                    if hasattr(llm, "set_cache"):
                        llm.set_cache(None)
                    elif hasattr(llm, "cache"):
                        llm.cache = None
                except Exception:
                    pass
                self._current_cache_info = None
                print("[GGUFModelManager] Cache: OFF")
                return

            # Apply cache to model (different versions expose different APIs)
            applied = False
            if hasattr(llm, "set_cache"):
                llm.set_cache(cache_obj)
                applied = True
            # Some versions accept .cache attribute
            if not applied and hasattr(llm, "cache"):
                try:
                    llm.cache = cache_obj
                    applied = True
                except Exception:
                    pass

            if applied:
                self._current_cache_info = {
                    "persistent_cache": persistent_cache,
                    "runtime_cache": runtime_cache,
                    "desc": cache_desc,
                    "disk_dir": disk_dir,
                }
                print(f"[GGUFModelManager] Cache enabled: {cache_desc}")
            else:
                print("[GGUFModelManager] Cache object created but could not be applied")
        except Exception as e:
            print(
                "[GGUFModelManager] Cache setup failed "
                f"(persistent={persistent_cache}, runtime={runtime_cache}): {e}"
            )
            # Fail closed: detach possibly stale cache object after setup failure.
            try:
                if hasattr(llm, "set_cache"):
                    llm.set_cache(None)
                elif hasattr(llm, "cache"):
                    llm.cache = None
            except Exception:
                pass
            self._current_cache_info = None

    def invalidate_cache(self, llm, remove_disk_data: bool = False) -> None:
        """
        Best-effort invalidation for cache after state mismatch.
        - Detach cache from model
        - Optionally remove disk cache directory
        """
        info = getattr(self, "_current_cache_info", None) or {}
        desc = str(info.get("desc", "") or "")
        try:
            if hasattr(llm, "set_cache"):
                llm.set_cache(None)
            elif hasattr(llm, "cache"):
                llm.cache = None
        except Exception:
            pass

        disk_dir = str(info.get("disk_dir", "") or "")
        if remove_disk_data and not disk_dir and desc.startswith("disk:"):
            disk_dir = desc[len("disk:"):].strip()
        if remove_disk_data and disk_dir:
            if disk_dir:
                try:
                    shutil.rmtree(disk_dir, ignore_errors=True)
                    os.makedirs(disk_dir, exist_ok=True)
                except Exception as e:
                    # P1: Log disk cache removal failure
                    log_error_safely("GGUFModelManager", e, f"Failed to remove disk cache: {disk_dir}", LOG_LEVEL_MINIMAL)

        self._current_cache_info = None

    def unload_model(self):
        """Unload model from memory."""
        if self.model is not None:
            print(f"[GGUFModelManager] Unloading model: {self.current_model_path}")
        try:
            if self.model is not None:
                # Detach any runtime/persistent cache (RAM/Trie/Disk) from llama-cpp.
                try:
                    self.invalidate_cache(self.model, remove_disk_data=False)
                except Exception:
                    pass
            if self.model is not None:
                del self.model
        finally:
            self.model = None

        try:
            if self.chat_handler is not None:
                del self.chat_handler
        finally:
            self.chat_handler = None

        self.current_model_path = None
        self.current_mmproj_path = None
        self._signature = None
        try:
            global _runtime_container
            container = _runtime_container
            if container is not None:
                container.mem_kv_state.clear()
        except Exception:
            pass

        # Encourage timely cleanup (important for llama-cpp backends and mmproj state)
        import gc as _gc
        _gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            if hasattr(_torch, "xpu") and _torch.xpu.is_available():
                _torch.xpu.empty_cache()
        except Exception:
            pass

_runtime_container: Optional[RuntimeContainer] = None


def _create_default_runtime_container() -> RuntimeContainer:
    return RuntimeContainer(model_manager=GGUFModelManager())


def _resolve_runtime_container(
    runtime_container: Optional[RuntimeContainer] = None,
) -> RuntimeContainer:
    global _runtime_container
    if runtime_container is not None:
        return runtime_container
    if _runtime_container is None:
        _runtime_container = _create_default_runtime_container()
    return _runtime_container


def _get_or_create_dialogue_cycle_model_manager(
    role: str,
    runtime_container: Optional[RuntimeContainer] = None,
) -> "GGUFModelManager":
    container = _resolve_runtime_container(runtime_container)
    key = (role or "A").strip().upper() or "A"
    manager = container.dialogue_model_managers.get(key)
    if manager is None:
        manager = GGUFModelManager()
        container.dialogue_model_managers[key] = manager
    return manager


def _unload_dialogue_cycle_model_managers(
    runtime_container: Optional[RuntimeContainer] = None,
) -> None:
    container = _resolve_runtime_container(runtime_container)
    managers = container.dialogue_model_managers
    seen: set[int] = set()
    for manager in list(managers.values()):
        if manager is None:
            continue
        manager_id = id(manager)
        if manager_id in seen:
            continue
        seen.add(manager_id)
        try:
            manager.unload_model()
        except Exception:
            pass
    managers.clear()


def _unload_runtime_container_managers(
    runtime_container: Optional[RuntimeContainer] = None,
) -> None:
    container = _resolve_runtime_container(runtime_container)
    manager = container.model_manager
    if manager is not None:
        try:
            manager.unload_model()
        except Exception:
            pass
    container.model_manager = None
    _unload_dialogue_cycle_model_managers(runtime_container=container)


def _is_state_data_mismatch_error(err: Exception) -> bool:
    s = str(err or "")
    if "Failed to set llama state data" in s:
        return True
    if "'NoneType' object has no attribute 'input_ids'" in s:
        return True
    if "cache_item.input_ids" in s and "NoneType" in s:
        return True
    return False


def _processing_interrupted() -> bool:
    try:
        import comfy.model_management  # type: ignore
        return bool(comfy.model_management.processing_interrupted())
    except Exception:
        return False


def _throw_if_processing_interrupted() -> None:
    try:
        import comfy.model_management  # type: ignore
    except ImportError:
        return
    comfy.model_management.throw_exception_if_processing_interrupted()


def _is_interrupt_error(err: Exception) -> bool:
    try:
        import comfy.model_management  # type: ignore
        return isinstance(err, comfy.model_management.InterruptProcessingException)
    except Exception:
        return False


def _cache_debug_label(manager: Optional["GGUFModelManager"]) -> str:
    try:
        info = getattr(manager, "_current_cache_info", None) or {}
        desc = str(info.get("desc", "") or "off")
        disk_dir = str(info.get("disk_dir", "") or "")
        if disk_dir:
            return f"{desc};disk_dir={disk_dir}"
        return desc
    except Exception:
        return "unknown"


def _clear_kv_state_for_session(
    session_id: str,
    runtime_container: Optional[RuntimeContainer] = None,
) -> None:
    try:
        container = _resolve_runtime_container(runtime_container)
        container.mem_kv_state.pop(session_id, None)
    except Exception:
        pass


def _kv_state_debug_info(state: Any) -> str:
    """Return a compact debug string for llama state payload."""
    try:
        stype = type(state).__name__
    except Exception:
        stype = "(unknown)"
    size = "n/a"
    llama_state_size = "n/a"
    try:
        size = str(len(state))
    except Exception:
        try:
            nbytes = getattr(state, "nbytes", None)
            if nbytes is not None:
                size = str(int(nbytes))
        except Exception:
            pass
    try:
        lss = getattr(state, "llama_state_size", None)
        if lss is not None:
            llama_state_size = str(int(lss))
    except Exception:
        pass
    return f"type={stype}, size={size}, llama_state_size={llama_state_size}"


def _saved_llama_state_size(state: Any) -> Optional[int]:
    """Best-effort extraction of llama_state_size from LlamaState-like payload."""
    try:
        lss = getattr(state, "llama_state_size", None)
        if lss is not None:
            return int(lss)
    except Exception:
        pass
    try:
        raw = getattr(state, "llama_state", None)
        if raw is not None:
            return int(len(raw))
    except Exception:
        pass
    return None


def _current_llama_state_size(llm: Any) -> Optional[int]:
    """Best-effort query of current backend state size for compatibility pre-check."""
    try:
        import llama_cpp  # type: ignore
        capi = getattr(llama_cpp, "llama_cpp", None)
        if capi is None:
            return None
        fn = getattr(capi, "llama_state_get_size", None) or getattr(capi, "llama_get_state_size", None)
        if not callable(fn):
            return None
        ctx_obj = getattr(getattr(llm, "_ctx", None), "ctx", None)
        if ctx_obj is None:
            return None
        return int(fn(ctx_obj))
    except Exception:
        return None


def _build_mmproj_options(available_mmprojs: list[str]) -> list[str]:
    if available_mmprojs:
        return available_mmprojs + [_MMPROJ_AUTO, _MMPROJ_NOT_REQUIRED]
    return [_MMPROJ_AUTO, _MMPROJ_NOT_REQUIRED]


# ============================================================================
# UI Definition Helpers
# ============================================================================

def _get_available_models_and_mmprojs() -> tuple[list[str], list[str]]:
    roots = _get_llm_model_roots()
    available_models, available_mmprojs = _list_gguf_recursive_multi(roots)

    mmproj_options = _build_mmproj_options(available_mmprojs)

    if not available_models:
        available_models = [_NO_GGUF_MODELS_PLACEHOLDER]

    return available_models, mmproj_options


def _ui_model_input(available_models: list[str], tooltip: str) -> tuple:
    return (available_models, {"default": available_models[0], "tooltip": tooltip})


def _ui_mmproj_input(mmproj_options: list[str], tooltip: str) -> tuple:
    return (mmproj_options, {"default": _MMPROJ_AUTO, "tooltip": tooltip})

def _ui_int_input(
    default: int,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    step: Optional[int] = None,
    tooltip: Optional[str] = None,
) -> tuple:
    options: Dict[str, Any] = {"default": default}
    if min_value is not None:
        options["min"] = min_value
    if max_value is not None:
        options["max"] = max_value
    if step is not None:
        options["step"] = step
    if tooltip:
        options["tooltip"] = tooltip
    return ("INT", options)


def _ui_float_input(
    default: float,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    step: Optional[float] = None,
    tooltip: Optional[str] = None,
) -> tuple:
    options: Dict[str, Any] = {"default": default}
    if min_value is not None:
        options["min"] = min_value
    if max_value is not None:
        options["max"] = max_value
    if step is not None:
        options["step"] = step
    if tooltip:
        options["tooltip"] = tooltip
    return ("FLOAT", options)


def _ui_bool_input(default: bool, *, tooltip: Optional[str] = None) -> tuple:
    options: Dict[str, Any] = {"default": default}
    if tooltip:
        options["tooltip"] = tooltip
    return ("BOOLEAN", options)


def _input_types_session_chat_simple() -> dict:
    available_models, mmproj_options = _get_available_models_and_mmprojs()
    return {
        "required": {
            "user_text": ("STRING", {"multiline": True, "default": "", "tooltip": "User message for this turn"}),
            "session_id": ("STRING", {"default": "default", "tooltip": "Session ID (maps to a history file). Same ID continues the chat."}),
            "model": _ui_model_input(available_models, f"GGUF model file in models/{_LLM_MODELS_DIR_NAME}/"),
            "mmproj": _ui_mmproj_input(mmproj_options, "Manual selection is recommended."),
            "history_dir": ("STRING", {"default": "", "tooltip": "Optional directory for history/caches. Empty uses output/llm_session_sessions/."}),
        },
        "optional": {
            "image": ("IMAGE", {"tooltip": "Optional image input for this turn only (never saved to history)"}),
            "config_path": ("STRING", {"default": "", "tooltip": "Optional override path to simple_defaults.json (advanced)."}),
        },
    }


def _input_types_dialogue_cycle_simple() -> dict:
    available_models, _ = _get_available_models_and_mmprojs()
    return {
        "required": {
            "initial_user_text": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Initial user message (sent to Model A only)."
            }),
            "system": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "Shared system prompt. Leave empty to use config/default."
            }),
            "systemA": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "System prompt override for Model A. Leave empty to use config/default."
            }),
            "systemB": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "System prompt override for Model B. Leave empty to use config/default."
            }),
            "session_id": ("STRING", {
                "default": "default",
                "tooltip": "Session id. A uses {id}_A, B uses {id}_B."
            }),
            "cycles": ("INT", {
                "default": 1,
                "min": 1,
                "max": 100,
                "step": 1,
                "tooltip": "Number of turns. 1 = A then B."
            }),
            "modelA": _ui_model_input(available_models, "GGUF model for role A"),
            "modelB": _ui_model_input(available_models, "GGUF model for role B"),
            "history_dir": ("STRING", {
                "default": "",
                "tooltip": "Directory to store histories, summaries, transcript, and session-scoped disk caches. Empty uses ComfyUI output."
            }),
        },
        "optional": {
            "config_path": ("STRING", {
                "default": "",
                "tooltip": "Optional path to a JSON config file. If empty, uses <node_dir>/config/simple_defaults.json."
            }),
            "force_text_only": ("BOOLEAN", {
                "default": False,
                "tooltip": "Force text-only mode for both models (disables mmproj auto-detect)."
            }),
            "reset_session": ("BOOLEAN", {
                "default": False,
                "tooltip": "If true, overwrite existing session history with a fresh session. Session disk cache is kept."
            }),
        }
    }


def _input_types_session_chat() -> dict:
    available_models, mmproj_options = _get_available_models_and_mmprojs()
    session_chat_defaults = _FULL_UI_SESSION_CHAT_DEFAULTS
    return {
        "required": {
            "user_text": ("STRING", {
                "multiline": True,
                "default": "",
                "tooltip": "User message for this turn"
            }),
            "session_id": ("STRING", {
                "default": "default",
                "tooltip": "Session ID (maps to a history file). Same ID continues the chat."
            }),
                "model": _ui_model_input(available_models, f"GGUF model file in models/{_LLM_MODELS_DIR_NAME}/"),
            "mmproj": _ui_mmproj_input(mmproj_options, "Manual selection is recommended."),
            "system_prompt": ("STRING", {
                "multiline": True,
                "default": str(session_chat_defaults["system_prompt"]),
                "tooltip": "System prompt (conversation policy). Saved into the history file."
            }),
            "max_tokens": _ui_int_input(
                int(session_chat_defaults["max_tokens"]),
                min_value=1,
                max_value=32768,
                tooltip="Maximum tokens to generate for this turn",
            ),
            "temperature": _ui_float_input(
                float(session_chat_defaults["temperature"]),
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                tooltip="Sampling temperature",
            ),
            "top_p": _ui_float_input(
                float(session_chat_defaults["top_p"]),
                min_value=0.05,
                max_value=1.0,
                step=0.01,
                tooltip="Nucleus sampling (top_p). Lower = safer/more conservative.",
            ),
            "n_gpu_layers": _ui_int_input(
                int(session_chat_defaults["n_gpu_layers"]),
                min_value=-1,
                max_value=200,
                step=1,
                tooltip="Number of layers to offload to GPU. 0=CPU. -1=all.",
            ),
            "n_ctx": _ui_int_input(
                int(session_chat_defaults["n_ctx"]),
                min_value=512,
                max_value=131072,
                step=256,
                tooltip="Context length (must be supported by the model)",
            ),
        },
        "optional": {
            "image": ("IMAGE", {
                "tooltip": "Optional image input for this turn only (never saved to history)"
            }),
            "persistent_cache": (_PERSISTENT_CACHE_OPTIONS, {
                "default": str(session_chat_defaults["persistent_cache"]),
                "tooltip": "Persistent cache backend. LlamaDiskCache stores cache data under a session-specific cache directory in output/llm_session_sessions/cache/."
            }),
            "runtime_cache": (_RUNTIME_CACHE_OPTIONS, {
                "default": str(session_chat_defaults["runtime_cache"]),
                "tooltip": "Runtime cache backend. KV_cache uses save_state/load_state, RAM/Trie use llama.cpp cache in memory."
            }),
            "log_level": (_LOG_LEVEL_OPTIONS, {
                "default": str(session_chat_defaults["log_level"]),
                "tooltip": "Console logging verbosity for LLM Session Chat."
            }),
            "suppress_backend_logs": _ui_bool_input(
                bool(session_chat_defaults["suppress_backend_logs"]),
                tooltip="Suppress backend stdout/stderr during generation.",
            ),
            "repeat_penalty": _ui_float_input(
                float(session_chat_defaults["repeat_penalty"]),
                min_value=1.0,
                max_value=2.0,
                step=0.01,
                tooltip="Repetition penalty to reduce looping outputs (especially on continue).",
            ),
            "repeat_last_n": _ui_int_input(
                int(session_chat_defaults["repeat_last_n"]),
                min_value=0,
                max_value=4096,
                tooltip="Apply repeat_penalty over the last N tokens. 0 disables.",
            ),
            "rewrite_continue": _ui_bool_input(
                bool(session_chat_defaults["rewrite_continue"]),
                tooltip="Rewrite inputs starting with 'continue' into an explicit continuation instruction to reduce repetition.",
            ),
            "max_turns": _ui_int_input(
                int(session_chat_defaults["max_turns"]),
                min_value=0,
                max_value=200,
                tooltip="Keep only the last N turns in live context. 0 means no prior turns.",
            ),
            "summarize_old_history": _ui_bool_input(
                bool(session_chat_defaults["summarize_old_history"]),
                tooltip="Summarize overflow turns into a rolling summary when turns exceed max_turns.",
            ),
            "summary_chunk_turns": _ui_int_input(
                int(session_chat_defaults["summary_chunk_turns"]),
                min_value=1,
                max_value=50,
                tooltip="Summarize overflow in chunks of this many turns (reduces summary frequency).",
            ),
            "max_tokens_summary": _ui_int_input(
                int(session_chat_defaults["max_tokens_summary"]),
                min_value=16,
                max_value=2048,
                tooltip="Max tokens for summary generation (kept small for speed).",
            ),
            "summary_max_chars": _ui_int_input(
                int(session_chat_defaults["summary_max_chars"]),
                min_value=200,
                max_value=20000,
                tooltip="If the rolling summary exceeds this size, it will be re-summarized to stay compact.",
            ),
            "dynamic_max_tokens": _ui_bool_input(
                bool(session_chat_defaults["dynamic_max_tokens"]),
                tooltip="Dynamically shrink max_tokens (and/or turns) when prompt would exceed n_ctx.",
            ),
            "min_generation_tokens": _ui_int_input(
                int(session_chat_defaults["min_generation_tokens"]),
                min_value=1,
                max_value=4096,
                tooltip="Minimum tokens to allow for generation when dynamic_max_tokens is enabled.",
            ),
            "safety_margin_tokens": _ui_int_input(
                int(session_chat_defaults["safety_margin_tokens"]),
                min_value=0,
                max_value=2048,
                tooltip="Token margin reserved to reduce the chance of exceeding n_ctx.",
            ),
            "history_dir": ("STRING", {
                "default": "",
                "tooltip": "Optional directory for history files and session-scoped disk caches. Empty uses output/llm_session_sessions/"
            }),
            "reset_session": _ui_bool_input(
                bool(session_chat_defaults["reset_session"]),
                tooltip="If true, overwrite existing session history file with a fresh session. Session disk cache is kept.",
            ),
            "stream_to_console": _ui_bool_input(
                bool(session_chat_defaults["stream_to_console"]),
                tooltip="Stream tokens to console while generating.",
            ),
            "enable_thinking": _ui_bool_input(
                bool(session_chat_defaults["enable_thinking"]),
                tooltip="Enable model thinking/reasoning output for supported chat formats.",
            ),
        }
    }


def _input_types_dialogue_cycle() -> dict:
    available_models, mmproj_options = _get_available_models_and_mmprojs()
    dialogue_cycle_defaults = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS
    return {
        "required": {
            "initial_user_text": ("STRING", {"multiline": True, "default": "", "tooltip": "Initial user message (sent to Model A only)"}),
            "session_id": ("STRING", {"default": "default", "tooltip": "Base session id. A uses {id}_A, B uses {id}_B, transcript uses {id}.txt"}),
            "cycles": ("INT", {"default": int(dialogue_cycle_defaults["cycles"]), "min": 1, "max": 50, "step": 1, "tooltip": "Number of round trips. 1 = A then B."}),

            "modelA": _ui_model_input(available_models, "GGUF model for role A"),
            "mmprojA": _ui_mmproj_input(mmproj_options, "mmproj for modelA"),

            "modelB": _ui_model_input(available_models, "GGUF model for role B"),
            "mmprojB": _ui_mmproj_input(mmproj_options, "mmproj for modelB"),

            "system_prompt": ("STRING", {"multiline": True, "default": str(dialogue_cycle_defaults["system_prompt"]), "tooltip": "Shared system prompt for both roles"}),
            "system_prompt_A": ("STRING", {"multiline": True, "default": str(dialogue_cycle_defaults["system_prompt_A"]), "tooltip": "Role-specific system prompt for model A (overrides shared prompt if set)"}),
            "system_prompt_B": ("STRING", {"multiline": True, "default": str(dialogue_cycle_defaults["system_prompt_B"]), "tooltip": "Role-specific system prompt for model B (overrides shared prompt if set)"}),

            "max_tokens": _ui_int_input(int(dialogue_cycle_defaults["max_tokens"]), min_value=1, max_value=32768),
            "temperature": _ui_float_input(float(dialogue_cycle_defaults["temperature"]), min_value=0.0, max_value=2.0, step=0.05),
            "top_p": _ui_float_input(float(dialogue_cycle_defaults["top_p"]), min_value=0.05, max_value=1.0, step=0.01),
            "n_gpu_layers": _ui_int_input(int(dialogue_cycle_defaults["n_gpu_layers"]), min_value=-1, max_value=200, step=1),
            "n_ctx": _ui_int_input(int(dialogue_cycle_defaults["n_ctx"]), min_value=512, max_value=131072, step=256),
        },
        "optional": {
            "max_turns": _ui_int_input(int(dialogue_cycle_defaults["max_turns"]), min_value=0, max_value=200),
            "summarize_old_history": _ui_bool_input(bool(dialogue_cycle_defaults["summarize_old_history"])),
            "summary_chunk_turns": _ui_int_input(int(dialogue_cycle_defaults["summary_chunk_turns"]), min_value=1, max_value=50),
            "max_tokens_summary": _ui_int_input(int(dialogue_cycle_defaults["max_tokens_summary"]), min_value=16, max_value=2048),
            "summary_max_chars": _ui_int_input(int(dialogue_cycle_defaults["summary_max_chars"]), min_value=200, max_value=20000),

            "dynamic_max_tokens": _ui_bool_input(bool(dialogue_cycle_defaults["dynamic_max_tokens"])),
            "min_generation_tokens": _ui_int_input(int(dialogue_cycle_defaults["min_generation_tokens"]), min_value=1, max_value=4096),
            "safety_margin_tokens": _ui_int_input(int(dialogue_cycle_defaults["safety_margin_tokens"]), min_value=0, max_value=2048),

            "persistent_cache": (_PERSISTENT_CACHE_OPTIONS, {"default": str(dialogue_cycle_defaults["persistent_cache"]), "tooltip": "Persistent cache backend. LlamaDiskCache stores cache data under separate cache directories for each session id."}),
            "runtime_cache": (_RUNTIME_CACHE_OPTIONS, {"default": str(dialogue_cycle_defaults["runtime_cache"])}),

            "repeat_penalty": _ui_float_input(float(dialogue_cycle_defaults["repeat_penalty"]), min_value=1.0, max_value=2.0, step=0.01),
            "repeat_last_n": _ui_int_input(int(dialogue_cycle_defaults["repeat_last_n"]), min_value=0, max_value=4096),
            "rewrite_continue": _ui_bool_input(bool(dialogue_cycle_defaults["rewrite_continue"])),
            "log_level": (_LOG_LEVEL_OPTIONS, {"default": str(dialogue_cycle_defaults["log_level"])}),
            "suppress_backend_logs": _ui_bool_input(bool(dialogue_cycle_defaults["suppress_backend_logs"])),

            "history_dir": ("STRING", {"default": "", "tooltip": "Optional history directory. Empty => output/llm_session_sessions/. Disk caches are also stored there, separated by session id."}),
            "reset_session": _ui_bool_input(
                bool(dialogue_cycle_defaults["reset_session"]),
                tooltip="If true, resets both {id}_A and {id}_B histories (transcript file is not deleted). Session disk caches are kept.",
            ),
            "stream_to_console": _ui_bool_input(bool(dialogue_cycle_defaults["stream_to_console"]), tooltip="Stream tokens to console while generating."),
            "enable_thinking": _ui_bool_input(
                bool(dialogue_cycle_defaults["enable_thinking"]),
                tooltip="Enable model thinking/reasoning output for supported chat formats.",
            ),
        }
    }


# ============================================================================
# ComfyUI Node Implementations
# ============================================================================

# =============================================================================
# LLM Session Chat Wiring
# =============================================================================

def _build_turn_execution_dependencies(
    runtime_container: Optional[RuntimeContainer] = None,
) -> TurnExecutionDependencies:
    container = _resolve_runtime_container(runtime_container)
    return {
        "llama_cpp_available": LLAMA_CPP_AVAILABLE,
        "llama_cpp_import_error": _LLAMA_CPP_IMPORT_ERROR,
        "is_no_models_placeholder": _is_no_models_placeholder,
        "get_llm_model_roots": _get_llm_model_roots,
        "resolve_model_and_mmproj": _resolve_model_and_mmproj,
        "mmproj_auto": _MMPROJ_AUTO,
        "mmproj_not_required": _MMPROJ_NOT_REQUIRED,
        "load_history": load_history,
        "clear_kv_state_for_session": lambda session_id: _clear_kv_state_for_session(
            session_id, runtime_container=container
        ),
        "rewrite_continue_prompt": rewrite_continue_prompt,
        "detect_history_language": _detect_history_language,
        "session_cache_root": _session_cache_root,
        "build_chat_messages": build_chat_messages,
        "build_text_chat_request": _build_text_chat_request,
        "build_kv_state_signature": build_kv_state_signature,
        "try_restore_kv_state": try_restore_kv_state,
        "is_state_data_mismatch_error": _is_state_data_mismatch_error,
        "saved_llama_state_size": _saved_llama_state_size,
        "current_llama_state_size": _current_llama_state_size,
        "kv_state_debug_info": _kv_state_debug_info,
        "get_context_turns": _get_context_turns,
        "mem_kv_state": container.mem_kv_state,
        "maybe_compact_summary": maybe_compact_summary,
        "cache_debug_label": _cache_debug_label,
        "run_generation_with_adaptive_retry": run_generation_with_adaptive_retry,
        "make_suppress_backend_logs": _make_suppress_backend_logs,
        "processing_interrupted": _processing_interrupted,
        "throw_if_processing_interrupted": _throw_if_processing_interrupted,
        "is_interrupt_error": _is_interrupt_error,
        "iter_chat_completion_robust": _iter_chat_completion_robust,
        "create_chat_completion_robust": _create_chat_completion_robust,
        "extract_stream_content": _extract_stream_content,
        "retry_kwargs_with_repeat_last_n_fallback": _retry_kwargs_with_repeat_last_n_fallback,
        "strip_reasoning_output": _strip_reasoning_output,
        "next_turn_id": _next_turn_id,
        "now_iso": _now_iso_jst,
        "maybe_summarize_history": maybe_summarize_history,
        "atomic_write_json": _atomic_write_json,
        "try_save_kv_state": try_save_kv_state,
    }


# =============================================================================
# LLM Dialogue Cycle Wiring
# =============================================================================

def _build_dialogue_cycle_dependencies(
    runtime_container: Optional[RuntimeContainer] = None,
) -> DialogueCycleDependencies:
    container = _resolve_runtime_container(runtime_container)
    return DialogueCycleDependencies(
        now_iso=_now_iso_jst,
        transcript_path=_transcript_path,
        append_transcript_lines=_append_transcript_lines,
        clear_kv_state_for_session=lambda session_id: _clear_kv_state_for_session(
            session_id, runtime_container=container
        ),
        get_or_create_model_manager=lambda role: _get_or_create_dialogue_cycle_model_manager(
            role, runtime_container=container
        ),
        unload_model=lambda manager: manager.unload_model(),
        chat_one_turn=_chat_one_turn,
    )


def _build_dialogue_cycle_node_execution_dependencies(
    runtime_container: Optional[RuntimeContainer] = None,
) -> DialogueCycleNodeExecutionDependencies:
    service = ChatTurnService()
    container = _resolve_runtime_container(runtime_container)
    return DialogueCycleNodeExecutionDependencies(
        build_common_turn_kwargs=_build_dialogue_cycle_common_turn_kwargs,
        build_dialogue_cycle_request=_build_dialogue_cycle_request,
        build_dialogue_cycle_dependencies=lambda: _build_dialogue_cycle_dependencies(
            runtime_container=container
        ),
        run_dialogue_cycle_with_dependencies=service.run_dialogue_cycle_with_dependencies,
    )


def _build_dialogue_cycle_request(
    *,
    initial_user_text: str,
    session_id: str,
    cycles: int,
    system_prompt: str,
    system_prompt_A: str,
    system_prompt_B: str,
    runtime_cache: str,
    stream_to_console: bool,
    reset_session: bool,
    history_dir: str,
    common_turn_kwargs: Dict[str, Any],
    model_a: str,
    mmproj_a: str,
    model_b: str,
    mmproj_b: str,
) -> DialogueCycleRequest:
    return DialogueCycleRequest(
        initial_user_text=initial_user_text,
        session_id=session_id,
        cycles=cycles,
        system_prompt=system_prompt,
        system_prompt_A=system_prompt_A,
        system_prompt_B=system_prompt_B,
        runtime_cache=runtime_cache,
        stream_to_console=bool(stream_to_console),
        reset_session=bool(reset_session),
        history_dir=history_dir,
        turn_kwargs_A={**common_turn_kwargs, "model": model_a, "mmproj": mmproj_a},
        turn_kwargs_B={**common_turn_kwargs, "model": model_b, "mmproj": mmproj_b},
    )


def _build_dialogue_cycle_common_turn_kwargs(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    max_turns: int,
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    runtime_cache: str,
    log_level: str,
    suppress_backend_logs: bool,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
    advanced_generation_kwargs: Optional[Dict[str, Any]],
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n_gpu_layers": n_gpu_layers,
        "tensor_split": tensor_split,
        "n_ctx": n_ctx,
        "max_turns": max_turns,
        "summarize_old_history": summarize_old_history,
        "summary_chunk_turns": summary_chunk_turns,
        "max_tokens_summary": max_tokens_summary,
        "summary_max_chars": summary_max_chars,
        "dynamic_max_tokens": dynamic_max_tokens,
        "min_generation_tokens": min_generation_tokens,
        "safety_margin_tokens": safety_margin_tokens,
        "persistent_cache": persistent_cache,
        "repeat_penalty": repeat_penalty,
        "repeat_last_n": repeat_last_n,
        "rewrite_continue": rewrite_continue,
        "runtime_cache": runtime_cache,
        "log_level": log_level,
        "suppress_backend_logs": suppress_backend_logs,
        "chat_handler_overrides": chat_handler_overrides,
        "text_chat_builder_overrides": text_chat_builder_overrides,
        "advanced_generation_kwargs": advanced_generation_kwargs,
        "advanced_summary_generation_kwargs": advanced_summary_generation_kwargs,
    }


# =============================================================================
# LLM Session Chat Runtime Helpers
# =============================================================================

def _build_session_chat_turn_kwargs(
    *,
    user_text: str,
    session_id: str,
    model: str,
    mmproj: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    image: Any,
    max_turns: int,
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    runtime_cache: str,
    log_level: str,
    suppress_backend_logs: bool,
    history_dir: str,
    reset_session: bool,
    stream_to_console: bool,
    model_manager: Optional[Any],
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
    advanced_generation_kwargs: Optional[Dict[str, Any]],
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "user_text": user_text,
        "session_id": session_id,
        "model": model,
        "mmproj": mmproj,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n_gpu_layers": n_gpu_layers,
        "tensor_split": tensor_split,
        "n_ctx": n_ctx,
        "image": image,
        "max_turns": max_turns,
        "summarize_old_history": summarize_old_history,
        "summary_chunk_turns": summary_chunk_turns,
        "max_tokens_summary": max_tokens_summary,
        "summary_max_chars": summary_max_chars,
        "dynamic_max_tokens": dynamic_max_tokens,
        "min_generation_tokens": min_generation_tokens,
        "safety_margin_tokens": safety_margin_tokens,
        "persistent_cache": persistent_cache,
        "repeat_penalty": repeat_penalty,
        "repeat_last_n": repeat_last_n,
        "rewrite_continue": rewrite_continue,
        "runtime_cache": runtime_cache,
        "log_level": log_level,
        "suppress_backend_logs": suppress_backend_logs,
        "history_dir": history_dir,
        "reset_session": reset_session,
        "stream_to_console": stream_to_console,
        "model_manager": model_manager,
        "chat_handler_overrides": chat_handler_overrides,
        "text_chat_builder_overrides": text_chat_builder_overrides,
        "advanced_generation_kwargs": advanced_generation_kwargs,
        "advanced_summary_generation_kwargs": advanced_summary_generation_kwargs,
    }


def _resolve_model_path_for_session_chat(model: str) -> str:
    roots = _get_llm_model_roots()
    return _resolve_llm_relpath(model, roots=roots)


def _log_session_chat_total(start_time: float, status: str) -> None:
    dt = time.perf_counter() - start_time
    print(f"[LLM Session Chat] {status} in {dt:.2f} seconds")


def _session_chat_error_return(start_time: float, message: Optional[str] = None) -> tuple:
    if message:
        print(message)
    _log_session_chat_total(start_time, "Finished (error)")
    return ("",)


def _resolve_valid_session_chat_model_path(model: str, start_time: float) -> Optional[str]:
    if _is_no_models_placeholder(model):
        _session_chat_error_return(
            start_time,
            f"[LLM Session Chat] Error: No GGUF models found in models/{_LLM_MODELS_DIR_NAME}/",
        )
        return None

    model_path = _resolve_model_path_for_session_chat(model)
    if not os.path.exists(model_path):
        _session_chat_error_return(
            start_time,
            f"[LLM Session Chat] Error: Model not found: {model_path}",
        )
        return None
    return model_path


def _build_session_chat_node_execution_dependencies(
    runtime_container: Optional[RuntimeContainer] = None,
) -> SessionChatNodeExecutionDependencies:
    container = _resolve_runtime_container(runtime_container)
    return SessionChatNodeExecutionDependencies(
        require_llama_cpp_available=_require_llama_cpp_available,
        resolve_valid_model_path=_resolve_valid_session_chat_model_path,
        get_or_create_model_manager=lambda: _get_or_create_model_manager(runtime_container=container),
        execute_session_chat_turn=_execute_session_chat_turn,
        session_chat_error_return=_session_chat_error_return,
        log_session_chat_total=_log_session_chat_total,
    )


def _build_session_chat_node_execution_request(
    *,
    user_text: str,
    session_id: str,
    model: str,
    mmproj: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    image: Any,
    max_turns: int,
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    runtime_cache: str,
    log_level: str,
    suppress_backend_logs: bool,
    history_dir: str,
    reset_session: bool,
    stream_to_console: bool,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
    advanced_generation_kwargs: Optional[Dict[str, Any]],
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]],
) -> SessionChatNodeExecutionRequest:
    return SessionChatNodeExecutionRequest(
        model=model,
        turn_kwargs=_build_session_chat_turn_kwargs(
            user_text=user_text,
            session_id=session_id,
            model=model,
            mmproj=mmproj,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            n_ctx=n_ctx,
            image=image,
            max_turns=max_turns,
            summarize_old_history=summarize_old_history,
            summary_chunk_turns=summary_chunk_turns,
            max_tokens_summary=max_tokens_summary,
            summary_max_chars=summary_max_chars,
            dynamic_max_tokens=dynamic_max_tokens,
            min_generation_tokens=min_generation_tokens,
            safety_margin_tokens=safety_margin_tokens,
            persistent_cache=persistent_cache,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            rewrite_continue=rewrite_continue,
            runtime_cache=runtime_cache,
            log_level=log_level,
            suppress_backend_logs=suppress_backend_logs,
            history_dir=history_dir,
            reset_session=reset_session,
            stream_to_console=stream_to_console,
            model_manager=None,
            chat_handler_overrides=chat_handler_overrides,
            text_chat_builder_overrides=text_chat_builder_overrides,
            advanced_generation_kwargs=advanced_generation_kwargs,
            advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
        ),
    )


def _require_llama_cpp_available() -> None:
    if not LLAMA_CPP_AVAILABLE:
        msg = "llama_cpp is not available"
        if "_LLAMA_CPP_IMPORT_ERROR" in globals():
            msg += f" ({_LLAMA_CPP_IMPORT_ERROR})"
        raise RuntimeError(msg)


def _get_or_create_model_manager(
    model_manager: Optional["GGUFModelManager"] = None,
    runtime_container: Optional[RuntimeContainer] = None,
) -> "GGUFModelManager":
    if model_manager is not None:
        return model_manager

    container = _resolve_runtime_container(runtime_container)
    existing = container.model_manager
    if existing is not None:
        return existing

    container.model_manager = GGUFModelManager()
    return container.model_manager


def _execute_session_chat_turn(
    *,
    user_text: str,
    session_id: str,
    model: str,
    mmproj: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    image: Any,
    max_turns: Optional[int],
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    runtime_cache: str,
    log_level: str,
    suppress_backend_logs: bool,
    history_dir: str,
    reset_session: bool,
    stream_to_console: bool,
    model_manager: Optional[Any],
    runtime_container: Optional[RuntimeContainer] = None,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    advanced_generation_kwargs: Optional[Dict[str, Any]] = None,
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None,
) -> TurnExecutionResult:
    service = TurnExecutionService()
    return service.execute_session_chat_turn(
        user_text=user_text,
        session_id=session_id,
        model=model,
        mmproj=mmproj,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        n_ctx=n_ctx,
        image=image,
        max_turns=max_turns,
        summarize_old_history=summarize_old_history,
        summary_chunk_turns=summary_chunk_turns,
        max_tokens_summary=max_tokens_summary,
        summary_max_chars=summary_max_chars,
        dynamic_max_tokens=dynamic_max_tokens,
        min_generation_tokens=min_generation_tokens,
        safety_margin_tokens=safety_margin_tokens,
        persistent_cache=persistent_cache,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        rewrite_continue=rewrite_continue,
        runtime_cache=runtime_cache,
        log_level=log_level,
        suppress_backend_logs=suppress_backend_logs,
        history_dir=history_dir,
        reset_session=reset_session,
        stream_to_console=stream_to_console,
        model_manager=model_manager,
        chat_handler_overrides=chat_handler_overrides,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
        advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
        dependencies=_build_turn_execution_dependencies(runtime_container=runtime_container),
    )


def _execute_dialogue_cycle_turn(
    *,
    user_text: str,
    session_id: str,
    model: str,
    mmproj: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    max_turns: Optional[int],
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    runtime_cache: str,
    log_level: str,
    suppress_backend_logs: bool,
    history_dir: str,
    reset_session: bool,
    stream_to_console: bool,
    model_manager: Optional[Any],
    runtime_container: Optional[RuntimeContainer] = None,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    advanced_generation_kwargs: Optional[Dict[str, Any]] = None,
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None,
) -> TurnExecutionResult:
    service = TurnExecutionService()
    return service.execute_dialogue_cycle_turn(
        user_text=user_text,
        session_id=session_id,
        model=model,
        mmproj=mmproj,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        n_ctx=n_ctx,
        image=None,
        max_turns=max_turns,
        summarize_old_history=summarize_old_history,
        summary_chunk_turns=summary_chunk_turns,
        max_tokens_summary=max_tokens_summary,
        summary_max_chars=summary_max_chars,
        dynamic_max_tokens=dynamic_max_tokens,
        min_generation_tokens=min_generation_tokens,
        safety_margin_tokens=safety_margin_tokens,
        persistent_cache=persistent_cache,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        rewrite_continue=rewrite_continue,
        runtime_cache=runtime_cache,
        log_level=log_level,
        suppress_backend_logs=suppress_backend_logs,
        history_dir=history_dir,
        reset_session=reset_session,
        stream_to_console=stream_to_console,
        model_manager=model_manager,
        chat_handler_overrides=chat_handler_overrides,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
        advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
        dependencies=_build_turn_execution_dependencies(runtime_container=runtime_container),
    )


def _run_session_chat_from_inputs(
    *,
    user_text: str,
    session_id: str,
    model: str,
    mmproj: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    image: Any,
    max_turns: int,
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    runtime_cache: str,
    log_level: str,
    suppress_backend_logs: bool,
    history_dir: str,
    reset_session: bool,
    stream_to_console: bool,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
    advanced_generation_kwargs: Optional[Dict[str, Any]],
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]],
) -> tuple:
    request = _build_session_chat_node_execution_request(
        user_text=user_text,
        session_id=session_id,
        model=model,
        mmproj=mmproj,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        n_ctx=n_ctx,
        image=image,
        max_turns=max_turns,
        summarize_old_history=summarize_old_history,
        summary_chunk_turns=summary_chunk_turns,
        max_tokens_summary=max_tokens_summary,
        summary_max_chars=summary_max_chars,
        dynamic_max_tokens=dynamic_max_tokens,
        min_generation_tokens=min_generation_tokens,
        safety_margin_tokens=safety_margin_tokens,
        persistent_cache=persistent_cache,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        rewrite_continue=rewrite_continue,
        runtime_cache=runtime_cache,
        log_level=log_level,
        suppress_backend_logs=suppress_backend_logs,
        history_dir=history_dir,
        reset_session=reset_session,
        stream_to_console=stream_to_console,
        chat_handler_overrides=chat_handler_overrides,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
        advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
    )
    dependencies = _build_session_chat_node_execution_dependencies()
    service = SessionChatNodeExecutionService()
    return service.run(request=request, dependencies=dependencies)


# =============================================================================
# LLM Session Chat
# =============================================================================

class LLMSessionChatNode:
    """
    LLM Session Chat - Local GGUF vision language models with file-based chat history.

    Phase 1 design:
    - History is stored/updated as a JSON file under output/llm_session_sessions/
    - History is NOT shown in ComfyUI UI (only assistant response is returned)
    - Images are used ONLY for the current turn and never persisted
    - Supports max_turns / summarize_old_history / system_prompt
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _input_types_session_chat()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("assistant_text",)
    FUNCTION = "chat_stream"
    CATEGORY = _LLM_SESSION_CATEGORY
    DESCRIPTION = "Persistent multi-turn chat with local GGUF models using file-based history."

    def chat_stream(self,
             user_text: str,
             session_id: str,
             model: str,
             mmproj: str,
             system_prompt: str,
             max_tokens: int,
             temperature: float,
             top_p: float,
             n_gpu_layers: int,
             n_ctx: int,
             image=None,
             max_turns: int = _FULL_UI_SESSION_CHAT_DEFAULTS["max_turns"],
             summarize_old_history: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["summarize_old_history"],
             summary_chunk_turns: int = _FULL_UI_SESSION_CHAT_DEFAULTS["summary_chunk_turns"],
             max_tokens_summary: int = _FULL_UI_SESSION_CHAT_DEFAULTS["max_tokens_summary"],
             summary_max_chars: int = _FULL_UI_SESSION_CHAT_DEFAULTS["summary_max_chars"],
             dynamic_max_tokens: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["dynamic_max_tokens"],
             min_generation_tokens: int = _FULL_UI_SESSION_CHAT_DEFAULTS["min_generation_tokens"],
             safety_margin_tokens: int = _FULL_UI_SESSION_CHAT_DEFAULTS["safety_margin_tokens"],
             persistent_cache: str = _FULL_UI_SESSION_CHAT_DEFAULTS["persistent_cache"],
             repeat_penalty: float = _FULL_UI_SESSION_CHAT_DEFAULTS["repeat_penalty"],
             repeat_last_n: int = _FULL_UI_SESSION_CHAT_DEFAULTS["repeat_last_n"],
             rewrite_continue: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["rewrite_continue"],
             runtime_cache: str = _FULL_UI_SESSION_CHAT_DEFAULTS["runtime_cache"],
             log_level: str = _FULL_UI_SESSION_CHAT_DEFAULTS["log_level"],
             suppress_backend_logs: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["suppress_backend_logs"],
             history_dir: str = "",
             reset_session: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["reset_session"],
             stream_to_console: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["stream_to_console"],
             enable_thinking: bool = _FULL_UI_SESSION_CHAT_DEFAULTS["enable_thinking"],
             tensor_split: Optional[List[float]] = None,
             chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
             text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
             advanced_generation_kwargs: Optional[Dict[str, Any]] = None,
             advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None) -> tuple:
        chat_handler_overrides = _merge_enable_thinking_chat_handler_overrides(
            chat_handler_overrides,
            enable_thinking,
        )
        text_chat_builder_overrides = _merge_enable_thinking_text_chat_builder_overrides(
            text_chat_builder_overrides,
            enable_thinking,
        )
        return _run_session_chat_from_inputs(
            user_text=user_text,
            session_id=session_id,
            model=model,
            mmproj=mmproj,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            n_ctx=n_ctx,
            image=image,
            max_turns=max_turns,
            summarize_old_history=summarize_old_history,
            summary_chunk_turns=summary_chunk_turns,
            max_tokens_summary=max_tokens_summary,
            summary_max_chars=summary_max_chars,
            dynamic_max_tokens=dynamic_max_tokens,
            min_generation_tokens=min_generation_tokens,
            safety_margin_tokens=safety_margin_tokens,
            persistent_cache=persistent_cache,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            rewrite_continue=rewrite_continue,
            runtime_cache=runtime_cache,
            log_level=log_level,
            suppress_backend_logs=suppress_backend_logs,
            history_dir=history_dir,
            reset_session=reset_session,
            stream_to_console=stream_to_console,
            chat_handler_overrides=chat_handler_overrides,
            text_chat_builder_overrides=text_chat_builder_overrides,
            advanced_generation_kwargs=advanced_generation_kwargs,
            advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
        )


# =============================================================================
# LLM Dialogue Cycle Runtime Helpers
# =============================================================================

def _chat_one_turn(
    *,
    user_text: str,
    session_id: str,
    model: str,
    mmproj: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    max_turns: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["max_turns"],
    summarize_old_history: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["summarize_old_history"],
    summary_chunk_turns: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["summary_chunk_turns"],
    max_tokens_summary: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["max_tokens_summary"],
    summary_max_chars: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["summary_max_chars"],
    dynamic_max_tokens: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["dynamic_max_tokens"],
    min_generation_tokens: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["min_generation_tokens"],
    safety_margin_tokens: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["safety_margin_tokens"],
    persistent_cache: str = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["persistent_cache"],
    repeat_penalty: float = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["repeat_penalty"],
    repeat_last_n: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["repeat_last_n"],
    rewrite_continue: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["rewrite_continue"],
    runtime_cache: str = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["runtime_cache"],
    log_level: str = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["log_level"],
    suppress_backend_logs: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["suppress_backend_logs"],
    history_dir: str = "",
    reset_session: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["reset_session"],
    stream_to_console: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["stream_to_console"],
    model_manager: Optional[GGUFModelManager] = None,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    advanced_generation_kwargs: Optional[Dict[str, Any]] = None,
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    One chat turn using the same history/summary logic as LLMSessionChatNode,
    but returns assistant_text only.
    """
    _require_llama_cpp_available()

    mgr = _get_or_create_model_manager(model_manager)
    result = _execute_dialogue_cycle_turn(
        user_text=user_text,
        session_id=session_id,
        model=model,
        mmproj=mmproj,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        n_ctx=n_ctx,
        max_turns=max_turns,
        summarize_old_history=summarize_old_history,
        summary_chunk_turns=summary_chunk_turns,
        max_tokens_summary=max_tokens_summary,
        summary_max_chars=summary_max_chars,
        dynamic_max_tokens=dynamic_max_tokens,
        min_generation_tokens=min_generation_tokens,
        safety_margin_tokens=safety_margin_tokens,
        persistent_cache=persistent_cache,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        rewrite_continue=rewrite_continue,
        runtime_cache=runtime_cache,
        log_level=log_level,
        suppress_backend_logs=suppress_backend_logs,
        history_dir=history_dir,
        reset_session=reset_session,
        stream_to_console=stream_to_console,
        model_manager=mgr,
        chat_handler_overrides=chat_handler_overrides,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
        advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
    )
    if not result.generation_succeeded:
        if log_level == "debug" and result.error is not None:
            print(f"[LLM Dialogue Cycle] generation failed: {result.error}")
        return ""

    if not result.persistence_succeeded:
        logger = get_module_logger("LLM Dialogue Cycle")
        detail = f": {result.persistence_error}" if result.persistence_error is not None else ""
        logger(
            f"[LLM Dialogue Cycle] Warning: response generated but history was not saved{detail}",
            LOG_LEVEL_MINIMAL,
        )

    return result.assistant_text


def _run_dialogue_cycle_from_inputs(
    *,
    initial_user_text: str,
    session_id: str,
    cycles: int,
    modelA: str,
    mmprojA: str,
    modelB: str,
    mmprojB: str,
    system_prompt: str,
    system_prompt_A: str,
    system_prompt_B: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    max_turns: int,
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    runtime_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    log_level: str,
    suppress_backend_logs: bool,
    history_dir: str,
    reset_session: bool,
    stream_to_console: bool,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
    advanced_generation_kwargs: Optional[Dict[str, Any]],
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]],
) -> str:
    request = _build_dialogue_cycle_node_execution_request(
        initial_user_text=initial_user_text,
        session_id=session_id,
        cycles=cycles,
        modelA=modelA,
        mmprojA=mmprojA,
        modelB=modelB,
        mmprojB=mmprojB,
        system_prompt=system_prompt,
        system_prompt_A=system_prompt_A,
        system_prompt_B=system_prompt_B,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        n_ctx=n_ctx,
        max_turns=max_turns,
        summarize_old_history=summarize_old_history,
        summary_chunk_turns=summary_chunk_turns,
        max_tokens_summary=max_tokens_summary,
        summary_max_chars=summary_max_chars,
        dynamic_max_tokens=dynamic_max_tokens,
        min_generation_tokens=min_generation_tokens,
        safety_margin_tokens=safety_margin_tokens,
        persistent_cache=persistent_cache,
        runtime_cache=runtime_cache,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        rewrite_continue=rewrite_continue,
        log_level=log_level,
        suppress_backend_logs=suppress_backend_logs,
        stream_to_console=stream_to_console,
        reset_session=reset_session,
        history_dir=history_dir,
        chat_handler_overrides=chat_handler_overrides,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
        advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
    )
    dependencies = _build_dialogue_cycle_node_execution_dependencies(
        runtime_container=_resolve_runtime_container()
    )
    service = DialogueCycleNodeExecutionService()
    return service.run(
        request=request,
        dependencies=dependencies,
    )


def _build_dialogue_cycle_node_execution_request(
    *,
    initial_user_text: str,
    session_id: str,
    cycles: int,
    modelA: str,
    mmprojA: str,
    modelB: str,
    mmprojB: str,
    system_prompt: str,
    system_prompt_A: str,
    system_prompt_B: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n_gpu_layers: int,
    tensor_split: Optional[List[float]],
    n_ctx: int,
    max_turns: int,
    summarize_old_history: bool,
    summary_chunk_turns: int,
    max_tokens_summary: int,
    summary_max_chars: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    persistent_cache: str,
    runtime_cache: str,
    repeat_penalty: float,
    repeat_last_n: int,
    rewrite_continue: bool,
    log_level: str,
    suppress_backend_logs: bool,
    history_dir: str,
    reset_session: bool,
    stream_to_console: bool,
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
    advanced_generation_kwargs: Optional[Dict[str, Any]],
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]],
) -> DialogueCycleNodeExecutionRequest:
    return DialogueCycleNodeExecutionRequest(
        initial_user_text=initial_user_text,
        session_id=session_id,
        cycles=cycles,
        modelA=modelA,
        mmprojA=mmprojA,
        modelB=modelB,
        mmprojB=mmprojB,
        system_prompt=system_prompt,
        system_prompt_A=system_prompt_A,
        system_prompt_B=system_prompt_B,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n_gpu_layers=n_gpu_layers,
        tensor_split=tensor_split,
        n_ctx=n_ctx,
        max_turns=max_turns,
        summarize_old_history=summarize_old_history,
        summary_chunk_turns=summary_chunk_turns,
        max_tokens_summary=max_tokens_summary,
        summary_max_chars=summary_max_chars,
        dynamic_max_tokens=dynamic_max_tokens,
        min_generation_tokens=min_generation_tokens,
        safety_margin_tokens=safety_margin_tokens,
        persistent_cache=persistent_cache,
        runtime_cache=runtime_cache,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        rewrite_continue=rewrite_continue,
        log_level=log_level,
        suppress_backend_logs=suppress_backend_logs,
        stream_to_console=stream_to_console,
        reset_session=reset_session,
        history_dir=history_dir,
        chat_handler_overrides=chat_handler_overrides,
        text_chat_builder_overrides=text_chat_builder_overrides,
        advanced_generation_kwargs=advanced_generation_kwargs,
        advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
    )


# =============================================================================
# LLM Dialogue Cycle
# =============================================================================

class LLMDialogueCycleNode:
    """
    LLM Dialogue Cycle
    - Run A->B->A->B... inside one node.
    - Separate histories: {session_id}_A and {session_id}_B
    - Save full transcript to {session_id}.txt (append)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _input_types_dialogue_cycle()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript_text",)
    FUNCTION = "chat_cycle"
    CATEGORY = _LLM_SESSION_CATEGORY
    DESCRIPTION = "Run a chat cycle between two local GGUF models inside one node (A<->B), saving transcript."

    def chat_cycle(
        self,
        initial_user_text: str,
        session_id: str,
        cycles: int,
        modelA: str,
        mmprojA: str,
        modelB: str,
        mmprojB: str,
        system_prompt: str,
        system_prompt_A: str,
        system_prompt_B: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        n_gpu_layers: int,
        n_ctx: int,
        max_turns: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["max_turns"],
        summarize_old_history: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["summarize_old_history"],
        summary_chunk_turns: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["summary_chunk_turns"],
        max_tokens_summary: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["max_tokens_summary"],
        summary_max_chars: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["summary_max_chars"],
        dynamic_max_tokens: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["dynamic_max_tokens"],
        min_generation_tokens: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["min_generation_tokens"],
        safety_margin_tokens: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["safety_margin_tokens"],
        persistent_cache: str = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["persistent_cache"],
        runtime_cache: str = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["runtime_cache"],
        repeat_penalty: float = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["repeat_penalty"],
        repeat_last_n: int = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["repeat_last_n"],
        rewrite_continue: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["rewrite_continue"],
        log_level: str = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["log_level"],
        suppress_backend_logs: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["suppress_backend_logs"],
        history_dir: str = "",
        reset_session: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["reset_session"],
        stream_to_console: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["stream_to_console"],
        enable_thinking: bool = _FULL_UI_DIALOGUE_CYCLE_DEFAULTS["enable_thinking"],
        tensor_split: Optional[List[float]] = None,
        chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        advanced_generation_kwargs: Optional[Dict[str, Any]] = None,
        advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        chat_handler_overrides = _merge_enable_thinking_chat_handler_overrides(
            chat_handler_overrides,
            enable_thinking,
        )
        text_chat_builder_overrides = _merge_enable_thinking_text_chat_builder_overrides(
            text_chat_builder_overrides,
            enable_thinking,
        )
        transcript_text = _run_dialogue_cycle_from_inputs(
            initial_user_text=initial_user_text,
            session_id=session_id,
            cycles=cycles,
            modelA=modelA,
            mmprojA=mmprojA,
            modelB=modelB,
            mmprojB=mmprojB,
            system_prompt=system_prompt,
            system_prompt_A=system_prompt_A,
            system_prompt_B=system_prompt_B,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            n_ctx=n_ctx,
            max_turns=max_turns,
            summarize_old_history=summarize_old_history,
            summary_chunk_turns=summary_chunk_turns,
            max_tokens_summary=max_tokens_summary,
            summary_max_chars=summary_max_chars,
            dynamic_max_tokens=dynamic_max_tokens,
            min_generation_tokens=min_generation_tokens,
            safety_margin_tokens=safety_margin_tokens,
            persistent_cache=persistent_cache,
            runtime_cache=runtime_cache,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            rewrite_continue=rewrite_continue,
            log_level=log_level,
            suppress_backend_logs=suppress_backend_logs,
            history_dir=history_dir,
            reset_session=reset_session,
            stream_to_console=stream_to_console,
            chat_handler_overrides=chat_handler_overrides,
            text_chat_builder_overrides=text_chat_builder_overrides,
            advanced_generation_kwargs=advanced_generation_kwargs,
            advanced_summary_generation_kwargs=advanced_summary_generation_kwargs,
        )
        return (transcript_text,)


# =============================================================================
# Simple Wrappers
# =============================================================================

# =============================================================================
# LLM Session Chat (Simple)
# =============================================================================

class LLMSessionChatSimpleNode:
    """LLM Session Chat (Simple)

    Minimal UI inputs with config-driven defaults.
    Defaults are loaded from config/simple_defaults.json when present.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _input_types_session_chat_simple()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("assistant_text",)
    FUNCTION = "chat_stream"
    CATEGORY = _LLM_SESSION_CATEGORY
    DESCRIPTION = "Minimal entry node for persistent local GGUF chat sessions (config-driven defaults)."

    def chat_stream(
        self,
        user_text: str,
        session_id: str,
        model: str,
        mmproj: str,
        history_dir: str,
        image=None,
        config_path: str = _SIMPLE_WRAPPER_DEFAULTS["config_path"],
        stream_to_console: bool = _SIMPLE_WRAPPER_DEFAULTS["stream_to_console"],
    ) -> tuple:
        defaults, chat_handler_overrides, text_chat_builder_overrides = _load_simple_defaults_bundle(
            config_path=config_path
        )
        chat_kwargs = _build_session_chat_simple_chat_kwargs(
            defaults=defaults,
            history_dir=history_dir,
            chat_handler_overrides=chat_handler_overrides,
            text_chat_builder_overrides=text_chat_builder_overrides,
        )

        # Delegate to the full node implementation
        node = LLMSessionChatNode()
        return node.chat_stream(
            user_text=user_text,
            session_id=session_id,
            model=model,
            mmproj=mmproj,
            image=image,
            **chat_kwargs,
        )


# =============================================================================
# LLM Dialogue Cycle (Simple)
# =============================================================================

class LLMDialogueCycleSimpleNode:
    """
    LLM Dialogue Cycle (Simple)

    Minimal-input wrapper around LLM Dialogue Cycle using a JSON config file for defaults.
    - Keeps UI simple (few parameters)
    - Users can override defaults via config/simple_defaults.json
    - Safe fallback to built-in defaults if the config is missing or invalid
    """

    @classmethod
    def INPUT_TYPES(cls):
        return _input_types_dialogue_cycle_simple()

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcript_text",)
    FUNCTION = "chat_cycle_simple"
    CATEGORY = _LLM_SESSION_CATEGORY
    DESCRIPTION = "Run a two-model dialogue cycle with minimal UI; defaults are configurable via JSON."

    def chat_cycle_simple(
        self,
        initial_user_text: str,
        system: str,
        systemA: str,
        systemB: str,
        session_id: str,
        cycles: int,
        modelA: str,
        modelB: str,
        history_dir: str,
        config_path: str = _SIMPLE_WRAPPER_DEFAULTS["config_path"],
        force_text_only: bool = _SIMPLE_WRAPPER_DEFAULTS["force_text_only"],
        reset_session: bool = _SIMPLE_WRAPPER_DEFAULTS["reset_session"],
    ) -> tuple:
        # Resolve models list placeholder
        if _is_no_models_placeholder(modelA) or _is_no_models_placeholder(modelB):
            return ("",)

        # Load defaults from config (or fallback)
        defaults, chat_handler_overrides, text_chat_builder_overrides = _load_simple_defaults_bundle(
            config_path or ""
        )

        # System prompts (shared + optional overrides)
        # UI fields take priority when non-empty; otherwise fall back to config/defaults.
        system_prompt, system_prompt_A, system_prompt_B = _resolve_simple_system_prompts(
            defaults,
            system,
            systemA,
            systemB,
        )
        chat_kwargs = _build_dialogue_cycle_simple_chat_kwargs(
            defaults=defaults,
            force_text_only=force_text_only,
            history_dir=history_dir,
            reset_session=reset_session,
            chat_handler_overrides=chat_handler_overrides,
            text_chat_builder_overrides=text_chat_builder_overrides,
        )

        node = LLMDialogueCycleNode()
        return node.chat_cycle(
            initial_user_text=initial_user_text,
            session_id=session_id,
            cycles=int(cycles),
            modelA=modelA,
            modelB=modelB,
            system_prompt=system_prompt,
            system_prompt_A=system_prompt_A,
            system_prompt_B=system_prompt_B,
            **chat_kwargs,
        )


class UnloadLLMModelNode:
    """Output node that unloads the current LLM model to free VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unload_now": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Toggle true and queue this node to unload the current LLM model.",
                    },
                ),
            },
            "optional": {
                "trigger": ("*",),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "unload_model"
    CATEGORY = "LLM/Session"
    OUTPUT_NODE = True

    def unload_model(self, unload_now: bool, trigger: Any = None):
        if bool(unload_now):
            _unload_runtime_container_managers(runtime_container=_resolve_runtime_container())
        return (trigger,)


# ============================================================================
# ComfyUI Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LLMSessionChatSimpleNode": LLMSessionChatSimpleNode,
    "LLMDialogueCycleSimpleNode": LLMDialogueCycleSimpleNode,
    "LLMSessionChatNode": LLMSessionChatNode,
    "LLMDialogueCycleNode": LLMDialogueCycleNode,
    "UnloadLLMModelNode": UnloadLLMModelNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMSessionChatSimpleNode": "LLM Session Chat (Simple)",
    "LLMDialogueCycleSimpleNode": "LLM Dialogue Cycle (Simple)",
    "LLMSessionChatNode": "LLM Session Chat",
    "LLMDialogueCycleNode": "LLM Dialogue Cycle",
    "UnloadLLMModelNode": "Unload LLM Model",
}


# ============================================================================
# Cleanup on module unload
# ============================================================================

def cleanup():
    """Cleanup on module unload"""
    container = _runtime_container
    if container is None:
        return
    _unload_runtime_container_managers(runtime_container=container)

import atexit
atexit.register(cleanup)
