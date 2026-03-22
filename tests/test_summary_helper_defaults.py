from __future__ import annotations

import importlib
import inspect
import sys
import types

from core.defaults import FULL_UI_DEFAULTS, SUMMARY_HELPER_DEFAULTS


def _load_nodes_module(monkeypatch):
    fake_folder_paths = types.SimpleNamespace(
        models_dir="C:/models",
        get_folder_paths=lambda _key: [],
        get_filename_list=lambda _key: [],
        get_output_directory=lambda: "C:/output",
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    sys.modules.pop("llm_session_nodes", None)
    return importlib.import_module("llm_session_nodes")


def test_summary_helper_function_defaults_are_centralized(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    compact_sig = inspect.signature(module.maybe_compact_summary)
    summarize_sig = inspect.signature(module.maybe_summarize_history)
    summarize_one_sig = inspect.signature(module._summarize_with_model)

    assert compact_sig.parameters["summary_max_chars"].default == FULL_UI_DEFAULTS["session_chat"]["summary_max_chars"]
    assert compact_sig.parameters["max_tokens_summary"].default == FULL_UI_DEFAULTS["session_chat"]["max_tokens_summary"]
    assert compact_sig.parameters["temperature"].default == SUMMARY_HELPER_DEFAULTS["temperature"]
    assert compact_sig.parameters["suppress_logs"].default == SUMMARY_HELPER_DEFAULTS["suppress_logs"]

    assert summarize_sig.parameters["summarize_old_history"].default == FULL_UI_DEFAULTS["session_chat"]["summarize_old_history"]
    assert summarize_sig.parameters["summary_chunk_turns"].default == FULL_UI_DEFAULTS["session_chat"]["summary_chunk_turns"]
    assert summarize_sig.parameters["max_tokens_summary"].default == FULL_UI_DEFAULTS["session_chat"]["max_tokens_summary"]
    assert summarize_sig.parameters["summary_max_chars"].default == FULL_UI_DEFAULTS["session_chat"]["summary_max_chars"]
    assert summarize_sig.parameters["temperature"].default == SUMMARY_HELPER_DEFAULTS["temperature"]
    assert summarize_sig.parameters["suppress_logs"].default == SUMMARY_HELPER_DEFAULTS["suppress_logs"]

    assert summarize_one_sig.parameters["suppress_logs"].default == SUMMARY_HELPER_DEFAULTS["suppress_logs"]
