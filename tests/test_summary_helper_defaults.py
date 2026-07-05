from __future__ import annotations

import inspect

from core.defaults import FULL_UI_DEFAULTS, SUMMARY_HELPER_DEFAULTS



def test_summary_helper_function_defaults_are_centralized(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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
