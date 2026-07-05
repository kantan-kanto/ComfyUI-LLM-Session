from __future__ import annotations

import inspect

from core.defaults import SIMPLE_WRAPPER_DEFAULTS



def test_simple_wrapper_signature_defaults(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    chat_sig = inspect.signature(module.LLMSessionChatSimpleNode.chat_stream)
    cycle_sig = inspect.signature(module.LLMDialogueCycleSimpleNode.chat_cycle_simple)
    load_sig = inspect.signature(module.load_history)

    assert chat_sig.parameters["config_path"].default == SIMPLE_WRAPPER_DEFAULTS["config_path"]
    assert chat_sig.parameters["stream_to_console"].default == SIMPLE_WRAPPER_DEFAULTS["stream_to_console"]

    assert cycle_sig.parameters["config_path"].default == SIMPLE_WRAPPER_DEFAULTS["config_path"]
    assert cycle_sig.parameters["force_text_only"].default == SIMPLE_WRAPPER_DEFAULTS["force_text_only"]
    assert cycle_sig.parameters["reset_session"].default == SIMPLE_WRAPPER_DEFAULTS["reset_session"]

    assert load_sig.parameters["reset_session"].default == SIMPLE_WRAPPER_DEFAULTS["reset_session"]
