from __future__ import annotations

import importlib
import inspect
import sys
import types

from core.defaults import SIMPLE_WRAPPER_DEFAULTS


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


def test_simple_wrapper_signature_defaults(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    chat_sig = inspect.signature(module.LLMSessionChatSimpleNode.chat_stream)
    cycle_sig = inspect.signature(module.LLMDialogueCycleSimpleNode.chat_cycle_simple)
    load_sig = inspect.signature(module.load_history)

    assert chat_sig.parameters["config_path"].default == SIMPLE_WRAPPER_DEFAULTS["config_path"]
    assert chat_sig.parameters["stream_to_console"].default == SIMPLE_WRAPPER_DEFAULTS["stream_to_console"]

    assert cycle_sig.parameters["config_path"].default == SIMPLE_WRAPPER_DEFAULTS["config_path"]
    assert cycle_sig.parameters["force_text_only"].default == SIMPLE_WRAPPER_DEFAULTS["force_text_only"]
    assert cycle_sig.parameters["reset_session"].default == SIMPLE_WRAPPER_DEFAULTS["reset_session"]

    assert load_sig.parameters["reset_session"].default == SIMPLE_WRAPPER_DEFAULTS["reset_session"]
