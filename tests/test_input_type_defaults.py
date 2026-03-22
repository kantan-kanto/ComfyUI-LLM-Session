from __future__ import annotations

import importlib
import sys
import types


def _load_nodes_module(monkeypatch):
    fake_folder_paths = types.SimpleNamespace(
        models_dir="C:/models",
        get_folder_paths=lambda _key: [],
        get_filename_list=lambda _key: [],
        get_output_directory=lambda: "C:/output",
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
    sys.modules.pop("llm_session_nodes", None)
    module = importlib.import_module("llm_session_nodes")
    monkeypatch.setattr(
        module,
        "_get_available_models_and_mmprojs",
        lambda: (["dummy.gguf"], [module._MMPROJ_AUTO, module._MMPROJ_NOT_REQUIRED]),
    )
    return module


def test_session_chat_input_types_default_values(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    input_types = module._input_types_session_chat()
    required = input_types["required"]
    optional = input_types["optional"]

    assert required["system_prompt"][1]["default"] == "You are a helpful assistant."
    assert optional["max_turns"][1]["default"] == 12
    assert optional["summary_chunk_turns"][1]["default"] == 3
    assert optional["persistent_cache"][1]["default"] == "off"
    assert optional["runtime_cache"][1]["default"] == "LlamaTrieCache"
    assert optional["log_level"][1]["default"] == "timing"
    assert optional["stream_to_console"][1]["default"] is False


def test_dialogue_cycle_input_types_default_values(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    input_types = module._input_types_dialogue_cycle()
    required = input_types["required"]
    optional = input_types["optional"]

    assert required["cycles"][1]["default"] == 1
    assert required["system_prompt"][1]["default"] == "You are a helpful assistant."
    assert optional["max_turns"][1]["default"] == 12
    assert optional["summary_chunk_turns"][1]["default"] == 3
    assert optional["persistent_cache"][1]["default"] == "off"
    assert optional["runtime_cache"][1]["default"] == "LlamaTrieCache"
    assert optional["log_level"][1]["default"] == "timing"
    assert optional["stream_to_console"][1]["default"] is False
