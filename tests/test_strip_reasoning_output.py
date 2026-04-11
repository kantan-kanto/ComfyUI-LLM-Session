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
    return importlib.import_module("llm_session_nodes")


def test_strip_reasoning_output_handles_gemma4_channel_delimiter(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    raw = (
        "<|channel>thought\n"
        "hogehoge\n"
        "hogehoge\n"
        "<channel|>final line 1\n"
        "final line 2"
    )

    cleaned = module._strip_reasoning_output(raw)

    assert cleaned == "final line 1\nfinal line 2"
