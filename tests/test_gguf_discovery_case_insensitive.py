from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


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


def test_list_gguf_recursive_handles_extension_and_mmproj_case(monkeypatch, tmp_path: Path):
    module = _load_nodes_module(monkeypatch)

    (tmp_path / "ModelA.GGUF").write_text("x", encoding="utf-8")
    (tmp_path / "MMPROJ-ModelA.GGUF").write_text("x", encoding="utf-8")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    models, mmprojs = module._list_gguf_recursive(str(tmp_path))

    assert "ModelA.GGUF" in models
    assert "MMPROJ-ModelA.GGUF" in mmprojs
