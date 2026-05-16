from __future__ import annotations

import importlib
import json
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


def test_simple_defaults_accept_tensor_split(monkeypatch, tmp_path):
    module = _load_nodes_module(monkeypatch)
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "n_gpu_layers": -1,
                "tensor_split": [1.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["tensor_split"] == [1.0, 0.0]


def test_simple_defaults_ignore_invalid_tensor_split(monkeypatch, tmp_path):
    module = _load_nodes_module(monkeypatch)
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensor_split": [0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["tensor_split"] is None


def test_model_manager_passes_tensor_split_to_llama(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    calls = []

    class DummyLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "Llama", DummyLlama)

    manager = module.GGUFModelManager()
    manager.load_model(
        model_path="C:/models/test.gguf",
        mmproj_path=module._MMPROJ_NOT_REQUIRED,
        n_ctx=1024,
        n_gpu_layers=-1,
        tensor_split=[1.0, 0.0],
    )

    assert calls[0]["tensor_split"] == [1.0, 0.0]
