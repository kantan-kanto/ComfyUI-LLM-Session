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


def test_detect_model_family_gemma4_aliases(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    family_dash = module._detect_model_family("C:/models/LLM/Gemma-4-Vision-Instruct.gguf")
    family_nodash = module._detect_model_family("C:/models/LLM/Gemma4-Vision-Instruct.gguf")

    assert family_dash == "gemma4"
    assert family_nodash == "gemma4"


def test_detect_gemma4_variant(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    assert module._detect_gemma4_variant("C:/models/Gemma-4-E2B-it.gguf") == "e2b"
    assert module._detect_gemma4_variant("C:/models/Gemma4-E4B-it.gguf") == "e4b"
    assert module._detect_gemma4_variant("C:/models/Gemma-4-26B-A4B-it.gguf") == "26ba4b"
    assert module._detect_gemma4_variant("C:/models/Gemma-4-31B-it.gguf") == "31b"
    assert module._detect_gemma4_variant("C:/models/Gemma-4-unknown.gguf") is None


def test_gemma4_e2b_false_vision_warning(monkeypatch, capsys):
    module = _load_nodes_module(monkeypatch)

    module._warn_if_gemma4_vision_thinking_required(
        "C:/models/Gemma-4-E2B-it.gguf",
        "gemma4",
        {"enable_thinking": False},
    )

    captured = capsys.readouterr()
    assert "Gemma4 E2B/E4B vision models" in captured.out
    assert "enable_thinking=True" in captured.out


def test_gemma4_toggle_capable_variants_do_not_warn(monkeypatch, capsys):
    module = _load_nodes_module(monkeypatch)

    module._warn_if_gemma4_vision_thinking_required(
        "C:/models/Gemma-4-31B-it.gguf",
        "gemma4",
        {"enable_thinking": False},
    )
    module._warn_if_gemma4_vision_thinking_required(
        "C:/models/Gemma-4-26B-A4B-it.gguf",
        "gemma4",
        {"enable_thinking": False},
    )
    module._warn_if_gemma4_vision_thinking_required(
        "C:/models/Gemma-4-E4B-it.gguf",
        "gemma4",
        {"enable_thinking": True},
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_detect_model_family_lfm25_aliases(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    family_dash = module._detect_model_family("C:/models/LLM/LFM2.5-VL-Instruct.gguf")
    family_nodash = module._detect_model_family("C:/models/LLM/LFM2_5VL-Instruct.gguf")

    assert family_dash == "lfm2.5-vl"
    assert family_nodash == "lfm2.5-vl"
