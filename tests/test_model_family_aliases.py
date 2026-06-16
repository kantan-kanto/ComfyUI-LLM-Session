from __future__ import annotations

import importlib
import sys
import types

import pytest


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


def test_detect_model_family_minicpm_v46_aliases(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    family_dash = module._detect_model_family("C:/models/LLM/MiniCPM-V-4.6.gguf")
    family_underscore = module._detect_model_family("C:/models/LLM/MiniCPM-V-4_6.gguf")
    family_compact = module._detect_model_family("C:/models/LLM/MiniCPMV46.gguf")

    assert family_dash == "minicpm-v-4.6"
    assert family_underscore == "minicpm-v-4.6"
    assert family_compact == "minicpm-v-4.6"
    assert module.DECLARED_CHAT_HANDLER_MAP["minicpm-v-4.6"] == "MiniCPMV46ChatHandler"
    assert module.CHAT_HANDLER_KWARGS_MAP["minicpm-v-4.6"]["enable_thinking"] is False


def test_chat_handler_instantiation_prefers_mmproj_path(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class NewHandler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    handler = module._instantiate_chat_handler(
        NewHandler,
        "C:/models/mmproj-gemma4.gguf",
        {"enable_thinking": False},
    )

    assert handler.kwargs == {
        "mmproj_path": "C:/models/mmproj-gemma4.gguf",
        "enable_thinking": False,
    }


def test_chat_handler_instantiation_falls_back_to_clip_model_path(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class OldHandler:
        def __init__(self, **kwargs):
            if "mmproj_path" in kwargs:
                raise TypeError("got an unexpected keyword argument 'mmproj_path'")
            self.kwargs = kwargs

    handler = module._instantiate_chat_handler(
        OldHandler,
        "C:/models/mmproj-gemma4.gguf",
        {"enable_thinking": False},
    )

    assert handler.kwargs == {
        "clip_model_path": "C:/models/mmproj-gemma4.gguf",
        "enable_thinking": False,
    }


def test_chat_handler_instantiation_falls_back_when_clip_model_path_is_required(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class RequiredClipPathHandler:
        def __init__(self, clip_model_path, **kwargs):
            self.clip_model_path = clip_model_path
            self.kwargs = kwargs

    handler = module._instantiate_chat_handler(
        RequiredClipPathHandler,
        "C:/models/mmproj-gemma4.gguf",
        {"enable_thinking": False},
    )

    assert handler.clip_model_path == "C:/models/mmproj-gemma4.gguf"
    assert handler.kwargs == {"enable_thinking": False}


def test_chat_handler_instantiation_preserves_unrelated_type_errors(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class BadKwargHandler:
        def __init__(self, **_kwargs):
            raise TypeError("got an unexpected keyword argument 'bad_kwarg'")

    with pytest.raises(TypeError, match="bad_kwarg"):
        module._instantiate_chat_handler(
            BadKwargHandler,
            "C:/models/mmproj-gemma4.gguf",
            {"bad_kwarg": True},
        )


def _prepare_vision_manager_test(module, monkeypatch, handler_cls):
    class DummyLlama:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(kwargs)

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "Llama", DummyLlama)
    monkeypatch.setattr(module, "chat_handler_factory_map", {"gemma4": object()})
    monkeypatch.setattr(module, "chat_handler_map", {"gemma4": "Gemma4ChatHandler"})
    monkeypatch.setattr(module, "chat_handler_class_registry", {"Gemma4ChatHandler": handler_cls})
    return DummyLlama


def test_model_manager_uses_mmproj_path_for_new_chat_handlers(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class NewHandler:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(kwargs)

    dummy_llama = _prepare_vision_manager_test(module, monkeypatch, NewHandler)
    manager = module.GGUFModelManager()

    manager.load_model(
        model_path="C:/models/Gemma-4-test.gguf",
        mmproj_path="C:/models/mmproj-gemma4.gguf",
        n_ctx=1024,
        n_gpu_layers=0,
        vision_required=True,
    )

    assert NewHandler.calls
    assert NewHandler.calls[0]["mmproj_path"].replace("\\", "/") == "C:/models/mmproj-gemma4.gguf"
    assert "clip_model_path" not in NewHandler.calls[0]
    assert dummy_llama.calls
    assert "chat_handler" in dummy_llama.calls[0]


def test_model_manager_keeps_text_fallback_when_vision_is_not_required(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class FailingHandler:
        def __init__(self, **_kwargs):
            raise RuntimeError("handler boom")

    dummy_llama = _prepare_vision_manager_test(module, monkeypatch, FailingHandler)
    manager = module.GGUFModelManager()

    manager.load_model(
        model_path="C:/models/Gemma-4-test.gguf",
        mmproj_path="C:/models/mmproj-gemma4.gguf",
        n_ctx=1024,
        n_gpu_layers=0,
        vision_required=False,
    )

    assert dummy_llama.calls
    assert "chat_handler" not in dummy_llama.calls[0]


def test_model_manager_raises_when_required_mmproj_auto_detect_fails(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyHandler:
        def __init__(self, **_kwargs):
            pass

    _prepare_vision_manager_test(module, monkeypatch, DummyHandler)
    manager = module.GGUFModelManager()

    with pytest.raises(RuntimeError, match="Failed to auto-detect mmproj"):
        manager.load_model(
            model_path="C:/models/Gemma-4-test.gguf",
            mmproj_path=None,
            n_ctx=1024,
            n_gpu_layers=0,
            vision_required=True,
        )


def test_model_manager_raises_when_required_handler_initialization_fails(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class FailingHandler:
        def __init__(self, **_kwargs):
            raise RuntimeError("handler boom")

    _prepare_vision_manager_test(module, monkeypatch, FailingHandler)
    manager = module.GGUFModelManager()

    with pytest.raises(RuntimeError, match="Vision chat handler initialization failed"):
        manager.load_model(
            model_path="C:/models/Gemma-4-test.gguf",
            mmproj_path="C:/models/mmproj-gemma4.gguf",
            n_ctx=1024,
            n_gpu_layers=0,
            vision_required=True,
        )


def test_model_manager_reports_missing_required_handler_with_backend_guidance(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyLlama:
        calls = []

        def __init__(self, **_kwargs):
            self.calls.append(_kwargs)
            pass

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "Llama", DummyLlama)
    monkeypatch.setattr(module, "chat_handler_factory_map", {})
    monkeypatch.setattr(module, "chat_handler_map", {})
    monkeypatch.setattr(module, "chat_handler_class_registry", {})
    manager = module.GGUFModelManager()

    with pytest.raises(RuntimeError) as exc_info:
        manager.load_model(
            model_path="C:/models/Gemma-4-test.gguf",
            mmproj_path="C:/models/mmproj-gemma4.gguf",
            n_ctx=1024,
            n_gpu_layers=0,
            vision_required=True,
        )

    msg = str(exc_info.value)
    assert "does not provide the required multimodal chat handler" in msg
    assert "Detected model family: gemma4" in msg
    assert "Required handler: Gemma4ChatHandler" in msg
    assert "Installed llama-cpp-python:" in msg
    assert "https://github.com/JamePeng/llama-cpp-python" in msg
    assert DummyLlama.calls == []


def test_model_manager_raises_when_required_family_is_not_supported(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    class DummyHandler:
        def __init__(self, **_kwargs):
            pass

    _prepare_vision_manager_test(module, monkeypatch, DummyHandler)
    manager = module.GGUFModelManager()

    with pytest.raises(RuntimeError) as exc_info:
        manager.load_model(
            model_path="C:/models/unknown-model.gguf",
            mmproj_path="C:/models/mmproj-unknown.gguf",
            n_ctx=1024,
            n_gpu_layers=0,
            vision_required=True,
        )

    msg = str(exc_info.value)
    assert "does not provide the required multimodal chat handler" in msg
    assert "Detected model family: unknown" in msg
    assert "Required handler: unknown" in msg
    assert "did not match any known multimodal family aliases" in msg


def test_resolve_model_and_mmproj_raises_when_explicit_mmproj_is_missing(monkeypatch, tmp_path):
    module = _load_nodes_module(monkeypatch)
    model_path = tmp_path / "Gemma-4-test.gguf"
    model_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="mmproj not found"):
        module._resolve_model_and_mmproj(
            [str(tmp_path)],
            "Gemma-4-test.gguf",
            "mmproj-gemma4.gguf",
        )
