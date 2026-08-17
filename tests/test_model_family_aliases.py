from __future__ import annotations

import types
import pytest


@pytest.mark.parametrize(
    ("model_path", "expected_family"),
    [
        ("C:/models/LLM/Gemma-4-Vision-Instruct.gguf", "gemma4"),
        ("C:/models/LLM/Gemma4-Vision-Instruct.gguf", "gemma4"),
        ("C:/models/LLM/LFM2.5-VL-Instruct.gguf", "lfm2.5-vl"),
        ("C:/models/LLM/LFM2_5VL-Instruct.gguf", "lfm2.5-vl"),
        ("C:/models/LLM/MiniCPM-V-4.6.gguf", "minicpm-v-4.6"),
        ("C:/models/LLM/MiniCPM-V-4_6.gguf", "minicpm-v-4.6"),
        ("C:/models/LLM/MiniCPMV46.gguf", "minicpm-v-4.6"),
        ("C:/models/LLM/Qwen-3.5-27B-Q4_K_M.gguf", "qwen3.5"),
        ("C:/models/LLM/Qwen-3_5-27B-Q4_K_M.gguf", "qwen3.5"),
        ("C:/models/LLM/Qwen-3.6-27B-Q4_K_M.gguf", "qwen3.5"),
        ("C:/models/LLM/Qwen-3_6-27B-Q4_K_M.gguf", "qwen3.5"),
        ("C:/models/LLM/Qwen3.8-27B-Q4_K_M.gguf", "qwen3.8"),
        ("C:/models/LLM/Qwen3_8-27B-Q4_K_M.gguf", "qwen3.8"),
        ("C:/models/LLM/Qwen38-27B-Q4_K_M.gguf", "qwen3.8"),
        ("C:/models/LLM/Qwen-3.8-27B-Q4_K_M.gguf", "qwen3.8"),
        ("C:/models/LLM/Qwen-3_8-27B-Q4_K_M.gguf", "qwen3.8"),
    ],
)
def test_detect_model_family_aliases(load_nodes_module, model_path, expected_family):
    module = load_nodes_module()

    assert module._detect_model_family(model_path) == expected_family


@pytest.mark.parametrize(
    ("model_path", "expected_variant"),
    [
        ("C:/models/Gemma-4-E2B-it.gguf", "e2b"),
        ("C:/models/Gemma4-E4B-it.gguf", "e4b"),
        ("C:/models/Gemma-4-26B-A4B-it.gguf", "26ba4b"),
        ("C:/models/Gemma-4-31B-it.gguf", "31b"),
        ("C:/models/Gemma-4-unknown.gguf", None),
    ],
)
def test_detect_gemma4_variant(load_nodes_module, model_path, expected_variant):
    module = load_nodes_module()

    assert module._detect_gemma4_variant(model_path) == expected_variant


def test_minicpm_v46_declares_handler_and_generation_defaults(load_nodes_module):
    module = load_nodes_module()

    assert module.DECLARED_CHAT_HANDLER_MAP["minicpm-v-4.6"] == "MiniCPMV46ChatHandler"
    assert module.CHAT_HANDLER_KWARGS_MAP["minicpm-v-4.6"]["enable_thinking"] is False


def test_qwen38_reuses_qwen35_handler_and_generation_defaults(load_nodes_module):
    module = load_nodes_module()

    assert module._detect_model_family("C:/models/Qwen3.8-27B.gguf") == "qwen3.8"
    assert module.DECLARED_CHAT_HANDLER_MAP["qwen3.8"] == "Qwen35ChatHandler"
    assert module.CHAT_HANDLER_KWARGS_MAP["qwen3.8"] == {
        "enable_thinking": False,
        "image_min_tokens": 1024,
    }


@pytest.mark.parametrize(
    ("model_name", "mmproj_name"),
    [
        ("Qwen-3.5-27B-Q4_K_M.gguf", "mmproj-Qwen-3.5-27B-f16.gguf"),
        ("Qwen-3.6-27B-Q4_K_M.gguf", "mmproj-Qwen-3_6-27B-f16.gguf"),
        ("Qwen3.8-27B-Q4_K_M.gguf", "mmproj-Qwen3.8-27B-f16.gguf"),
    ],
)
def test_qwen3x_mmproj_auto_detection_uses_family_aliases(
    load_nodes_module,
    tmp_path,
    model_name,
    mmproj_name,
):
    module = load_nodes_module()
    model_path = tmp_path / model_name
    mmproj_path = tmp_path / mmproj_name
    model_path.write_text("model", encoding="utf-8")
    mmproj_path.write_text("projector", encoding="utf-8")

    manager = module.GGUFModelManager()
    family = module._detect_model_family(str(model_path))
    detected = manager._auto_detect_mmproj(str(model_path), family)

    assert detected == manager._normalize_path(str(mmproj_path))



def test_gemma4_e2b_false_vision_warning(load_nodes_module, monkeypatch, capsys):
    module = load_nodes_module()

    module._warn_if_gemma4_vision_thinking_required(
        "C:/models/Gemma-4-E2B-it.gguf",
        "gemma4",
        {"enable_thinking": False},
    )

    captured = capsys.readouterr()
    assert "Gemma4 E2B/E4B vision models" in captured.out
    assert "enable_thinking=True" in captured.out


def test_gemma4_toggle_capable_variants_do_not_warn(load_nodes_module, monkeypatch, capsys):
    module = load_nodes_module()

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


def test_chat_handler_loading_prefers_declared_gemma3_handler(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    class Gemma3ChatHandler:
        pass

    class Gemma4ChatHandler:
        pass

    handler_module = types.SimpleNamespace(
        Gemma3ChatHandler=Gemma3ChatHandler,
        Gemma4ChatHandler=Gemma4ChatHandler,
    )

    handler_map, factory_map, registry = module._load_available_chat_handlers(
        {
            "gemma3": "Gemma3ChatHandler",
            "gemma4": "Gemma4ChatHandler",
        },
        module.CHAT_HANDLER_KWARGS_MAP,
        handler_module,
    )

    assert handler_map["gemma3"] == "Gemma3ChatHandler"
    assert "gemma3" in factory_map
    assert registry["Gemma3ChatHandler"] is Gemma3ChatHandler


def test_chat_handler_loading_falls_back_to_gemma4_for_pypi_gemma3(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    class Gemma4ChatHandler:
        pass

    handler_module = types.SimpleNamespace(Gemma4ChatHandler=Gemma4ChatHandler)

    handler_map, factory_map, registry = module._load_available_chat_handlers(
        {
            "gemma3": "Gemma3ChatHandler",
            "gemma4": "Gemma4ChatHandler",
        },
        module.CHAT_HANDLER_KWARGS_MAP,
        handler_module,
    )

    assert handler_map["gemma3"] == "Gemma4ChatHandler"
    assert handler_map["gemma4"] == "Gemma4ChatHandler"
    assert "gemma3" in factory_map
    assert registry["Gemma4ChatHandler"] is Gemma4ChatHandler


def test_chat_handler_loading_does_not_fallback_without_compat_handler(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    handler_map, factory_map, registry = module._load_available_chat_handlers(
        {
            "gemma3": "Gemma3ChatHandler",
            "gemma4": "Gemma4ChatHandler",
        },
        module.CHAT_HANDLER_KWARGS_MAP,
        types.SimpleNamespace(),
    )

    assert "gemma3" not in handler_map
    assert "gemma3" not in factory_map
    assert registry == {}


def test_chat_handler_instantiation_prefers_mmproj_path(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_chat_handler_instantiation_falls_back_to_clip_model_path(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_chat_handler_instantiation_falls_back_when_clip_model_path_is_required(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_chat_handler_instantiation_drops_unsupported_enable_thinking(load_nodes_module, monkeypatch, capsys):
    module = load_nodes_module()

    class PyPIStyleGemma4Handler:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "enable_thinking" in kwargs:
                raise TypeError("got an unexpected keyword argument 'enable_thinking'")
            self.kwargs = kwargs

    handler = module._instantiate_chat_handler(
        PyPIStyleGemma4Handler,
        "C:/models/mmproj-gemma4.gguf",
        {"enable_thinking": False},
    )

    assert PyPIStyleGemma4Handler.calls == [
        {
            "mmproj_path": "C:/models/mmproj-gemma4.gguf",
            "enable_thinking": False,
        },
        {"mmproj_path": "C:/models/mmproj-gemma4.gguf"},
    ]
    assert handler.kwargs == {"mmproj_path": "C:/models/mmproj-gemma4.gguf"}
    assert "enable_thinking" in capsys.readouterr().out


def test_chat_handler_instantiation_drops_unsupported_image_min_tokens(load_nodes_module, monkeypatch, capsys):
    module = load_nodes_module()

    class PyPIStyleQwen25VLHandler:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "image_min_tokens" in kwargs:
                raise TypeError("got an unexpected keyword argument 'image_min_tokens'")
            self.kwargs = kwargs

    handler = module._instantiate_chat_handler(
        PyPIStyleQwen25VLHandler,
        "C:/models/mmproj-qwen25vl.gguf",
        {"image_min_tokens": 1024},
    )

    assert PyPIStyleQwen25VLHandler.calls == [
        {
            "mmproj_path": "C:/models/mmproj-qwen25vl.gguf",
            "image_min_tokens": 1024,
        },
        {"mmproj_path": "C:/models/mmproj-qwen25vl.gguf"},
    ]
    assert handler.kwargs == {"mmproj_path": "C:/models/mmproj-qwen25vl.gguf"}
    assert "image_min_tokens" in capsys.readouterr().out


def test_chat_handler_instantiation_preserves_unrelated_type_errors(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_model_manager_uses_mmproj_path_for_new_chat_handlers(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_model_manager_reports_gemma3_compat_handler_fallback(load_nodes_module, monkeypatch, capsys):
    module = load_nodes_module()

    class CompatGemma4Handler:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(kwargs)

    class DummyLlama:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(kwargs)

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "Llama", DummyLlama)
    monkeypatch.setattr(module, "chat_handler_factory_map", {"gemma3": object()})
    monkeypatch.setattr(module, "chat_handler_map", {"gemma3": "Gemma4ChatHandler"})
    monkeypatch.setattr(
        module,
        "chat_handler_class_registry",
        {"Gemma4ChatHandler": CompatGemma4Handler},
    )
    manager = module.GGUFModelManager()

    manager.load_model(
        model_path="C:/models/Gemma-3-test.gguf",
        mmproj_path="C:/models/mmproj-gemma3.gguf",
        n_ctx=1024,
        n_gpu_layers=0,
        vision_required=True,
    )

    captured = capsys.readouterr()
    assert "Gemma3ChatHandler is unavailable" in captured.out
    assert "using Gemma4ChatHandler for gemma3 Vision compatibility" in captured.out
    assert CompatGemma4Handler.calls
    assert DummyLlama.calls
    assert "chat_handler" in DummyLlama.calls[0]


def test_model_manager_keeps_text_fallback_when_vision_is_not_required(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_model_manager_raises_when_required_mmproj_auto_detect_fails(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_model_manager_raises_when_required_handler_initialization_fails(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_model_manager_reports_missing_required_handler_with_backend_guidance(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_model_manager_raises_when_required_family_is_not_supported(load_nodes_module, monkeypatch):
    module = load_nodes_module()

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


def test_resolve_model_and_mmproj_raises_when_explicit_mmproj_is_missing(load_nodes_module, monkeypatch, tmp_path):
    module = load_nodes_module()
    model_path = tmp_path / "Gemma-4-test.gguf"
    model_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="mmproj not found"):
        module._resolve_model_and_mmproj(
            [str(tmp_path)],
            "Gemma-4-test.gguf",
            "mmproj-gemma4.gguf",
        )
