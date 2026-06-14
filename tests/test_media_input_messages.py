from __future__ import annotations

import base64
import importlib
import inspect
import sys
import types

import numpy as np
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


class FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)
        self.shape = self._array.shape

    def size(self, dim):
        return self._array.shape[dim]

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __getitem__(self, index):
        return FakeTensor(self._array[index])


def test_build_chat_messages_accepts_image_batch_media(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    image_batch = FakeTensor(np.zeros((2, 2, 2, 3), dtype=np.float32))

    messages = module.build_chat_messages(
        history={"turns": []},
        user_text="describe",
        media=image_batch,
        model_path="C:/models/gemma-4-12B-it.gguf",
    )

    content = messages[-1]["content"]
    image_parts = [part for part in content if part["type"] == "image_url"]
    assert len(image_parts) == 2
    assert content[-1] == {"type": "text", "text": "describe"}


def test_build_chat_messages_accepts_gemma4_audio_media(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    audio = {"waveform": np.zeros((1, 1, 160), dtype=np.float32), "sample_rate": 16000}

    messages = module.build_chat_messages(
        history={"turns": []},
        user_text="transcribe",
        media=audio,
        model_path="C:/models/google-gemma-4-12B.gguf",
    )

    content = messages[-1]["content"]
    assert content[0] == {"type": "text", "text": "transcribe"}
    audio_part = content[1]
    assert audio_part["type"] == "input_audio"
    assert audio_part["input_audio"]["format"] == "wav"
    assert base64.b64decode(audio_part["input_audio"]["data"]).startswith(b"RIFF")


def test_build_chat_messages_rejects_audio_for_non_gemma4(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    audio = {"waveform": np.zeros((1, 1, 160), dtype=np.float32), "sample_rate": 16000}

    with pytest.raises(ValueError, match="Gemma 4"):
        module.build_chat_messages(
            history={"turns": []},
            user_text="transcribe",
            media=audio,
            model_path="C:/models/qwen3-vl.gguf",
        )


def test_validate_chat_media_rejects_audio_for_non_gemma4_before_build(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    audio = {"waveform": np.zeros((1, 1, 160), dtype=np.float32), "sample_rate": 16000}

    with pytest.raises(ValueError, match="Gemma 4"):
        module.validate_chat_media(media=audio, model_path="C:/models/qwen3-vl.gguf")


def test_validate_chat_media_rejects_invalid_audio_shape(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    audio = {"waveform": np.zeros((2, 1, 160), dtype=np.float32), "sample_rate": 16000}

    with pytest.raises(ValueError, match="batches are not supported"):
        module.validate_chat_media(media=audio, model_path="C:/models/gemma-4-12B.gguf")


def test_build_chat_messages_rejects_unsupported_media(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported media input"):
        module.build_chat_messages(
            history={"turns": []},
            user_text="hello",
            media=object(),
            model_path="C:/models/gemma-4-12B-it.gguf",
        )


def test_legacy_image_media_shim_prefers_media_when_present(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    media = object()
    image = object()

    assert module._resolve_legacy_image_media(media, image) is media
    assert module._resolve_legacy_image_media(None, image) is image


def test_session_chat_methods_still_accept_legacy_image_kwarg(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    full_signature = inspect.signature(module.LLMSessionChatNode.chat_stream)
    simple_signature = inspect.signature(module.LLMSessionChatSimpleNode.chat_stream)

    assert "media" in full_signature.parameters
    assert "image" in full_signature.parameters
    assert "media" in simple_signature.parameters
    assert "image" in simple_signature.parameters
