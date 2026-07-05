from __future__ import annotations

import base64
import inspect

import numpy as np
import pytest


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


def test_build_chat_messages_accepts_image_batch_media(load_nodes_module):
    module = load_nodes_module()
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


def test_build_chat_messages_accepts_gemma4_audio_media(load_nodes_module):
    module = load_nodes_module()
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


def test_build_chat_messages_rejects_audio_for_non_gemma4(load_nodes_module):
    module = load_nodes_module()
    audio = {"waveform": np.zeros((1, 1, 160), dtype=np.float32), "sample_rate": 16000}

    with pytest.raises(ValueError, match="Gemma 4"):
        module.build_chat_messages(
            history={"turns": []},
            user_text="transcribe",
            media=audio,
            model_path="C:/models/qwen3-vl.gguf",
        )


def test_validate_chat_media_rejects_audio_for_non_gemma4_before_build(load_nodes_module):
    module = load_nodes_module()
    audio = {"waveform": np.zeros((1, 1, 160), dtype=np.float32), "sample_rate": 16000}

    with pytest.raises(ValueError, match="Gemma 4"):
        module.validate_chat_media(media=audio, model_path="C:/models/qwen3-vl.gguf")


def test_validate_chat_media_rejects_invalid_audio_shape(load_nodes_module):
    module = load_nodes_module()
    audio = {"waveform": np.zeros((2, 1, 160), dtype=np.float32), "sample_rate": 16000}

    with pytest.raises(ValueError, match="batches are not supported"):
        module.validate_chat_media(media=audio, model_path="C:/models/gemma-4-12B.gguf")


def test_build_chat_messages_rejects_unsupported_media(load_nodes_module):
    module = load_nodes_module()

    with pytest.raises(ValueError, match="Unsupported media input"):
        module.build_chat_messages(
            history={"turns": []},
            user_text="hello",
            media=object(),
            model_path="C:/models/gemma-4-12B-it.gguf",
        )


def test_legacy_image_media_shim_prefers_media_when_present(load_nodes_module):
    module = load_nodes_module()
    media = object()
    image = object()

    assert module._resolve_legacy_image_media(media, image) is media
    assert module._resolve_legacy_image_media(None, image) is image


def test_session_chat_methods_still_accept_legacy_image_kwarg(load_nodes_module):
    module = load_nodes_module()

    full_signature = inspect.signature(module.LLMSessionChatNode.chat_stream)
    simple_signature = inspect.signature(module.LLMSessionChatSimpleNode.chat_stream)

    assert "media" in full_signature.parameters
    assert "image" in full_signature.parameters
    assert "media" in simple_signature.parameters
    assert "image" in simple_signature.parameters
