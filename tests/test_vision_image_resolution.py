from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
from PIL import Image


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


def _write_config(tmp_path, config):
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(json.dumps({"schema_version": 1, **config}), encoding="utf-8")
    return str(config_path)


def _encoded_image_size(message_content):
    image_part = next(part for part in message_content if part["type"] == "image_url")
    encoded = image_part["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).size


def test_simple_defaults_keep_legacy_image_max_pixels_when_unset(load_nodes_module, tmp_path):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(_write_config(tmp_path, {}))

    assert defaults["image_max_pixels"] == 262144
    assert "image_max_tokens" not in defaults["chat_handler_overrides"].get("gemma4", {})


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (1048576, 1048576),
        (1000, 65536),
        (99999999, 4194304),
    ],
)
def test_simple_defaults_clamp_image_max_pixels(load_nodes_module, tmp_path, configured, expected):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(_write_config(tmp_path, {"image_max_pixels": configured}))

    assert defaults["image_max_pixels"] == expected


def test_simple_defaults_invalid_image_max_pixels_warns_and_uses_legacy_budget(
    load_nodes_module, tmp_path, capsys
):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(_write_config(tmp_path, {"image_max_pixels": "large"}))

    assert defaults["image_max_pixels"] == 262144
    assert "Invalid image_max_pixels" in capsys.readouterr().out


def test_simple_defaults_read_gemma4_image_max_tokens_with_enable_thinking(load_nodes_module, tmp_path):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"gemma4": {"enable_thinking": True, "image_max_tokens": 448}})
    )

    assert defaults["chat_handler_overrides"]["gemma4"] == {
        "enable_thinking": True,
        "image_max_tokens": 448,
    }
    assert "image_max_tokens" not in defaults["text_chat_builder_overrides"]["gemma4"]


def test_simple_defaults_limit_gemma4_image_max_tokens_to_ubatch_size(
    load_nodes_module, tmp_path, capsys
):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"gemma4": {"image_max_tokens": 1120}})
    )

    assert defaults["chat_handler_overrides"]["gemma4"]["image_max_tokens"] == 512
    assert "limited to 512" in capsys.readouterr().out


@pytest.mark.parametrize("configured", [0, -1, "many"])
def test_simple_defaults_ignore_invalid_gemma4_image_max_tokens(
    load_nodes_module, tmp_path, capsys, configured
):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"gemma4": {"image_max_tokens": configured}})
    )

    assert "image_max_tokens" not in defaults["chat_handler_overrides"].get("gemma4", {})
    assert "Invalid gemma4.image_max_tokens" in capsys.readouterr().out


def test_session_chat_simple_forwards_image_max_pixels(load_nodes_module, tmp_path, monkeypatch):
    module = load_nodes_module(available_models=["dummy.gguf"])
    observed = {}
    monkeypatch.setattr(
        module,
        "_run_session_chat_from_inputs",
        lambda **kwargs: observed.update(kwargs) or ("ok",),
    )

    module.LLMSessionChatSimpleNode().chat_stream(
        user_text="describe",
        session_id="sid",
        model="dummy.gguf",
        mmproj="(Auto-detect)",
        history_dir="",
        config_path=_write_config(tmp_path, {"image_max_pixels": 1048576}),
    )

    assert observed["image_max_pixels"] == 1048576


def test_session_chat_turn_kwargs_reach_turn_execution_request(load_nodes_module, monkeypatch):
    module = load_nodes_module(available_models=["dummy.gguf"])
    observed = {}
    monkeypatch.setattr(
        module.TurnExecutionService,
        "execute_turn",
        lambda _self, request: observed.setdefault("request", request),
    )
    defaults = module._load_simple_defaults(None)
    chat_kwargs = module._build_session_chat_simple_chat_kwargs(
        defaults=defaults,
        model="dummy.gguf",
        history_dir="",
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
    )
    chat_kwargs["image_max_pixels"] = 1048576
    node_request = module._build_session_chat_node_execution_request(
        user_text="describe",
        session_id="sid",
        model="dummy.gguf",
        mmproj="(Auto-detect)",
        media=None,
        enable_thinking=False,
        official_sampling_profile="",
        **chat_kwargs,
    )

    module._execute_session_chat_turn(**node_request.turn_kwargs)

    assert observed["request"].image_max_pixels == 1048576


@pytest.mark.parametrize(
    ("image_max_pixels", "expected_size"),
    [
        (None, (512, 512)),
        (1048576, (1024, 1024)),
    ],
)
def test_build_chat_messages_downscales_images_to_image_max_pixels(
    load_nodes_module, image_max_pixels, expected_size
):
    module = load_nodes_module()
    image = FakeTensor(np.zeros((1, 1024, 1024, 3), dtype=np.float32))

    messages = module.build_chat_messages(
        history={"turns": []},
        user_text="describe",
        media=image,
        model_path="C:/models/gemma-4-26B-A4B-it.gguf",
        image_max_pixels=image_max_pixels,
    )

    assert _encoded_image_size(messages[-1]["content"]) == expected_size


def test_chat_handler_instantiation_drops_unsupported_image_max_tokens(load_nodes_module, capsys):
    module = load_nodes_module()

    class OlderGemma4Handler:
        calls = []

        def __init__(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "image_max_tokens" in kwargs:
                raise TypeError("got an unexpected keyword argument 'image_max_tokens'")
            self.kwargs = kwargs

    handler = module._instantiate_chat_handler(
        OlderGemma4Handler,
        "C:/models/mmproj-gemma-4.gguf",
        {"enable_thinking": True, "image_max_tokens": 448},
    )

    assert handler.kwargs == {"mmproj_path": "C:/models/mmproj-gemma-4.gguf", "enable_thinking": True}
    assert "image_max_tokens" in capsys.readouterr().out
