from __future__ import annotations

import importlib
import json
import sys
import types

from core.defaults import LOG_LEVEL_OPTIONS, PERSISTENT_CACHE_OPTIONS, RUNTIME_CACHE_OPTIONS


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

    assert "media" in optional
    assert "image" not in optional
    assert optional["media"][0] == "*"
    assert required["system_prompt"][1]["default"] == "You are a helpful assistant."
    assert optional["max_turns"][1]["default"] == 12
    assert optional["summary_chunk_turns"][1]["default"] == 3
    assert optional["persistent_cache"][1]["default"] == "off"
    assert optional["runtime_cache"][1]["default"] == "LlamaTrieCache"
    assert optional["log_level"][1]["default"] == "timing"
    assert optional["stream_to_console"][1]["default"] is False
    assert optional["enable_thinking"][1]["default"] is False
    assert optional["persistent_cache"][0] == list(PERSISTENT_CACHE_OPTIONS)
    assert optional["runtime_cache"][0] == list(RUNTIME_CACHE_OPTIONS)
    assert optional["log_level"][0] == list(LOG_LEVEL_OPTIONS)


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
    assert optional["enable_thinking"][1]["default"] is False
    assert optional["persistent_cache"][0] == list(PERSISTENT_CACHE_OPTIONS)
    assert optional["runtime_cache"][0] == list(RUNTIME_CACHE_OPTIONS)
    assert optional["log_level"][0] == list(LOG_LEVEL_OPTIONS)


def test_session_chat_simple_uses_media_input(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    input_types = module._input_types_session_chat_simple()
    optional = input_types["optional"]

    assert "media" in optional
    assert "image" not in optional
    assert optional["media"][0] == "*"


def test_enable_thinking_overrides_supported_chat_formats(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    chat_handler_overrides = module._merge_enable_thinking_chat_handler_overrides(
        {"qwen3.5": {"image_min_tokens": 2048}},
        True,
    )
    text_chat_builder_overrides = module._merge_enable_thinking_text_chat_builder_overrides(
        None,
        True,
    )

    assert chat_handler_overrides["qwen3.5"]["enable_thinking"] is True
    assert chat_handler_overrides["qwen3.5"]["image_min_tokens"] == 2048
    assert chat_handler_overrides["gemma4"]["enable_thinking"] is True
    assert text_chat_builder_overrides["qwen3.5"]["enable_thinking"] is True
    assert text_chat_builder_overrides["gemma4"]["enable_thinking"] is True


def test_model_specific_config_override_wins_over_full_node_default(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    chat_handler_overrides = module._merge_enable_thinking_chat_handler_overrides(
        {"gemma4": {"enable_thinking": True}},
        False,
    )
    text_chat_builder_overrides = module._merge_enable_thinking_text_chat_builder_overrides(
        {"gemma4": {"enable_thinking": True}},
        False,
    )

    assert chat_handler_overrides["gemma4"]["enable_thinking"] is True
    assert text_chat_builder_overrides["gemma4"]["enable_thinking"] is True


def test_simple_defaults_enable_thinking_overrides_supported_chat_formats(monkeypatch, tmp_path):
    module = _load_nodes_module(monkeypatch)
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "qwen3.5": {"enable_thinking": "false"},
                "minicpm-v-4.6": {"enable_thinking": "false"},
                "gemma4": {"enable_thinking": "true"},
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["chat_handler_overrides"]["qwen3.5"]["enable_thinking"] is False
    assert defaults["text_chat_builder_overrides"]["qwen3.5"]["enable_thinking"] is False
    assert defaults["chat_handler_overrides"]["minicpm-v-4.6"]["enable_thinking"] is False
    assert defaults["text_chat_builder_overrides"]["minicpm-v-4.6"]["enable_thinking"] is False
    assert defaults["chat_handler_overrides"]["gemma4"]["enable_thinking"] is True
    assert defaults["text_chat_builder_overrides"]["gemma4"]["enable_thinking"] is True


def test_simple_defaults_reads_only_advanced_seed_kwargs(monkeypatch, tmp_path):
    module = _load_nodes_module(monkeypatch)
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "advanced_generation_kwargs": {"seed": "123", "top_k": 40},
                "advanced_summary_generation_kwargs": {"seed": 456, "temperature": 0.0},
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["advanced_generation_kwargs"] == {"seed": 123}
    assert defaults["advanced_summary_generation_kwargs"] == {"seed": 456}


def test_simple_defaults_warns_for_unsupported_advanced_keys(monkeypatch, tmp_path, capsys):
    module = _load_nodes_module(monkeypatch)
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "log_level": "timing",
                "advanced_generation_kwargs": {"seed": 123, "top_k": 40, "min_p": 0.05},
                "advanced_summary_generation_kwargs": {"seed": 456, "temperature": 0.0},
            }
        ),
        encoding="utf-8",
    )

    module._load_simple_defaults(str(config_path))

    out = capsys.readouterr().out
    assert "Warning: Ignoring unsupported advanced_generation_kwargs keys: min_p, top_k" in out
    assert "Warning: Ignoring unsupported advanced_summary_generation_kwargs keys: temperature" in out


def test_simple_defaults_suppresses_unsupported_advanced_key_warning_in_minimal_log(monkeypatch, tmp_path, capsys):
    module = _load_nodes_module(monkeypatch)
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "log_level": "minimal",
                "advanced_generation_kwargs": {"seed": 123, "top_k": 40},
            }
        ),
        encoding="utf-8",
    )

    module._load_simple_defaults(str(config_path))

    assert capsys.readouterr().out == ""


def test_simple_defaults_omits_invalid_advanced_seed_kwargs(monkeypatch, tmp_path):
    module = _load_nodes_module(monkeypatch)
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "advanced_generation_kwargs": {"seed": None},
                "advanced_summary_generation_kwargs": {"seed": "not-an-int"},
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["advanced_generation_kwargs"] == {}
    assert defaults["advanced_summary_generation_kwargs"] == {}
