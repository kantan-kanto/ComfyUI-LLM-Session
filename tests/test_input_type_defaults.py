from __future__ import annotations

import json

from core.defaults import LOG_LEVEL_OPTIONS, PERSISTENT_CACHE_OPTIONS, RUNTIME_CACHE_OPTIONS


def test_session_chat_input_types_default_values(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])

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
    assert "reasoning_effort" not in required
    assert "reasoning_effort" not in optional


def test_dialogue_cycle_input_types_default_values(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])

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
    assert "reasoning_effort" not in required
    assert "reasoning_effort" not in optional


def test_session_chat_simple_uses_media_input(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])

    input_types = module._input_types_session_chat_simple()
    optional = input_types["optional"]

    assert "media" in optional
    assert "image" not in optional
    assert optional["media"][0] == "*"


def test_enable_thinking_overrides_supported_chat_formats(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])

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


def test_model_specific_config_override_wins_over_full_node_default(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])

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


def test_simple_defaults_enable_thinking_overrides_supported_chat_formats(load_nodes_module, tmp_path):
    module = load_nodes_module(available_models=["dummy.gguf"])
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


def test_simple_defaults_reads_qwen38_reasoning_effort_without_qwen35_fallback(
    load_nodes_module, tmp_path
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "qwen3.5": {"enable_thinking": True, "reasoning_effort": "low"},
                "qwen3.8": {"enable_thinking": True, "reasoning_effort": "xhigh"},
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["reasoning_effort"] == "xhigh"
    assert defaults["chat_handler_overrides"]["qwen3.8"]["enable_thinking"] is True
    assert defaults["text_chat_builder_overrides"]["qwen3.8"]["enable_thinking"] is True


def test_simple_defaults_invalid_qwen38_reasoning_effort_warns_and_uses_medium(
    load_nodes_module, tmp_path, capsys
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps({"schema_version": 1, "qwen3.8": {"reasoning_effort": "maximum"}}),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["reasoning_effort"] == "medium"
    assert "Invalid qwen3.8.reasoning_effort" in capsys.readouterr().out


def test_simple_defaults_does_not_use_qwen35_reasoning_effort_for_qwen38(
    load_nodes_module, tmp_path
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps({"schema_version": 1, "qwen3.5": {"reasoning_effort": "xhigh"}}),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["reasoning_effort"] == "medium"


def test_simple_defaults_reads_model_specific_official_sampling_overrides(
    load_nodes_module, tmp_path
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "qwen3.8": {"official_sampling_override": True},
                "gemma4": {"official_sampling_override": "true"},
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["official_sampling_overrides"] == {
        "qwen3.8": True,
        "gemma4": True,
    }


def test_simple_defaults_disables_official_sampling_overrides_by_default(
    load_nodes_module, tmp_path
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["official_sampling_overrides"] == {
        "qwen3.8": False,
        "gemma4": False,
    }


def test_official_sampling_override_uses_qwen38_thinking_mode(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])
    base_kwargs = {
        "temperature": 0.2,
        "top_p": 0.4,
        "repeat_penalty": 1.4,
        "enable_thinking": False,
        "chat_handler_overrides": {"qwen3.8": {"enable_thinking": True}},
        "advanced_generation_kwargs": {
            "seed": 123,
            "top_k": 40,
            "min_p": 0.3,
            "present_penalty": 0.4,
        },
    }

    resolved = module._resolve_official_sampling_turn_kwargs(
        model="Qwen3.8-27B.gguf",
        turn_kwargs=base_kwargs,
        official_sampling_overrides={"qwen3.8": True},
    )

    assert resolved["official_sampling_profile"] == "qwen3.8-thinking"
    assert resolved["temperature"] == 1.0
    assert resolved["top_p"] == 0.95
    assert resolved["repeat_penalty"] == 1.0
    assert resolved["advanced_generation_kwargs"] == {
        "seed": 123,
        "top_k": 20,
        "min_p": 0.0,
        "present_penalty": 0.0,
    }
    assert base_kwargs["temperature"] == 0.2


def test_disabled_official_sampling_override_preserves_manual_values(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])
    base_kwargs = {
        "temperature": 0.2,
        "top_p": 0.4,
        "repeat_penalty": 1.4,
        "advanced_generation_kwargs": {"top_k": 40, "min_p": 0.3},
    }

    resolved = module._resolve_official_sampling_turn_kwargs(
        model="Qwen3.8-27B.gguf",
        turn_kwargs=base_kwargs,
        official_sampling_overrides={"qwen3.8": False},
    )

    assert resolved == base_kwargs
    assert "official_sampling_profile" not in resolved


def test_session_chat_simple_applies_official_sampling_after_model_selection(
    load_nodes_module, tmp_path
):
    module = load_nodes_module(available_models=["Qwen3.8-27B.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "temperature": 0.2,
                "top_p": 0.4,
                "repeat_penalty": 1.4,
                "qwen3.8": {
                    "enable_thinking": True,
                    "official_sampling_override": True,
                },
                "advanced_generation_kwargs": {
                    "seed": 123,
                    "top_k": 40,
                    "min_p": 0.3,
                    "present_penalty": 0.4,
                },
            }
        ),
        encoding="utf-8",
    )
    defaults = module._load_simple_defaults(str(config_path))

    chat_kwargs = module._build_session_chat_simple_chat_kwargs(
        defaults=defaults,
        model="Qwen3.8-27B.gguf",
        history_dir="",
        chat_handler_overrides=defaults["chat_handler_overrides"],
        text_chat_builder_overrides=defaults["text_chat_builder_overrides"],
    )

    assert chat_kwargs["official_sampling_profile"] == "qwen3.8-thinking"
    assert chat_kwargs["temperature"] == 1.0
    assert chat_kwargs["top_p"] == 0.95
    assert chat_kwargs["repeat_penalty"] == 1.0
    assert chat_kwargs["advanced_generation_kwargs"] == {
        "seed": 123,
        "top_k": 20,
        "min_p": 0.0,
        "present_penalty": 0.0,
    }


def test_official_sampling_override_uses_qwen38_non_thinking_mode(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])

    resolved = module._resolve_official_sampling_turn_kwargs(
        model="Qwen3.8-27B.gguf",
        turn_kwargs={
            "temperature": 1.2,
            "top_p": 0.5,
            "repeat_penalty": 1.3,
            "enable_thinking": False,
            "advanced_generation_kwargs": {},
        },
        official_sampling_overrides={"qwen3.8": True},
    )

    assert resolved["official_sampling_profile"] == "qwen3.8-non-thinking"
    assert resolved["temperature"] == 0.7
    assert resolved["top_p"] == 0.8
    assert resolved["repeat_penalty"] == 1.0
    assert resolved["advanced_generation_kwargs"] == {
        "top_k": 20,
        "min_p": 0.0,
        "present_penalty": 1.5,
    }


def test_dialogue_cycle_resolves_official_sampling_per_model(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])
    common_turn_kwargs = {
        "temperature": 0.2,
        "top_p": 0.4,
        "repeat_penalty": 1.4,
        "enable_thinking": False,
        "chat_handler_overrides": {"qwen3.8": {"enable_thinking": True}},
        "advanced_generation_kwargs": {
            "seed": 123,
            "top_k": 40,
            "min_p": 0.3,
            "present_penalty": 0.4,
        },
        "official_sampling_overrides": {"qwen3.8": True, "gemma4": True},
    }

    request = module._build_dialogue_cycle_request(
        initial_user_text="hello",
        session_id="sampling",
        cycles=1,
        system_prompt="",
        system_prompt_A="",
        system_prompt_B="",
        runtime_cache="off",
        stream_to_console=False,
        reset_session=True,
        history_dir="",
        common_turn_kwargs=common_turn_kwargs,
        model_a="Qwen3.8-27B.gguf",
        mmproj_a="None",
        model_b="gemma-4-31b.gguf",
        mmproj_b="None",
    )

    assert request.turn_kwargs_A["official_sampling_profile"] == "qwen3.8-thinking"
    assert request.turn_kwargs_A["temperature"] == 1.0
    assert request.turn_kwargs_A["advanced_generation_kwargs"]["top_k"] == 20
    assert request.turn_kwargs_B["official_sampling_profile"] == "gemma4"
    assert request.turn_kwargs_B["temperature"] == 1.0
    assert request.turn_kwargs_B["top_p"] == 0.95
    assert request.turn_kwargs_B["repeat_penalty"] == 1.4
    assert request.turn_kwargs_B["advanced_generation_kwargs"] == {
        "seed": 123,
        "top_k": 64,
        "min_p": 0.3,
        "present_penalty": 0.4,
    }
    assert "official_sampling_overrides" not in request.turn_kwargs_A
    assert "official_sampling_overrides" not in request.turn_kwargs_B


def test_simple_defaults_reads_supported_advanced_generation_kwargs(load_nodes_module, tmp_path):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "advanced_generation_kwargs": {
                    "seed": "123",
                    "top_k": 40,
                    "min_p": 0.05,
                    "present_penalty": 0.0,
                },
                "advanced_summary_generation_kwargs": {"seed": 456, "temperature": 0.0},
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["advanced_generation_kwargs"] == {
        "seed": 123,
        "top_k": 40,
        "min_p": 0.05,
        "present_penalty": 0.0,
    }
    assert defaults["advanced_summary_generation_kwargs"] == {"seed": 456}


def test_simple_defaults_does_not_inject_advanced_generation_backend_defaults(
    load_nodes_module, tmp_path
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["advanced_generation_kwargs"] == {}


def test_dialogue_cycle_simple_forwards_shared_advanced_generation_kwargs(
    load_nodes_module, tmp_path
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "advanced_generation_kwargs": {
                    "top_k": 64,
                    "min_p": 0.05,
                    "present_penalty": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    defaults = module._load_simple_defaults(str(config_path))

    chat_kwargs = module._build_dialogue_cycle_simple_chat_kwargs(
        defaults=defaults,
        force_text_only=False,
        history_dir="",
        reset_session=False,
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
    )

    assert chat_kwargs["advanced_generation_kwargs"] == {
        "top_k": 64,
        "min_p": 0.05,
        "present_penalty": 0.0,
    }


def test_simple_defaults_warns_for_unsupported_advanced_keys(load_nodes_module, tmp_path, capsys):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "log_level": "timing",
                "advanced_generation_kwargs": {"seed": 123, "top_k": 40, "typical_p": 0.95},
                "advanced_summary_generation_kwargs": {"seed": 456, "temperature": 0.0},
            }
        ),
        encoding="utf-8",
    )

    module._load_simple_defaults(str(config_path))

    out = capsys.readouterr().out
    assert "Warning: Ignoring unsupported advanced_generation_kwargs keys: typical_p" in out
    assert "Warning: Ignoring unsupported advanced_summary_generation_kwargs keys: temperature" in out


def test_simple_defaults_suppresses_unsupported_advanced_key_warning_in_minimal_log(
    load_nodes_module, tmp_path, capsys
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "log_level": "minimal",
                "advanced_generation_kwargs": {"seed": 123, "typical_p": 0.95},
            }
        ),
        encoding="utf-8",
    )

    module._load_simple_defaults(str(config_path))

    assert capsys.readouterr().out == ""


def test_simple_defaults_omits_invalid_advanced_sampling_kwargs(
    load_nodes_module, tmp_path, capsys
):
    module = load_nodes_module(available_models=["dummy.gguf"])
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "advanced_generation_kwargs": {
                    "top_k": 20.5,
                    "min_p": 1.1,
                    "present_penalty": -0.1,
                },
            }
        ),
        encoding="utf-8",
    )

    defaults = module._load_simple_defaults(str(config_path))

    assert defaults["advanced_generation_kwargs"] == {}
    out = capsys.readouterr().out
    assert "advanced_generation_kwargs.top_k" in out
    assert "advanced_generation_kwargs.min_p" in out
    assert "advanced_generation_kwargs.present_penalty" in out


def test_simple_defaults_omits_invalid_advanced_seed_kwargs(load_nodes_module, tmp_path):
    module = load_nodes_module(available_models=["dummy.gguf"])
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
