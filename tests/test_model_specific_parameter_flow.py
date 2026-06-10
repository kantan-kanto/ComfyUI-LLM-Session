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


def test_gemma4_text_builder_consumes_enable_thinking_false(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    messages = [
        {"role": "system", "content": "System rules."},
        {"role": "user", "content": "Hello"},
    ]

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/Gemma-4-31B-it.gguf",
        mmproj_path=module._MMPROJ_NOT_REQUIRED,
        messages=messages,
        text_chat_builder_overrides={"gemma4": {"enable_thinking": False}},
    )

    assert request is not None
    assert request["model_family"] == "gemma4"
    assert request["config"]["enable_thinking"] is False
    assert "System rules.\n\nHello" in request["prompt"]
    assert request["prompt"].endswith("<start_of_turn>model\n")
    assert "<|think|>" not in request["prompt"]
    assert "<|channel>thought\n<channel|>" not in request["prompt"]
    assert request["stop"] == ["<end_of_turn>", "<eos>"]


def test_gemma4_text_builder_consumes_enable_thinking_true(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    messages = [{"role": "user", "content": "Think if needed."}]

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/Gemma4-31B-it.gguf",
        mmproj_path="",
        messages=messages,
        text_chat_builder_overrides={"gemma4": {"enable_thinking": True}},
    )

    assert request is not None
    assert request["config"]["enable_thinking"] is True
    assert request["prompt"].startswith("<start_of_turn>user\n<|think|>\n\nThink if needed.")
    assert request["prompt"].endswith("<start_of_turn>model\n")
    assert "<|channel>thought\n<channel|>" not in request["prompt"]


def test_gemma4_summary_forced_override_wins_over_request_override(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    overrides = module._merge_text_chat_builder_overrides(
        model_path="C:/models/LLM/gemma-4-31B-it.gguf",
        base_overrides={"gemma4": {"enable_thinking": True}},
        forced_overrides_map=module.SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP,
    )

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/gemma-4-31B-it.gguf",
        mmproj_path=None,
        messages=[{"role": "user", "content": "Summarize."}],
        text_chat_builder_overrides=overrides,
    )

    assert overrides == {"gemma4": {"enable_thinking": False}}
    assert request is not None
    assert request["config"]["enable_thinking"] is False
    assert request["prompt"].endswith("<start_of_turn>model\n")
    assert "<|think|>" not in request["prompt"]
    assert "<|channel>thought\n<channel|>" not in request["prompt"]


def test_minicpm_v46_text_builder_consumes_enable_thinking_false(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    messages = [
        {"role": "system", "content": "System rules."},
        {"role": "user", "content": "Hello"},
    ]

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/MiniCPM-V-4.6.gguf",
        mmproj_path=module._MMPROJ_NOT_REQUIRED,
        messages=messages,
        text_chat_builder_overrides={"minicpm-v-4.6": {"enable_thinking": False}},
    )

    assert request is not None
    assert request["model_family"] == "minicpm-v-4.6"
    assert request["config"]["enable_thinking"] is False
    assert "<|im_start|>system\nSystem rules.<|im_end|>" in request["prompt"]
    assert "<|im_start|>user\nHello<|im_end|>" in request["prompt"]
    assert request["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert request["stop"] == ["<|endoftext|>", "<|im_end|>"]


def test_minicpm_v46_text_builder_consumes_enable_thinking_true(monkeypatch):
    module = _load_nodes_module(monkeypatch)
    messages = [{"role": "user", "content": "Think if needed."}]

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/MiniCPMV46.gguf",
        mmproj_path="",
        messages=messages,
        text_chat_builder_overrides={"minicpm-v-4.6": {"enable_thinking": True}},
    )

    assert request is not None
    assert request["config"]["enable_thinking"] is True
    assert request["prompt"].endswith("<|im_start|>assistant\n<think>\n")
    assert "</think>" not in request["prompt"]


def test_minicpm_v46_summary_forced_override_wins_over_request_override(monkeypatch):
    module = _load_nodes_module(monkeypatch)

    overrides = module._merge_text_chat_builder_overrides(
        model_path="C:/models/LLM/minicpm-v-4_6.gguf",
        base_overrides={"minicpm-v-4.6": {"enable_thinking": True}},
        forced_overrides_map=module.SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP,
    )

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/minicpm-v-4_6.gguf",
        mmproj_path=None,
        messages=[{"role": "user", "content": "Summarize."}],
        text_chat_builder_overrides=overrides,
    )

    assert overrides == {"minicpm-v-4.6": {"enable_thinking": False}}
    assert request is not None
    assert request["config"]["enable_thinking"] is False
    assert request["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
