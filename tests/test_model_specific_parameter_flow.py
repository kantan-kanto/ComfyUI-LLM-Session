from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("effort", "expected_prefix"),
    [
        ("xhigh", "Reasoning effort is set to xhigh."),
        ("low", "Reasoning effort is set to low."),
    ],
)
def test_qwen38_reasoning_effort_prefixes_original_system_prompt(
    load_nodes_module, effort, expected_prefix
):
    module = load_nodes_module()

    effective = module._build_effective_system_prompt(
        model_path="C:/models/Qwen3.8-27B.gguf",
        system_prompt="User system rules.",
        enable_thinking=True,
        reasoning_effort=effort,
    )

    assert effective.startswith(expected_prefix)
    assert effective.endswith("\n\nUser system rules.")


@pytest.mark.parametrize(
    ("model_path", "enable_thinking", "effort"),
    [
        ("C:/models/Qwen3.8-27B.gguf", True, "medium"),
        ("C:/models/Qwen3.8-27B.gguf", False, "xhigh"),
        ("C:/models/Qwen3.5-27B.gguf", True, "xhigh"),
        ("C:/models/Qwen3.6-27B.gguf", True, "low"),
    ],
)
def test_reasoning_effort_does_not_change_other_prompt_modes(
    load_nodes_module, model_path, enable_thinking, effort
):
    module = load_nodes_module()

    assert module._build_effective_system_prompt(
        model_path=model_path,
        system_prompt="User system rules.",
        enable_thinking=enable_thinking,
        reasoning_effort=effort,
    ) == "User system rules."


def test_qwen38_text_builder_consumes_enable_thinking_false(load_nodes_module):
    module = load_nodes_module()
    messages = [
        {"role": "system", "content": "System rules."},
        {"role": "user", "content": "Hello"},
    ]

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/Qwen3.8-27B-Q4_K_M.gguf",
        mmproj_path=module._MMPROJ_NOT_REQUIRED,
        messages=messages,
        text_chat_builder_overrides={"qwen3.8": {"enable_thinking": False}},
    )

    assert request is not None
    assert request["model_family"] == "qwen3.8"
    assert request["config"]["enable_thinking"] is False
    assert "<|im_start|>system\nSystem rules.<|im_end|>" in request["prompt"]
    assert "<|im_start|>user\nHello<|im_end|>" in request["prompt"]
    assert request["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert request["stop"] == ["<|im_end|>", "<|endoftext|>"]


def test_qwen38_text_builder_consumes_enable_thinking_true(load_nodes_module):
    module = load_nodes_module()

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/Qwen-3.8-27B-Q4_K_M.gguf",
        mmproj_path="",
        messages=[{"role": "user", "content": "Think if needed."}],
        text_chat_builder_overrides={"qwen3.8": {"enable_thinking": True}},
    )

    assert request is not None
    assert request["model_family"] == "qwen3.8"
    assert request["config"]["enable_thinking"] is True
    assert request["prompt"].endswith("<|im_start|>assistant\n<think>\n")
    assert "</think>" not in request["prompt"]


def test_qwen38_summary_forced_override_wins_over_request_override(load_nodes_module):
    module = load_nodes_module()

    overrides = module._merge_text_chat_builder_overrides(
        model_path="C:/models/LLM/Qwen3_8-27B-Q4_K_M.gguf",
        base_overrides={"qwen3.8": {"enable_thinking": True}},
        forced_overrides_map=module.SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP,
    )

    request = module._build_text_chat_request(
        model_path="C:/models/LLM/Qwen3_8-27B-Q4_K_M.gguf",
        mmproj_path=None,
        messages=[{"role": "user", "content": "Summarize."}],
        text_chat_builder_overrides=overrides,
    )

    assert overrides == {"qwen3.8": {"enable_thinking": False}}
    assert request is not None
    assert request["config"]["enable_thinking"] is False
    assert request["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")



def test_gemma4_text_builder_consumes_enable_thinking_false(load_nodes_module):
    module = load_nodes_module()
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


def test_gemma4_text_builder_consumes_enable_thinking_true(load_nodes_module):
    module = load_nodes_module()
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


def test_gemma4_summary_forced_override_wins_over_request_override(load_nodes_module):
    module = load_nodes_module()

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


def test_minicpm_v46_text_builder_consumes_enable_thinking_false(load_nodes_module):
    module = load_nodes_module()
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


def test_minicpm_v46_text_builder_consumes_enable_thinking_true(load_nodes_module):
    module = load_nodes_module()
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


def test_minicpm_v46_summary_forced_override_wins_over_request_override(load_nodes_module):
    module = load_nodes_module()

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
