from __future__ import annotations

import json

import pytest


def _write_config(tmp_path, config):
    config_path = tmp_path / "simple_defaults.json"
    config_path.write_text(json.dumps({"schema_version": 1, **config}), encoding="utf-8")
    return str(config_path)


def _patch_llama(module, monkeypatch):
    calls = []

    class DummyLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "Llama", DummyLlama)
    return calls


def _load(module, manager, advanced_backend_kwargs=None):
    return manager.load_model(
        model_path="C:/models/test.gguf",
        mmproj_path=module._MMPROJ_NOT_REQUIRED,
        n_ctx=4096,
        n_gpu_layers=0,
        advanced_backend_kwargs=advanced_backend_kwargs,
    )


def test_simple_defaults_omit_backend_kwargs_when_unset(load_nodes_module, tmp_path):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"advanced_backend_kwargs": {"n_batch": None, "n_ubatch": None}})
    )

    assert defaults["advanced_backend_kwargs"] == {}


def test_simple_defaults_accept_n_batch_and_n_ubatch(load_nodes_module, tmp_path, capsys):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(
            tmp_path,
            {"n_ctx": 8192, "advanced_backend_kwargs": {"n_batch": 4096, "n_ubatch": 2048}},
        )
    )

    assert defaults["advanced_backend_kwargs"] == {"n_batch": 4096, "n_ubatch": 2048}
    assert "Warning" not in capsys.readouterr().out


@pytest.mark.parametrize("key", ["n_batch", "n_ubatch"])
def test_simple_defaults_raise_small_backend_batch_to_gemma4_image_token_limit(
    load_nodes_module, tmp_path, capsys, key
):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"advanced_backend_kwargs": {key: 128}})
    )

    assert defaults["advanced_backend_kwargs"] == {key: 512}
    assert f"advanced_backend_kwargs.{key}=128 is too small" in capsys.readouterr().out


@pytest.mark.parametrize("invalid", ["1024", 1024.0, True, [1024]])
def test_simple_defaults_ignore_invalid_backend_batch_with_warning(
    load_nodes_module, tmp_path, capsys, invalid
):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"advanced_backend_kwargs": {"n_batch": invalid, "n_ubatch": 1024}})
    )

    assert defaults["advanced_backend_kwargs"] == {"n_ubatch": 1024}
    assert "Ignoring invalid advanced_backend_kwargs.n_batch" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config", "limit"),
    [
        ({"n_ctx": 2048, "advanced_backend_kwargs": {"n_ubatch": 4096}}, 2048),
        ({"n_ctx": 8192, "advanced_backend_kwargs": {"n_batch": 1024, "n_ubatch": 2048}}, 1024),
    ],
)
def test_simple_defaults_warn_when_backend_will_limit_n_ubatch(
    load_nodes_module, tmp_path, capsys, config, limit
):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(_write_config(tmp_path, config))

    assert defaults["advanced_backend_kwargs"]["n_ubatch"] == config["advanced_backend_kwargs"]["n_ubatch"]
    assert f"the backend limits it to {limit}" in capsys.readouterr().out


def test_simple_defaults_warn_about_unsupported_backend_keys(load_nodes_module, tmp_path, capsys):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"advanced_backend_kwargs": {"n_batch": 1024, "ctx_checkpoints": 4}})
    )

    assert defaults["advanced_backend_kwargs"] == {"n_batch": 1024}
    assert "Ignoring unsupported advanced_backend_kwargs keys: ctx_checkpoints" in capsys.readouterr().out


def test_simple_defaults_keep_backend_kwargs_out_of_generation_kwargs(load_nodes_module, tmp_path):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(
            tmp_path,
            {
                "advanced_generation_kwargs": {"seed": 7},
                "advanced_backend_kwargs": {"n_batch": 1024, "n_ubatch": 1024},
            },
        )
    )

    assert defaults["advanced_generation_kwargs"] == {"seed": 7}


def test_model_manager_passes_backend_kwargs_to_llama(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    calls = _patch_llama(module, monkeypatch)

    _load(module, module.GGUFModelManager(), {"n_batch": 2048, "n_ubatch": 1024})

    assert calls[0]["n_batch"] == 2048
    assert calls[0]["n_ubatch"] == 1024


def test_model_manager_leaves_backend_batch_defaults_when_unset(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    calls = _patch_llama(module, monkeypatch)

    _load(module, module.GGUFModelManager())

    assert "n_batch" not in calls[0]
    assert "n_ubatch" not in calls[0]


def test_model_manager_reloads_when_backend_kwargs_change(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    calls = _patch_llama(module, monkeypatch)
    manager = module.GGUFModelManager()

    first = _load(module, manager, {"n_ubatch": 1024})
    cached = _load(module, manager, {"n_ubatch": 1024})
    reloaded = _load(module, manager, {"n_ubatch": 2048})

    assert cached is first
    assert reloaded is not first
    assert [call.get("n_ubatch") for call in calls] == [1024, 2048]


def _load_vision(module, monkeypatch, advanced_backend_kwargs=None):
    calls = _patch_llama(module, monkeypatch)

    class DummyHandler:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(module, "chat_handler_factory_map", {"gemma4": object()})
    monkeypatch.setattr(module, "chat_handler_map", {"gemma4": "Gemma4ChatHandler"})
    monkeypatch.setattr(module, "chat_handler_class_registry", {"Gemma4ChatHandler": DummyHandler})
    module.GGUFModelManager().load_model(
        model_path="C:/models/Gemma-4-test.gguf",
        mmproj_path="C:/models/mmproj-gemma4.gguf",
        n_ctx=4096,
        n_gpu_layers=0,
        vision_required=True,
        advanced_backend_kwargs=advanced_backend_kwargs,
    )
    return calls[0]


def test_simple_defaults_accept_verbosity_and_logits_all(load_nodes_module, tmp_path, capsys):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"advanced_backend_kwargs": {"verbosity": 3, "logits_all": False}})
    )

    assert defaults["advanced_backend_kwargs"] == {"verbosity": 3, "logits_all": False}
    assert "Warning" not in capsys.readouterr().out


@pytest.mark.parametrize("invalid", [6, -1, True, "3", 3.0])
def test_simple_defaults_ignore_invalid_verbosity_with_warning(load_nodes_module, tmp_path, capsys, invalid):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"advanced_backend_kwargs": {"verbosity": invalid}})
    )

    assert defaults["advanced_backend_kwargs"] == {}
    assert "Ignoring invalid advanced_backend_kwargs.verbosity" in capsys.readouterr().out


@pytest.mark.parametrize("invalid", [0, 1, "false"])
def test_simple_defaults_ignore_invalid_logits_all_with_warning(load_nodes_module, tmp_path, capsys, invalid):
    module = load_nodes_module(available_models=["dummy.gguf"])

    defaults = module._load_simple_defaults(
        _write_config(tmp_path, {"advanced_backend_kwargs": {"logits_all": invalid}})
    )

    assert defaults["advanced_backend_kwargs"] == {}
    assert "Ignoring invalid advanced_backend_kwargs.logits_all" in capsys.readouterr().out


def test_vision_load_keeps_logits_all_true_when_unset(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    call = _load_vision(module, monkeypatch)

    assert call["logits_all"] is True
    assert "chat_handler" in call


def test_vision_load_uses_configured_logits_all(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    call = _load_vision(module, monkeypatch, {"logits_all": False})

    assert call["logits_all"] is False


def test_text_only_load_passes_logits_all_only_when_configured(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    calls = _patch_llama(module, monkeypatch)

    _load(module, module.GGUFModelManager())
    _load(module, module.GGUFModelManager(), {"logits_all": True})

    assert "logits_all" not in calls[0]
    assert calls[1]["logits_all"] is True


def test_model_manager_passes_verbosity_to_llama(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    calls = _patch_llama(module, monkeypatch)

    _load(module, module.GGUFModelManager())
    _load(module, module.GGUFModelManager(), {"verbosity": 3})

    assert "verbosity" not in calls[0]
    assert calls[1]["verbosity"] == 3
    assert calls[1]["verbose"] is False


def test_model_manager_retries_without_verbosity_when_backend_rejects_it(
    load_nodes_module, monkeypatch, capsys
):
    module = load_nodes_module()
    calls = []

    class OldLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if "verbosity" in kwargs:
                raise TypeError("Llama.__init__() got an unexpected keyword argument 'verbosity'")

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "Llama", OldLlama)

    _load(module, module.GGUFModelManager(), {"verbosity": 3, "n_ubatch": 1024})

    assert len(calls) == 2
    assert "verbosity" not in calls[1]
    assert calls[1]["n_ubatch"] == 1024
    assert "rejected 'verbosity'" in capsys.readouterr().out


def test_model_manager_does_not_swallow_unrelated_type_error(load_nodes_module, monkeypatch):
    module = load_nodes_module()

    class BrokenLlama:
        def __init__(self, **kwargs):
            raise TypeError("unsupported operand type(s)")

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "Llama", BrokenLlama)

    with pytest.raises(TypeError, match="unsupported operand"):
        _load(module, module.GGUFModelManager(), {"verbosity": 3})


def test_session_chat_simple_forwards_backend_kwargs(load_nodes_module, tmp_path, monkeypatch):
    module = load_nodes_module(available_models=["dummy.gguf"])
    observed = {}
    monkeypatch.setattr(
        module,
        "_run_session_chat_from_inputs",
        lambda **kwargs: observed.update(kwargs) or ("ok",),
    )

    module.LLMSessionChatSimpleNode().chat_stream(
        user_text="hello",
        session_id="sid",
        model="dummy.gguf",
        mmproj="(Auto-detect)",
        history_dir="",
        config_path=_write_config(tmp_path, {"advanced_backend_kwargs": {"n_ubatch": 1024}}),
    )

    assert observed["advanced_backend_kwargs"] == {"n_ubatch": 1024}


def test_session_chat_turn_kwargs_carry_backend_kwargs_to_turn_execution(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])
    defaults = module._load_simple_defaults("")
    chat_kwargs = module._build_session_chat_simple_chat_kwargs(
        defaults=defaults,
        model="dummy.gguf",
        history_dir="",
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
    )
    chat_kwargs["advanced_backend_kwargs"] = {"n_ubatch": 1024}

    node_request = module._build_session_chat_node_execution_request(
        user_text="hello",
        session_id="sid",
        model="dummy.gguf",
        mmproj="(Auto-detect)",
        media=None,
        enable_thinking=False,
        official_sampling_profile="",
        **chat_kwargs,
    )

    assert node_request.turn_kwargs["advanced_backend_kwargs"] == {"n_ubatch": 1024}


def test_dialogue_cycle_simple_applies_backend_kwargs_to_both_models(
    load_nodes_module, tmp_path, monkeypatch
):
    module = load_nodes_module(available_models=["a.gguf", "b.gguf"])
    observed = []

    def execute_dialogue_cycle_turn(**kwargs):
        observed.append((kwargs["model"], kwargs["advanced_backend_kwargs"]))
        return module.TurnExecutionResult(assistant_text="reply", generation_succeeded=True)

    monkeypatch.setattr(module, "LLAMA_CPP_AVAILABLE", True)
    monkeypatch.setattr(module, "_execute_dialogue_cycle_turn", execute_dialogue_cycle_turn)

    module.LLMDialogueCycleSimpleNode().chat_cycle_simple(
        initial_user_text="hello",
        system="",
        systemA="",
        systemB="",
        session_id="sid",
        cycles=1,
        modelA="a.gguf",
        modelB="b.gguf",
        history_dir=str(tmp_path),
        config_path=_write_config(tmp_path, {"advanced_backend_kwargs": {"n_ubatch": 1024}}),
    )

    assert observed == [("a.gguf", {"n_ubatch": 1024}), ("b.gguf", {"n_ubatch": 1024})]
