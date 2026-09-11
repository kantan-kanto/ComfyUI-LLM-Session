from __future__ import annotations

import sys
import types
from dataclasses import dataclass


@dataclass
class _Record:
    kind: str
    id: str | None
    options: dict


def _fake_type(kind: str):
    return types.SimpleNamespace(
        Input=lambda input_id, **kwargs: _Record(kind, input_id, kwargs),
        Output=lambda output_id=None, **kwargs: _Record(kind, output_id, kwargs),
    )


class _FakeTemplatePrefix:
    def __init__(self, *, input, prefix, min, max):
        self.input = input
        self.prefix = prefix
        self.min = min
        self.max = max


class _FakeSchema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeNodeOutput:
    def __init__(self, *values):
        self.values = values


def _fake_comfy_io():
    fake_io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=_FakeSchema,
        NodeOutput=_FakeNodeOutput,
        String=_fake_type("STRING"),
        Int=_fake_type("INT"),
        Float=_fake_type("FLOAT"),
        Boolean=_fake_type("BOOLEAN"),
        Combo=_fake_type("COMBO"),
        AnyType=_fake_type("*"),
        Image=_fake_type("IMAGE"),
        Audio=_fake_type("AUDIO"),
        MultiType=types.SimpleNamespace(
            Input=lambda input_id, **kwargs: _Record("COMFY_MULTITYPED_V3", input_id, kwargs)
        ),
        Autogrow=types.SimpleNamespace(
            Input=lambda input_id, **kwargs: _Record("COMFY_AUTOGROW_V3", input_id, kwargs),
            TemplatePrefix=_FakeTemplatePrefix,
        ),
    )
    fake_io.Custom = lambda io_type: _fake_type(io_type)
    return fake_io


def test_v3_adapters_expose_autogrow_media_and_preserve_node_ids(
    load_nodes_module,
    monkeypatch,
):
    fake_io = _fake_comfy_io()
    comfy_api_module = types.ModuleType("comfy_api")
    comfy_latest_module = types.ModuleType("comfy_api.latest")
    comfy_latest_module.io = fake_io
    comfy_api_module.latest = comfy_latest_module
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api_module)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_latest_module)

    module = load_nodes_module(available_models=["dummy.gguf"])

    full_adapter = module.NODE_CLASS_MAPPINGS["LLMSessionChatNode"]
    simple_adapter = module.NODE_CLASS_MAPPINGS["LLMSessionChatSimpleNode"]
    assert full_adapter is module.LLMSessionChatV3Adapter
    assert simple_adapter is module.LLMSessionChatSimpleV3Adapter

    for adapter, expected_node_id in (
        (full_adapter, "LLMSessionChatNode"),
        (simple_adapter, "LLMSessionChatSimpleNode"),
    ):
        schema = adapter.define_schema()
        assert schema.node_id == expected_node_id
        assert schema.accept_all_inputs is True
        autogrow = next(item for item in schema.inputs if item.kind == "COMFY_AUTOGROW_V3")
        assert autogrow.id == "media_inputs"
        template = autogrow.options["template"]
        assert template.prefix == "media_"
        assert template.min == 0
        assert template.max == 9
        assert template.input.kind == "COMFY_MULTITYPED_V3"
        assert template.input.options["types"] == [fake_io.Image, fake_io.Audio]
        user_text = next(item for item in schema.inputs if item.id == "user_text")
        assert user_text.kind == "STRING"
        assert user_text.options["default"] == ""
        assert user_text.options["multiline"] is True


def test_legacy_nodes_remain_mapped_when_v3_autogrow_is_unavailable(load_nodes_module):
    module = load_nodes_module(available_models=["dummy.gguf"])

    assert module.LLMSessionChatV3Adapter is None
    assert module.LLMSessionChatSimpleV3Adapter is None
    assert module.NODE_CLASS_MAPPINGS["LLMSessionChatNode"] is module.LLMSessionChatNode
    assert (
        module.NODE_CLASS_MAPPINGS["LLMSessionChatSimpleNode"]
        is module.LLMSessionChatSimpleNode
    )


def test_v3_adapter_prefers_autogrow_media_over_legacy_media(
    load_nodes_module,
    monkeypatch,
):
    fake_io = _fake_comfy_io()
    comfy_api_module = types.ModuleType("comfy_api")
    comfy_latest_module = types.ModuleType("comfy_api.latest")
    comfy_latest_module.io = fake_io
    comfy_api_module.latest = comfy_latest_module
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api_module)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_latest_module)

    module = load_nodes_module(available_models=["dummy.gguf"])
    captured = {}
    monkeypatch.setattr(
        module.LLMSessionChatNode,
        "chat_stream",
        lambda self, *, media=None, **kwargs: captured.setdefault("media", media) or ("ok",),
    )
    first = object()
    second = object()

    module.LLMSessionChatV3Adapter.execute(
        user_text="hello",
        media_inputs={"media_1": second, "media_0": first},
        media=object(),
    )

    assert captured["media"] == (first, second)
