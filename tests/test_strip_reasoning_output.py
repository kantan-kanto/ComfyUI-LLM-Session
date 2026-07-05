from __future__ import annotations




def test_strip_reasoning_output_handles_gemma4_channel_delimiter(load_nodes_module, monkeypatch):
    module = load_nodes_module()
    raw = (
        "<|channel>thought\n"
        "hogehoge\n"
        "hogehoge\n"
        "<channel|>final line 1\n"
        "final line 2"
    )

    cleaned = module._strip_reasoning_output(raw)

    assert cleaned == "final line 1\nfinal line 2"
