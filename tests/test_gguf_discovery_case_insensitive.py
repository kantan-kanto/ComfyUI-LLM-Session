from __future__ import annotations

from pathlib import Path


def test_list_gguf_recursive_handles_extension_and_mmproj_case(load_nodes_module, tmp_path: Path):
    module = load_nodes_module()

    (tmp_path / "ModelA.GGUF").write_text("x", encoding="utf-8")
    (tmp_path / "MMPROJ-ModelA.GGUF").write_text("x", encoding="utf-8")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    models, mmprojs = module._list_gguf_recursive(str(tmp_path))

    assert "ModelA.GGUF" in models
    assert "MMPROJ-ModelA.GGUF" in mmprojs
