"""Shared pytest setup for repository-local imports."""

from __future__ import annotations

import importlib
import pathlib
import sys
import types
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SERVICES_TESTS = ROOT / "tests" / "services"
if str(SERVICES_TESTS) not in sys.path:
    sys.path.insert(0, str(SERVICES_TESTS))

if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = types.SimpleNamespace(
        models_dir=str(ROOT / "models"),
        output_directory=str(ROOT / "output"),
        get_folder_paths=lambda _key: [],
        get_output_directory=lambda: str(ROOT / "output"),
    )


@pytest.fixture(autouse=True)
def reset_logging_state():
    """Reset logging state before each test to avoid test interference."""
    from core.logging_utils import (
        _global_log_level,
        _module_log_levels,
        LOG_LEVEL_TIMING,
    )
    
    # Save original state
    original_global = _global_log_level
    original_module = dict(_module_log_levels)
    
    yield
    
    # Restore original state
    from core.logging_utils import set_global_log_level, _module_log_levels as mls
    set_global_log_level(original_global)
    mls.clear()
    mls.update(original_module)


@pytest.fixture
def load_nodes_module(monkeypatch):
    """Import llm_session_nodes with a stable ComfyUI folder_paths stub."""

    def _load(*, available_models=None, available_mmprojs=None):
        fake_folder_paths = types.SimpleNamespace(
            models_dir="C:/models",
            get_folder_paths=lambda _key: [],
            get_filename_list=lambda _key: [],
            get_output_directory=lambda: "C:/output",
        )
        monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)
        sys.modules.pop("llm_session_nodes", None)
        module = importlib.import_module("llm_session_nodes")
        if available_models is not None or available_mmprojs is not None:
            models = ["dummy.gguf"] if available_models is None else available_models
            mmprojs = (
                [module._MMPROJ_AUTO, module._MMPROJ_NOT_REQUIRED]
                if available_mmprojs is None
                else available_mmprojs
            )
            monkeypatch.setattr(
                module,
                "_get_available_models_and_mmprojs",
                lambda: (list(models), list(mmprojs)),
            )
        return module

    return _load
