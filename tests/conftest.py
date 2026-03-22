"""Shared pytest setup for repository-local imports."""

from __future__ import annotations

import pathlib
import sys
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
