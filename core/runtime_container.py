# core/runtime_container.py: runtime-scoped state container for model manager and KV cache.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RuntimeContainer:
    model_manager: Optional[Any] = None
    mem_kv_state: Dict[str, Any] = field(default_factory=dict)
    dialogue_model_managers: Dict[str, Any] = field(default_factory=dict)
