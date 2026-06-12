# ComfyUI-LLM-Session
# Copyright (C) 2026 kantan-kanto (https://github.com/kantan-kanto)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

__version__ = "1.2.3"

try:
    from .llm_session_nodes import (
        NODE_CLASS_MAPPINGS as sNODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as sNODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception:
    try:
        from llm_session_nodes import (
            NODE_CLASS_MAPPINGS as sNODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS as sNODE_DISPLAY_NAME_MAPPINGS,
        )
    except Exception as absolute_import_error:
        raise RuntimeError(
            "[LLM Session] Failed to import node mappings via both package and absolute imports."
        ) from absolute_import_error

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(sNODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(sNODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]
