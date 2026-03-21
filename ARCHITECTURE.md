# Architecture Overview

This document describes the module layering and dependency direction for the refactored LLM Session node implementation.

## Layer Roles
- `llm_session_nodes.py`: ComfyUI node entry points and UI/input wiring.
- `core/`: Reusable pure-ish logic (prompt rewrite, KV signature/load/save flow, generation retry loop, shared result types).
- `services/`: Orchestration logic that coordinates multi-step application flows.
- `infra/`: Side-effect helpers for filesystem/path persistence.

## Dependency Direction
Allowed:
- `llm_session_nodes.py` -> `core/`, `services/`, `infra/`
- `services/` -> `core/` (and injected callbacks)
- `core/` -> standard library only
- `infra/` -> standard library only

Disallowed:
- `core/` -> `services/` or node layer
- `infra/` -> `services/` or node layer

## One-Screen Dependency Diagram
```text
[ComfyUI Nodes / Entry]
llm_session_nodes.py
  |
  | uses
  v
+-------------------+      +-------------------+
|      core/        |      |    services/      |
|-------------------|      |-------------------|
| continue_rewrite  |<-----| chat_turn_service |
| kv_state          |      +-------------------+
| generation_runner |
| turn_types        |
+-------------------+
  |
  | side effects via wrapper calls
  v
+-------------------+
|      infra/       |
|-------------------|
| history_store     |
+-------------------+

External:
- llama-cpp-python (generation/cache backend)
- folder_paths (ComfyUI path resolution)
- filesystem (history json, transcript txt, cache dir)
```

## Key Module Summary
- `core/turn_types.py`: shared dataclass result types.
- `core/continue_rewrite.py`: language-aware rewrite of `continue` prompts.
- `core/kv_state.py`: KV state signature/build/restore/save helpers.
- `core/generation_runner.py`: shared generation retry and fallback flow.
- `services/chat_turn_service.py`: dialogue-cycle orchestration service.
- `infra/history_store.py`: history/transcript path and file I/O helpers.
