# Architecture Overview

This document describes the module layering and dependency direction for the refactored LLM Session node implementation.

## Layer Roles
- `llm_session_nodes.py`: ComfyUI node entry points, UI/input wiring, and runtime dependency assembly.
  This module is also the composition root where request/dependency objects are assembled for services.
- `core/`: Reusable pure-ish logic (prompt rewrite, KV signature/load/save flow, generation retry loop, shared result types, centralized defaults, runtime container).
- `services/`: Orchestration logic that coordinates multi-step application flows (including node-execution request/dependency orchestration services).
- `infra/`: Side-effect helpers for filesystem/path persistence.
- `tests/`: Refactoring safety-net tests that lock current behavior for core and service orchestration paths.

## Dependency Direction
Allowed:
- `llm_session_nodes.py` -> `core/`, `services/`, `infra/`
- `services/` -> `core/` (and injected callbacks)
- `core/` -> standard library only
- `infra/` -> standard library only
- `tests/` -> `llm_session_nodes.py`, `core/`, `services/`, `infra/`

Disallowed:
- `core/` -> `services/` or node layer
- `infra/` -> `services/` or node layer
- production layers (`llm_session_nodes.py`, `core/`, `services/`, `infra/`) -> `tests/`

## One-Screen Dependency Diagram
```text
[ComfyUI Nodes / Entry]
llm_session_nodes.py
  | 
  | uses
  ├--------------------------┐
  |                          |
  v                          v
+-------------------+      +----------------------------+
|      core/        |      |         services/          |
|-------------------|      |----------------------------|
| defaults          |      | chat_turn_service          |
| continue_rewrite  |<-----|  - ChatTurnService         |
| kv_state          |<-----|  - DialogueCycleNode       |
| generation_runner |      |    ExecutionService        |
| turn_types        |      | turn_execution_service     |
| runtime_container |      |  - TurnExecutionService    |
|                   |      |  - SessionChatNode         |
|                   |      |    ExecutionService        |
|                   |      | generation_execution_      |
|                   |      | service                    |
|                   |      |  - GenerationExecution     |
|                   |      |    Service                 |
|                   |      | kv_state_service           |
|                   |      |  - KvStateService          |
|                   |      | history_persistence_       |
|                   |      | service                    |
|                   |      |  - HistoryPersistence      |
|                   |      |    Service                 |
+-------------------+      +----------------------------+
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
- comfy.model_management interrupt APIs (injected from the node composition root)
- filesystem (history json, transcript txt, cache dir)
```

## Key Module Summary
- `core/defaults.py`: centralized behavior-preserving defaults for node config, UI defaults, and option labels.
- `core/turn_types.py`: shared dataclass result types.
- `core/continue_rewrite.py`: language-aware rewrite of `continue` prompts.
- `core/kv_state.py`: KV state signature/build/restore/save helpers.
- `core/generation_runner.py`: shared generation retry, fallback flow, and injected interrupt/abort handling.
- `core/runtime_container.py`: runtime-scoped container for injected session-chat manager, dialogue-cycle manager pool, and in-memory KV state.
- `llm_session_nodes.py` runtime behavior:
  - Default runtime container is lazily initialized via resolver helpers.
  - Dialogue-cycle managers are resolved from runtime container slots (`A`/`B`) so they can persist across runs.
  - Chat handler classes are tracked in an internal registry map (not `globals()` mutation).
  - Chat-format-specific UI overrides such as `enable_thinking` are assembled near the chat handler/text builder config maps before entering service orchestration.
  - ComfyUI interrupt/cancel callbacks are assembled here and injected into the generation runner so `core/` remains independent of ComfyUI imports.
- `services/chat_turn_service.py`: dialogue-cycle orchestration and node-execution orchestration (`DialogueCycleNodeExecutionService`).
- `services/turn_execution_service.py`: thin orchestration for shared single-turn execution and node-entry invocation (`SessionChatNodeExecutionService`).
- `services/generation_execution_service.py`: adaptive generation orchestration and assistant output normalization.
- `services/kv_state_service.py`: KV cache restore/save orchestration and mismatch recovery handling.
- `services/history_persistence_service.py`: turn append, optional summarization, and history JSON persistence.
- `infra/history_store.py`: history/transcript path and file I/O helpers.

## Parameter Precedence
Parameters may come from multiple layers: built-in model-family defaults, Full-node UI inputs,
Simple-node defaults, and explicit per-model overrides in `config/simple_defaults.json`.

When these layers overlap, the intended precedence is:

1. Explicit per-model config overrides, such as `gemma4.enable_thinking` in `config/simple_defaults.json`
2. Explicit Full-node UI input values
3. Simple built-in defaults / Full UI defaults
4. Model-family fallback maps such as `CHAT_HANDLER_KWARGS_MAP`, `TEXT_CHAT_BUILDER_CONFIG_MAP`, and `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP`

Simple nodes may delegate to Full-node methods for compatibility, but a Full-node default argument
must not overwrite an explicit Simple config override. Merge helpers that apply UI defaults should
preserve existing per-model override values and only fill missing keys.

For model-family-specific parameters, prefer a named helper that makes this rule visible in code,
for example a helper whose name includes `preserving_explicit_overrides`. Add a regression test
that proves a Simple config value wins over a Full-node default before adding or changing such
merge behavior.

For detailed wiring of `CHAT_HANDLER_KWARGS_MAP`, `TEXT_CHAT_BUILDER_CONFIG_MAP`,
and `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP`, see
[`model-specific-parameter-flow.md`](model-specific-parameter-flow.md).
