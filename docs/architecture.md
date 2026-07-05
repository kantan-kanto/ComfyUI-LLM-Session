# Architecture Overview

- Status: Canonical
- Last reviewed: 2026-07-05
- Update when: Module boundaries, dependency direction, or tracked top-level directories change.

This document describes the repository structure, module layering, and dependency
direction for the LLM Session node implementation.

Generated files, local editor settings, virtual environments, caches, and other
`.gitignore`-ignored directories are intentionally out of scope here. This
document focuses on source, tests, documentation, examples, and packaged assets
that are part of the repository.

## Directory Roles

- `llm_session_nodes.py`: ComfyUI node entry points, UI/input wiring, model
  manager integration, compatibility glue, and runtime dependency assembly. This
  file is also the composition root where request/dependency objects are built
  for service-layer orchestration.
- `core/`: Reusable pure-ish logic and shared runtime helpers. Code here should
  avoid ComfyUI imports, filesystem layout assumptions, and service-layer
  orchestration.
- `services/`: Application orchestration logic for generation, history
  persistence, KV-state coordination, Session Chat turns, and Dialogue Cycle
  node execution.
- `infra/`: Low-level side-effect helpers for history/transcript paths and
  filesystem persistence.
- `config/`: Versioned default/example configuration files, including Simple
  node defaults and advanced Simple config examples.
- `web/`: Frontend extension assets loaded by ComfyUI.
- `docs/`: Maintainer-facing design notes, change rules, audits, known issues,
  and architecture documentation.
- `examples/`: Example workflows or usage artifacts intended to be checked in.
- `images/`: Documentation images and packaged visual assets.
- `tests/`: Refactoring safety-net and regression tests that lock current
  behavior for node wiring, core helpers, services, config defaults, and
  compatibility paths.
- `.github/`: GitHub repository metadata such as issue templates and workflows.

## Dependency Direction

Allowed:

- `llm_session_nodes.py` -> `core/`, `services/`, `infra/`, `config/`
- `services/` -> `core/`, `infra/` only through explicit helpers or injected
  callbacks/dependencies
- `core/` -> standard library only
- `infra/` -> standard library only, except `core.logging_utils` as the current
  cross-cutting logging utility
- If more cross-cutting utilities are needed by both `core/` and `infra/`,
  consider introducing a lower-level `common/` layer instead of adding more
  `infra/` -> `core/` exceptions.
- `web/` -> browser/ComfyUI frontend APIs only
- `tests/` -> production modules and test fixtures
- `docs/`, `examples/`, `images/`, `.github/` -> no production runtime imports

Disallowed:

- `core/` -> `services/`, `infra/`, node layer, ComfyUI APIs, or repository
  path layout
- `infra/` -> `core/`, `services/`, node layer, or ComfyUI APIs, except
  `core.logging_utils` as noted above
- `services/` -> node layer or ComfyUI APIs unless passed in as callbacks from
  the composition root
- production runtime layers (`llm_session_nodes.py`, `core/`, `services/`,
  `infra/`) -> `tests/`, `docs/`, `examples/`, `images/`, or `.github/`

## One-Screen Dependency Diagram

```text
[ComfyUI Nodes / Entry]
llm_session_nodes.py
  | \
  |  \ reads versioned defaults/examples
  |   v
  | config/
  |
  | builds requests/dependencies and injects callbacks
  v
+-------------------+      +----------------------------+
|      core/        |<-----|         services/          |
|-------------------|      |----------------------------|
| defaults          |      | generation_execution       |
| generation_runner |      | turn_execution             |
| kv_state          |      | chat_turn                  |
| logging_utils     |      | history_persistence        |
| runtime_container |      | kv_state_service           |
| turn_types        |      +----------------------------+
+-------------------+                 |
                                      | filesystem side effects
                                      v
                           +-------------------+
                           |      infra/       |
                           |-------------------|
                           | history_store     |
                           +-------------------+

Adjacent repository assets:
- `web/`: ComfyUI frontend extension assets.
- `docs/`, `examples/`, `images/`, `.github/`: documentation, examples,
  packaged assets, and repository metadata.

External runtime dependencies:
- llama-cpp-python (generation/cache backend)
- folder_paths (ComfyUI path resolution)
- comfy.model_management interrupt APIs (injected from the node composition root)
- filesystem (history JSON and transcript TXT persistence)
```

## Key Module Summary

- `core/defaults.py`: centralized behavior-preserving defaults for node config,
  UI defaults, and option labels.
- `core/turn_types.py`: shared dataclass result types.
- `core/continue_rewrite.py`: language-aware rewrite of `continue` prompts.
- `core/kv_state.py`: KV state signature/build/restore/save helpers.
- `core/generation_runner.py`: shared generation retry, fallback flow, and
  injected interrupt/abort handling.
- `core/logging_utils.py`: centralized logging helpers and safe exception
  logging utilities.
- `core/runtime_container.py`: runtime-scoped container for injected Session Chat
  manager, Dialogue Cycle manager pool, and in-memory KV state.
- `services/generation_execution_service.py`: generation execution wrapper that
  owns adaptive context-overflow retry classification.
- `services/turn_execution_service.py`: Session Chat turn orchestration.
- `services/chat_turn_service.py`: Dialogue Cycle request/dependency types and
  node-execution orchestration.
- `services/history_persistence_service.py`: history save/load coordination used
  by service orchestration.
- `services/kv_state_service.py`: service-level KV-state restore/save
  coordination around generation.
- `infra/history_store.py`: history/transcript path and file I/O helpers.
- `llm_session_nodes.py` runtime behavior:
  - Default runtime container is lazily initialized via resolver helpers.
  - Dialogue Cycle managers are resolved from runtime container slots (`A`/`B`)
    so they can persist across runs.
  - Chat handler classes are tracked in an internal registry map rather than by
    mutating `globals()`.
  - Chat-format-specific UI overrides such as `enable_thinking` are assembled
    near the chat handler/text builder config maps before entering service
    orchestration.
  - ComfyUI interrupt/cancel callbacks are assembled here and injected into the
    generation runner so `core/` remains independent of ComfyUI imports.

## Parameter Precedence

Parameters may come from multiple layers: built-in model-family defaults,
Full-node UI inputs, Simple-node defaults, and explicit per-model overrides in
`config/simple_defaults.json`.

When these layers overlap, the intended precedence is:

1. Explicit per-model config overrides, such as `gemma4.enable_thinking` in
   `config/simple_defaults.json`
2. Explicit Full-node UI input values
3. Simple built-in defaults / Full UI defaults
4. Model-family fallback maps such as `CHAT_HANDLER_KWARGS_MAP`,
   `TEXT_CHAT_BUILDER_CONFIG_MAP`, and `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP`

Simple nodes may delegate to Full-node methods for compatibility, but a
Full-node default argument must not overwrite an explicit Simple config override.
Merge helpers that apply UI defaults should preserve existing per-model override
values and only fill missing keys.

For model-family-specific parameters, prefer a named helper that makes this rule
visible in code, for example a helper whose name includes
`preserving_explicit_overrides`. Add a regression test that proves a Simple
config value wins over a Full-node default before adding or changing such merge
behavior.

For detailed wiring of `CHAT_HANDLER_KWARGS_MAP`, `TEXT_CHAT_BUILDER_CONFIG_MAP`,
and `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP`, see
[`model-specific-parameter-flow.md`](model-specific-parameter-flow.md).
