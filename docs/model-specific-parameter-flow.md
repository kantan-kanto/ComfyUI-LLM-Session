# Model-Specific Parameter Flow

- Status: Canonical
- Last reviewed: 2026-08-16
- Update when: Model-family maps or parameter precedence behavior changes.

This document defines the implementation rules for model-family-specific
parameters. It exists to prevent configuration maps from becoming declarations
that are not actually consumed by a runtime path.

## Purpose

Model-specific parameters may affect different execution paths depending on how
the model is loaded and how messages are built. Adding a key to one map is not
enough by itself. The corresponding runtime path must read, validate, merge, and
consume that key.

Use this document when adding or changing:

- model-family detection or aliases
- chat handler kwargs
- text-only chat builder config
- summary text builder forced overrides
- Simple config per-model overrides
- backend compatibility fallbacks for unsupported handler kwargs

## Model Family Detection

Model-family-specific behavior starts with `_detect_model_family(model_path)`.
The detected family key is used to select entries from the model-specific maps.

When adding a new family or alias:

1. Add detection for the canonical family key.
2. Add aliases or filename patterns for known model naming variants.
3. Add tests that prove each expected filename resolves to the canonical key.
4. Keep map keys aligned with the canonical key returned by detection.

Detection should be conservative. If a filename cannot be confidently mapped to a
family, prefer no model-specific behavior over applying the wrong behavior.

## Runtime Paths

Model-specific parameters currently flow through three runtime paths.

### 1. Vision Chat Handler Path

This path is used when a model family is detected and an mmproj is available,
explicitly provided, or auto-detected so the model is loaded with vision support.

Flow:

1. `CHAT_HANDLER_KWARGS_MAP` defines built-in kwargs for each chat format.
2. `_load_available_chat_handlers()` builds handler factories from that map at
   module import time, then applies compatibility aliases only for handler
   classes that are missing from the installed backend.
3. `GGUFModelManager.load_model()` detects the model family from the model file
   name.
4. `_get_chat_handler_kwargs()` starts with
   `CHAT_HANDLER_KWARGS_MAP[model_family]`.
5. Per-model Simple config overrides are merged without replacing explicit
   override values with Full-node defaults.
6. `_instantiate_chat_handler()` passes the final kwargs to the backend handler.
7. Unsupported optional kwargs may be dropped only by targeted compatibility
   fallback logic.

Use this path for parameters consumed by llama-cpp-python chat handler classes,
such as `enable_thinking` or `image_min_tokens`.

### 2. Text-Only Builder Path

This path is used when the node runs without a vision chat handler and builds a
completion-style prompt in the node layer.

Flow:

1. `_build_text_chat()` detects the model family from the model path.
2. If the family exists in `TEXT_CHAT_BUILDER_CONFIG_MAP`, the map entry is
   copied into a local config dict.
3. Request-level `text_chat_builder_config` values are merged on top.
4. The builder consumes supported keys while constructing the prompt.

Use this path for parameters that affect text-only prompt construction, such as
whether thinking markup should be enabled or suppressed.

### 3. Summary Text Builder Forced Override Path

Summary generation may reuse text-chat prompt building, but it must be stable and
compact. Some model families need forced summary overrides even if the normal
request config differs.

Flow:

1. Summary helpers receive `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP`.
2. The detected model family selects forced summary config.
3. Forced summary config wins over request-level text builder config for summary
   generation.
4. Normal generation config is not implicitly copied into summary generation.

Use this path for parameters that must be fixed during summary generation, such
as `enable_thinking: false`.

## Model-Specific Maps

### `CHAT_HANDLER_KWARGS_MAP`

Defines built-in kwargs for backend chat handler classes.

Current examples:

```python
CHAT_HANDLER_KWARGS_MAP = {
    "gemma4": {"enable_thinking": False},
    "minicpm-v-4.6": {"enable_thinking": False},
    "qwen2.5-vl": {"image_min_tokens": 1024},
    "qwen3-vl": {"image_min_tokens": 1024},
    "qwen3.5": {"enable_thinking": False, "image_min_tokens": 1024},
}
```

Only add keys that the target chat handler path can consume or safely ignore via
targeted fallback.

### `TEXT_CHAT_BUILDER_CONFIG_MAP`

Defines built-in config for text-only prompt construction.

Current examples:

```python
TEXT_CHAT_BUILDER_CONFIG_MAP = {
    "gemma4": {"enable_thinking": False},
    "minicpm-v-4.6": {"enable_thinking": False},
    "qwen3.5": {"enable_thinking": False},
}
```

Only add keys that `_build_text_chat()` or its helpers explicitly consume.

### `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP`

Defines forced model-specific config for summary text prompt construction.

Current examples:

```python
SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP = {
    "gemma4": {"enable_thinking": False},
    "minicpm-v-4.6": {"enable_thinking": False},
    "qwen3.5": {"enable_thinking": False},
}
```

Forced summary overrides are intentionally stronger than request-level text
builder config during summary generation.

## Parameter Categories

### `enable_thinking`

`enable_thinking` may be consumed by all three runtime paths:

- chat handler kwargs for vision-capable handlers
- text-only builder config for completion-style prompts
- summary forced overrides for compact summary generation

When adding `enable_thinking` support for a new family, verify each path that can
run for that family. A model with both vision and text-only paths may need all
three map entries.

### `image_min_tokens`

`image_min_tokens` is currently a chat-handler kwarg for Qwen VL-style vision
handlers. It belongs in `CHAT_HANDLER_KWARGS_MAP` only unless a future text-only
or summary path explicitly consumes it.

Unsupported `image_min_tokens` should be handled by targeted backend fallback
only when the error clearly identifies that kwarg as unsupported.

### Future Parameters

For future parameters, first identify the runtime path:

- Backend handler constructor option: `CHAT_HANDLER_KWARGS_MAP`
- Text-only prompt construction option: `TEXT_CHAT_BUILDER_CONFIG_MAP`
- Summary-only forced behavior: `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP`

Do not add a parameter to every map by default. Add it only to paths that consume
or intentionally force that behavior.

## Override Precedence

Model-specific values may come from multiple layers:

1. Explicit per-model Simple config overrides
2. Explicit Full-node UI input values
3. Simple built-in defaults / Full UI defaults
4. Built-in model-family fallback maps

Simple nodes may delegate to Full-node methods for compatibility, but Full-node
default arguments must not overwrite explicit Simple config overrides.

Merge helpers that apply UI defaults must preserve existing per-model override
values and only fill missing keys. Prefer helper names that make this rule
visible, such as names containing `preserving_explicit_overrides`.

## Backend Compatibility And Fallback

Backend chat handler signatures differ across llama-cpp-python versions and
handler classes.

Allowed compatibility behavior:

- Prefer `mmproj_path` for new handlers.
- Fall back to `clip_model_path` only when the TypeError clearly indicates
  `mmproj_path` is unsupported.
- Drop unsupported optional kwargs such as `enable_thinking` or
  `image_min_tokens` only when the TypeError clearly identifies that kwarg.
- Preserve unrelated TypeErrors and surface them as real failures.

Do not catch all handler construction failures and retry with silently reduced
config. Broad fallback can hide real backend, model, or path errors.

## Simple Config Interaction

Simple config may provide per-model overrides for supported model-specific
parameters. These overrides should be applied consistently to the runtime paths
that consume the parameter.

Rules:

- Treat Simple config values as explicit per-model overrides.
- Do not let Full-node default arguments replace explicit Simple config values.
- Coerce known value types before applying them to runtime maps.
- Ignore unsupported per-model keys unless a supported runtime path consumes
  them.
- Add regression coverage before changing merge precedence.

## Required Tests

When adding or changing a model-specific parameter, add focused tests for the
paths that can consume it.

Recommended coverage:

- model-family detection and alias matching
- chat handler kwargs selection
- handler compatibility fallback for unsupported kwargs, when relevant
- text-only builder consumption
- summary forced override precedence
- Simple config override precedence over Full-node defaults
- negative tests proving unrelated TypeErrors are not swallowed

Existing tests to use as patterns:

- `tests/test_model_family_aliases.py`
- `tests/test_model_specific_parameter_flow.py`
- `tests/test_input_type_defaults.py`
- `tests/services/test_turn_execution_service.py`

## Current Model Family Examples

### Gemma4

Gemma4 uses `enable_thinking` in all three maps:

```python
CHAT_HANDLER_KWARGS_MAP["gemma4"] = {"enable_thinking": False}
TEXT_CHAT_BUILDER_CONFIG_MAP["gemma4"] = {"enable_thinking": False}
SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP["gemma4"] = {"enable_thinking": False}
```

This ensures vision chat handlers, text-only prompt building, and summary
generation all suppress thinking behavior by default.

### MiniCPM-V-4.6

MiniCPM-V-4.6 uses `enable_thinking` in all three maps:

```python
CHAT_HANDLER_KWARGS_MAP["minicpm-v-4.6"] = {"enable_thinking": False}
TEXT_CHAT_BUILDER_CONFIG_MAP["minicpm-v-4.6"] = {"enable_thinking": False}
SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP["minicpm-v-4.6"] = {"enable_thinking": False}
```

The chat-handler entry applies only to Vision mode. The text-builder entries
apply to Text-only completion-style prompts and summary generation.

### Qwen VL Families

Qwen VL-style vision handlers use `image_min_tokens`:

```python
CHAT_HANDLER_KWARGS_MAP["qwen2.5-vl"] = {"image_min_tokens": 1024}
CHAT_HANDLER_KWARGS_MAP["qwen3-vl"] = {"image_min_tokens": 1024}
```

`image_min_tokens` is not a text-builder or summary-builder setting.

### Qwen3.5-Compatible Families

Qwen3.5, Qwen3.6, and Qwen3.8 normalize to the canonical `qwen3.5` family.
They currently use all three maps:

```python
CHAT_HANDLER_KWARGS_MAP["qwen3.5"] = {
    "enable_thinking": False,
    "image_min_tokens": 1024,
}

TEXT_CHAT_BUILDER_CONFIG_MAP["qwen3.5"] = {
    "enable_thinking": False,
}

SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP["qwen3.5"] = {
    "enable_thinking": False,
}
```

This combines the Qwen VL-style image token budget with thinking suppression for
text-only and summary prompt building.

## Checklist For Adding A Model-Specific Parameter

Before adding a key to any model-specific map:

1. Identify the canonical model family key returned by `_detect_model_family()`.
2. Identify the runtime path that will consume the parameter.
3. Add the key only to the map or maps used by that path.
4. Add or update Simple config override handling if users may override it.
5. Ensure merge precedence preserves explicit per-model overrides.
6. Add backend compatibility fallback only for known unsupported-keyword cases.
7. Add tests for each runtime path the parameter affects.
8. Update this document if the parameter introduces a new category or rule.
9. Update user-facing documentation only if the parameter becomes configurable by
   users.
