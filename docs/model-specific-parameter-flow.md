# Model-Specific Parameter Flow

This document records how model-family-specific parameters are wired through the
node layer. It exists to prevent configuration maps from becoming declarations
that are not actually consumed by a runtime path.

## Qwen3.5 Reference Flow

Qwen3.5 currently uses all three model-specific maps:

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

These maps serve different runtime paths. Adding a key to a map is not enough by
itself; the corresponding execution path must read and consume that key.

Gemma4 follows the same design for `enable_thinking`:

```python
CHAT_HANDLER_KWARGS_MAP["gemma4"] = {
    "enable_thinking": False,
}

TEXT_CHAT_BUILDER_CONFIG_MAP["gemma4"] = {
    "enable_thinking": False,
}

SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP["gemma4"] = {
    "enable_thinking": False,
}
```

MiniCPM-V-4.6 also uses all three maps for `enable_thinking`. The chat-handler
entry applies only to Vision mode, while the text-builder entries apply only to
Text-only completion-style prompts:

```python
CHAT_HANDLER_KWARGS_MAP["minicpm-v-4.6"] = {
    "enable_thinking": False,
}

TEXT_CHAT_BUILDER_CONFIG_MAP["minicpm-v-4.6"] = {
    "enable_thinking": False,
}

SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP["minicpm-v-4.6"] = {
    "enable_thinking": False,
}
```

## Chat Handler Path

The chat handler path is used when a model family is detected and an mmproj is
available, or auto-detected, so the model is loaded with vision support.

Flow:

1. `CHAT_HANDLER_KWARGS_MAP` defines built-in kwargs for each chat format.
2. `_load_available_chat_handlers()` builds handler factories from that map at
   module import time, then applies compatibility aliases only for handler
   classes that are missing from the installed backend.
3. `GGUFModelManager.load_model()` detects the model family from the model file
   name.
4. `_get_chat_handler_kwargs()` starts with
   `CHAT_HANDLER_KWARGS_MAP[model_family]`.
5. If `chat_handler_overrides` contains the same model family, those values are
   merged on top.
6. The selected chat handler is instantiated through the compatibility helper:

```python
_instantiate_chat_handler(
    handler_cls,
    mmproj_path,
    active_chat_handler_kwargs,
)
```

The helper prefers `mmproj_path`, which is the current MTMD argument name in
recent JamePeng `llama-cpp-python` builds. If the installed handler stack rejects
that keyword with an mmproj-specific unexpected-keyword `TypeError`, the helper
falls back to the older `clip_model_path` keyword.

Compatibility aliases preserve the detected model family while adapting to
backend handler availability. For example, `gemma3` still declares
`Gemma3ChatHandler`; if that handler is unavailable but `Gemma4ChatHandler` is
available, the runtime uses `Gemma4ChatHandler` for Gemma 3 Vision compatibility.
When `Gemma3ChatHandler` is present, it remains the preferred handler.

Some backend builds expose a chat handler but do not accept every optional
model-family kwarg. When a handler rejects a known optional kwarg such as
`enable_thinking` or `image_min_tokens` with an unexpected-keyword `TypeError`,
the helper logs a warning, removes only that kwarg, and retries. Unrelated
`TypeError` failures are still propagated.

For Qwen3.5 vision/chat-handler mode, this is where `enable_thinking` and
`image_min_tokens` are consumed.

## Backend Compatibility Layer

The backend compatibility layer exists to absorb small differences between
installed `llama-cpp-python` builds without changing model-family detection or
public node inputs. Use it when a backend exposes enough capability to run a
model family, but its handler class names or optional constructor kwargs differ
from the preferred implementation.

Rules:

1. Keep the declared mapping canonical. `chat_handler_map` should name the
   preferred handler for the model family, usually the handler provided by the
   most complete backend support path.
2. Preserve the detected model family. Do not map a `gemma3` model to `gemma4`
   just to reuse a handler; instead, keep `model_family == "gemma3"` and adapt
   only the resolved handler class.
3. Prefer dedicated handlers. A compatibility alias must apply only when the
   declared handler is unavailable and the fallback handler is available.
4. Keep compatibility aliases explicit in `CHAT_HANDLER_COMPAT_ALIASES`. Do not
   infer aliases from similar model names or broad string matching.
5. Keep optional kwarg fallbacks explicit in `OPTIONAL_CHAT_HANDLER_KWARGS`.
   Retry without a kwarg only when the installed handler rejects that exact
   known optional kwarg with an unexpected-keyword `TypeError`.
6. Do not hide unrelated failures. Constructor errors that are not recognized
   compatibility cases must continue to propagate.
7. Log compatibility fallbacks. Users should be able to see when a different
   handler was used or when an optional kwarg was removed.

Checklist for adding a backend compatibility rule:

1. Confirm the preferred handler path still works and remains preferred when it
   is available.
2. Confirm the fallback path is narrower than the model-family detection path.
   The fallback should change handler resolution only, not prompt building,
   model-family-specific defaults, or text-only behavior.
3. Add focused tests for preferred-handler availability, fallback-handler
   availability, and fallback absence.
4. Add or update tests that prove unrelated `TypeError` failures are not
   swallowed when constructor kwargs are involved.
5. Document user-visible compatibility behavior in `COMPATIBILITY.md` when the
   fallback affects which backend builds can run a model.

## Text Chat Builder Path

The text chat builder path is used only when `_build_text_chat_request()` returns
a request. It is intended for chat formats that need completion-style prompts
instead of normal `create_chat_completion()` messages.

Flow:

1. `TEXT_CHAT_BUILDER_CONFIG_MAP` declares model families that may use a text
   builder and their default builder config.
2. `_build_text_chat_request()` rejects the text-builder path if `mmproj_path` is
   present, or if any message content includes non-text parts.
3. `_detect_model_family()` maps aliases such as `qwen3.5`, `qwen35`,
   `qwen3.6`, and `qwen36` to `qwen3.5`.
4. `_get_text_chat_builder_config()` starts with
   `TEXT_CHAT_BUILDER_CONFIG_MAP[model_family]`.
5. If `text_chat_builder_overrides` contains the same model family, those values
   are merged on top.
6. `_build_text_chat_request()` must explicitly dispatch the model family to a
   prompt builder.

For Qwen3.5, the dispatch is:

```python
if model_family == "qwen3.5":
    prompt, stop = _build_qwen35_text_prompt(messages, config)
```

For Gemma4, the dispatch follows the same rule:

```python
if model_family == "gemma4":
    prompt, stop = _build_gemma4_text_prompt(messages, config)
```

For MiniCPM-V-4.6, the dispatch follows the same rule:

```python
if model_family == "minicpm-v-4.6":
    prompt, stop = _build_minicpm_v46_text_prompt(messages, config)
```

`_build_qwen35_text_prompt()` consumes `config["enable_thinking"]`:

- `True`: appends an opening `<think>` marker.
- `False`: appends an empty `<think></think>` block to discourage thinking
  output.

`_build_gemma4_text_prompt()` also consumes `config["enable_thinking"]`:

- `True`: prepends `<|think|>` to the effective system prompt before it is
  folded into the first user turn.
- `False`: does not add any thinking marker or empty thought-channel prefix.

`_build_minicpm_v46_text_prompt()` consumes `config["enable_thinking"]`:

- `True`: appends an opening `<think>` marker after
  `<|im_start|>assistant`.
- `False`: appends an empty `<think></think>` block after
  `<|im_start|>assistant` to discourage thinking output.

The resulting request is later executed by `_create_text_or_chat_completion()`
using `llm.create_completion(...)`.

Important: a model family listed in `TEXT_CHAT_BUILDER_CONFIG_MAP` still needs an
explicit dispatch branch in `_build_text_chat_request()`. Without that branch,
the config is declared but not consumed.

## Summary Text Builder Forced Overrides

Summarization uses the same text-or-chat completion helper as normal chat, but
it may force stricter builder config to avoid reasoning in summaries.

Flow:

1. `_summarize_with_model()` and `maybe_compact_summary()` call
   `_create_text_or_chat_completion()`.
2. They pass:

```python
forced_text_chat_builder_overrides_map=SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP
```

3. `_create_text_or_chat_completion()` calls
   `_merge_text_chat_builder_overrides()`.
4. `_merge_text_chat_builder_overrides()` detects the model family from
   `model_path`.
5. It starts with request-level `text_chat_builder_overrides`, then applies
   `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP[model_family]` on top.
6. `_build_text_chat_request()` must still dispatch the model family to a real
   builder before the forced config can affect runtime behavior.

For Qwen3.5 summaries, `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP["qwen3.5"]` forces
`enable_thinking=False`, then `_build_qwen35_text_prompt()` consumes that value.
MiniCPM-V-4.6 follows the same summary rule through
`_build_minicpm_v46_text_prompt()`.

## Override Precedence

For normal user-facing generation, the intended precedence is documented in
`docs/architecture.md`:

1. Explicit per-model config overrides
2. Explicit Full-node UI input values
3. Simple built-in defaults / Full UI defaults
4. Model-family fallback maps

For summary generation, `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP` is intentionally
stronger than request-level builder overrides because summaries should be compact
and should not expose reasoning.

## Checklist For Adding A Model-Specific Parameter

When adding a model-specific parameter to one of these maps, verify all relevant
items:

1. The map declaration exists in the correct map.
2. Simple config loading reads the key if users can override it.
3. Full UI merge preserves explicit Simple config overrides.
4. The runtime path consumes the key:
   - chat handler kwargs are passed to the handler constructor, or
   - text builder config is passed to a concrete prompt builder, or
   - summary forced config reaches a concrete text builder.
5. `_build_text_chat_request()` has an explicit dispatch branch for any model
   listed in `TEXT_CHAT_BUILDER_CONFIG_MAP`.
6. Tests cover both:
   - precedence, so explicit config is not overwritten by defaults
   - consumption, so the configured key changes the built request or constructor
     kwargs used by the relevant path
