# Advanced Parameters

This page explains advanced JSON-based parameter settings for the Simple nodes.

Use these settings from a JSON file selected by the Simple node `config_path`;
this lets you override Simple-node defaults without editing
`config/simple_defaults.json` directly.

## Qwen3.8 Reasoning Effort

Qwen3.8 supports `reasoning_effort` only in Simple-node JSON configuration:

```json
{
  "qwen3.8": {
    "enable_thinking": true,
    "reasoning_effort": "xhigh"
  }
}
```

Supported values are `xhigh`, `medium`, and `low`. The node default is
`medium`. Invalid values produce a warning and fall back to `medium`.

The node converts `low` and `xhigh` into Qwen3.8 system instructions before
prompt construction; it does not pass `reasoning_effort` to
`llama-cpp-python`. `medium` adds no instruction. The setting is inactive when
`enable_thinking` is false, is not available in Full-node UI, and does not
inherit from the `qwen3.5` JSON entry.

## Supported Advanced Generation Settings

Advanced parameters are listed in the following sample file:

- `config/simple_advanced.example.json`

Simple nodes support `seed`, `top_k`, `min_p`, and `present_penalty` in the
`advanced_generation_kwargs` section. The
`advanced_summary_generation_kwargs` section supports `seed` only. Full nodes
do not expose these settings in the UI.

```json
{
  "advanced_generation_kwargs": {
    "seed": 12345,
    "top_k": 40,
    "min_p": 0.05,
    "present_penalty": 0.0
  }
}
```

Missing or `null` values are omitted, so the installed backend supplies its
own defaults. The values shown above match the current JamePeng backend defaults
for the three sampling controls; the node does not inject them when they are
absent. Valid values are:

- `top_k`: integer greater than or equal to `0`
- `min_p`: number from `0.0` to `1.0`
- `present_penalty`: number from `0.0` to `2.0`

Invalid values are omitted with a warning unless `log_level` is `minimal`.

```json
{
  "advanced_summary_generation_kwargs": {
    "seed": 456
  }
}
```

Note: `tensor_split` is not configured through `advanced_backend_kwargs` yet.
It remains a root-level Simple-node JSON config key for backward compatibility;
see [PARAMETERS.md](PARAMETERS.md).

### Why `seed` Is an Advanced Parameter

When the same model, prompt, media input, generation settings, session state,
runtime cache behavior, and backend behavior all match, a fixed seed can improve
the repeatability of stochastic sampling even when `temperature` is greater
than `0`.

However, it does not guarantee global determinism. Other factors can still make
output vary, and a fixed seed may not provide the repeatability users expect.
For that reason, `seed` is treated as an advanced parameter to try carefully
rather than as a general parameter.

### Model Recommendations and Node Parameter Names

The node does not automatically replace sampling settings when the model family
or thinking mode changes. To apply a model recommendation, set the corresponding
values explicitly in Simple JSON.

Qwen3.8 recommends the following settings:

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | presence penalty | repetition penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thinking | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Non-thinking | `0.7` | `0.8` | `20` | `0.0` | `1.5` | `1.0` |

Gemma 4 recommends `temperature: 1.0`, `top_p: 0.95`, and `top_k: 64`
across its documented use cases; it does not publish a separate setting table
for thinking and non-thinking modes.

The names are mapped into this node as follows:

- `temperature`, `top_p`, and `repeat_penalty` are root-level Simple settings.
- Model documentation's `repetition_penalty` corresponds to the node's
  `repeat_penalty`.
- Qwen documentation's `presence_penalty` corresponds conceptually to
  JamePeng's backend keyword `present_penalty`. Keep the JamePeng spelling in
  `advanced_generation_kwargs`.
- `top_k`, `min_p`, and `present_penalty` are Simple-only advanced generation
  settings.

These sampling controls change token selection, not prompt construction. They
therefore do not change the node's KV-cache prompt signature.

### Other Advanced Parameters

Unsupported parameters such as `typical_p` and the Mirostat fields that are
listed in `config/simple_advanced.example.json` are currently ignored. When
`log_level` is not `minimal`, the node prints a warning for those unsupported
keys.

## About `advanced_summary_generation_kwargs`

Summary generation has its own advanced section.

```json
{
  "advanced_summary_generation_kwargs": {
    "seed": 456
  }
}
```

Summary parameters are defined separately from normal generation parameters:

- `summary temperature`: `0.2`
- `summary max_tokens`: `max_tokens_summary`, default `128`
- `summary top_p`: llama.cpp default (not specified by the node)
- `summary repeat_penalty`: llama.cpp default (not specified by the node)

Allowing these parameters to be overridden through
`advanced_summary_generation_kwargs` is a possible future consideration.

## Reproducibility Notes

For reproducibility tests, try the following settings first.

In real-machine testing, output changed with `LlamaTrieCache` enabled even when
the same seed, prompt, model, and media were used. If you want to check
repeatability, `runtime_cache: "off"` is recommended.

```json
{
  "runtime_cache": "off",
  "reset_session": true,
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

## If Fixed Seed Output Still Changes

A fixed seed only controls the sampling random source. The effective generation
inputs must still match.

Check the following first:

- Set `runtime_cache` to `"off"`.
- Use `reset_session: true` or a fresh `session_id`.
- Use the same model file, mmproj file, media input, prompt, and config.
- Make sure history and summary text are not changing the prompt.
- If summary reproducibility matters, set
  `advanced_summary_generation_kwargs.seed`.
- Compare saved history `params` to confirm the effective settings.
- Backend, hardware, and llama-cpp-python differences may still change output.

## History Records

Explicit advanced generation settings accepted by node validation are recorded
in each saved turn's `params`. Defaults supplied implicitly by the backend are
not recorded. If compatibility fallback removes a keyword rejected by an older
backend, history still records the explicitly requested value.

For example:

```json
{
  "params": {
    "advanced_generation_kwargs": {
      "seed": 12345,
      "top_k": 20,
      "min_p": 0.0,
      "present_penalty": 1.5
    }
  }
}
```

If summary advanced settings are applied, they are also recorded:

```json
{
  "params": {
    "advanced_summary_generation_kwargs": {
      "seed": 456
    }
  }
}
```

## Not Yet Active

`config/simple_advanced.example.json` includes experimental fields for future
advanced backend or generation settings. Supported normal-generation keys are
`seed`, `top_k`, `min_p`, and `present_penalty`; summary generation supports
`seed` only. Other advanced keys remain inactive.

`tensor_split` is an advanced backend-style setting, but it is intentionally
kept outside `advanced_backend_kwargs` for now to avoid breaking existing
Simple-node JSON configs.
