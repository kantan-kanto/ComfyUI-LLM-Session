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

## Vision Image Resolution

Simple-node JSON can raise how much image detail reaches a Vision model:

```json
{
  "advanced_generation_kwargs": {
    "image_max_pixels": 1048576
  }
}
```

- `advanced_generation_kwargs.image_max_pixels`: `LLM Session Chat (Simple)`
  downscales IMAGE inputs larger than this pixel count before sending them to
  the model. The default is `262144` (about 512x512). Values are clamped to
  `65536`-`4194304`; invalid values produce a warning and use the default. The
  node applies this value itself and does not pass it to `llama-cpp-python`.

`1048576` (about 1024x1024) is the recommended value for Gemma 4 and Qwen3.x
and is used in `config/simple_advanced.example.json`. The built-in default stays
`262144` to avoid higher image token counts for other model families and on CPU
setups. The chat handler also applies a fixed per-image token range:

| Model family | Fixed handler setting | Approx. pixels |
| --- | --- | --- |
| Gemma 4 | `image_max_tokens: 512` | up to about 1,180,000 |
| Qwen3-VL / Qwen3.5 / Qwen3.6 / Qwen3.8 | `image_min_tokens: 1024` | from about 1,050,000 |

- Gemma 4 images above about 1,180,000 pixels are reduced to 512 tokens by the
  backend, so larger `image_max_pixels` values add no detail. The limit is fixed
  because llama.cpp aborts the process when one Gemma 4 image exceeds the
  default 512-token micro-batch. In local testing, one square image used 121
  tokens at 512x512 and 441 tokens at 1024x1024; a non-square image used 399
  tokens at `1048576`.
- Qwen images below the minimum are upscaled by the backend, so values below
  about `1048576` lose detail without reducing image tokens. In local testing
  with Qwen3.6, one square image used 1024 tokens at both the default and
  `1048576`.

More image tokens increase prompt processing time and context usage.

## Backend Batch Size

Simple-node JSON can set the llama.cpp prompt-processing batch sizes that
`llama-cpp-python` receives when the model is loaded:

```json
{
  "n_ctx": 8192,
  "advanced_backend_kwargs": {
    "n_batch": 2048,
    "n_ubatch": 2048
  }
}
```

- `advanced_backend_kwargs.n_batch`: logical batch size, the most tokens
  submitted to the backend in one decode call.
- `advanced_backend_kwargs.n_ubatch`: physical batch size, the most tokens the
  backend computes in one pass. Raising it reduces the number of passes needed
  to process a long prompt, which is the setting that usually affects prompt
  processing speed on GPU backends such as SYCL.

Both apply to `LLM Session Chat (Simple)` and `LLM Dialogue Cycle (Simple)`; in
Dialogue Cycle the same values are used for model A and model B. Full nodes do
not expose them.

Missing or `null` values are omitted, so the installed backend keeps its own
defaults (`n_batch: 2048` and `n_ubatch: 512` in the `llama-cpp-python` build
used for testing). Values must be JSON integers; other values are ignored with a
warning. Values below `512` are raised to `512`, because llama.cpp aborts the
process when one Gemma 4 image exceeds `n_ubatch`.

`llama-cpp-python` limits `n_batch` to `n_ctx` and `n_ubatch` to `n_batch`, so
`n_ubatch` only takes effect up to the smaller of the two. Set `n_batch` and
`n_ctx` at least as large as `n_ubatch`; the node prints a warning when the
configured `n_ubatch` will be limited.

A larger `n_ubatch` increases backend compute-buffer memory. Changing either
value reloads the model on the next run.

## Backend Log Verbosity and `logits_all`

Two more model-load settings are available in `advanced_backend_kwargs` for
both Simple nodes:

```json
{
  "suppress_backend_logs": false,
  "advanced_backend_kwargs": {
    "verbosity": 3,
    "logits_all": false
  }
}
```

- `advanced_backend_kwargs.verbosity`: native llama.cpp log level as an integer
  from `0` to `5` (`0` output only, `1` error, `2` warning, `3` info, `4`
  trace, `5` debug). When it is missing or `null`, the node keeps its previous
  error-only logging. `3` is the usual llama.cpp log level and is enough to
  check device and layer-offload information. Model-load logs are always
  printed; `suppress_backend_logs: true` can still hide logs written during
  generation, so set it to `false` when you need those.
- `advanced_backend_kwargs.logits_all`: `true` or `false`. When it is missing
  or `null`, the node keeps its previous behavior: `true` for Vision loads and
  the backend default (`false`) for text-only loads. An explicit value applies
  to both. `false` avoids keeping and copying logits for every prompt token,
  which lowers memory use and can speed up prompt processing. Vision with
  `logits_all: false` depends on the backend and model family, so verify it
  with your model before relying on it.

Invalid values are ignored with a warning. `verbosity` is specific to the
JamePeng `llama-cpp-python` build; if the installed backend rejects it, the
node prints a warning and loads the model without it. Changing either value
reloads the model on the next run.

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

By default, the node does not replace sampling settings when the model family or
thinking mode changes. Apply a recommendation either by setting the
corresponding values explicitly or by enabling the model-specific official
sampling override described below.

Qwen3.8 recommends the following settings:

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | `present_penalty` | `repeat_penalty` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thinking | `1.0` | `0.95` | `20` | `0.0` | `0.0` | `1.0` |
| Non-thinking | `0.7` | `0.8` | `20` | `0.0` | `1.5` | `1.0` |

Note: Qwen's official `presence_penalty` corresponds to this node's
`present_penalty`, and the official `repetition_penalty` corresponds to this
node's `repeat_penalty`.

Gemma 4 recommends `temperature: 1.0`, `top_p: 0.95`, and `top_k: 64`
across its documented use cases; it does not publish a separate setting table
for thinking and non-thinking modes.

The names are mapped into this node as follows:

- `temperature`, `top_p`, and `repeat_penalty` are root-level Simple settings.
- `top_k`, `min_p`, and `present_penalty` are Simple-only advanced generation
  settings.

These sampling controls change token selection, not prompt construction. They
therefore do not change the node's KV-cache prompt signature.

### Official Sampling Override

Simple nodes can apply the documented model recommendations automatically with
the model-specific `official_sampling_override` setting. Its default is
`false`.

```json
{
  "qwen3.8": {
    "enable_thinking": true,
    "reasoning_effort": "medium",
    "official_sampling_override": true
  },
  "gemma4": {
    "enable_thinking": false,
    "official_sampling_override": true
  }
}
```

When enabled, official values take priority over matching root-level settings
and `advanced_generation_kwargs`, even when those values were explicitly set.
Only parameters included in the documented recommendation are replaced:

- Qwen3.8 thinking replaces `temperature`, `top_p`, `repeat_penalty`, `top_k`,
  `min_p`, and `present_penalty` with the thinking row above.
- Qwen3.8 non-thinking replaces the same six parameters with the non-thinking
  row above.
- Gemma 4 replaces `temperature`, `top_p`, and `top_k` only. Explicit
  `repeat_penalty`, `min_p`, and `present_penalty` values remain unchanged
  because they are not part of the documented Gemma 4 recommendation.

Qwen3.8 selects its profile from `enable_thinking`; `reasoning_effort` does not
select a different sampling profile. Dialogue Cycle resolves the setting for A
and B independently, so mixed Qwen3.8 and Gemma 4 cycles use different values
for each model. Summary generation is not overridden.

Saved turn parameters contain the effective sampling values and an
`official_sampling_profile` value such as `qwen3.8-thinking`,
`qwen3.8-non-thinking`, or `gemma4`. The override does not change the KV-cache
prompt signature.

### Other Advanced Parameters

Unsupported parameters such as `typical_p`, the Mirostat fields, and the
`advanced_backend_kwargs` keys other than `n_batch`, `n_ubatch`, `verbosity`,
and `logits_all` that are
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

Explicit `advanced_backend_kwargs` values are recorded the same way:

```json
{
  "params": {
    "advanced_backend_kwargs": {
      "n_batch": 2048,
      "n_ubatch": 2048
    }
  }
}
```

## Not Yet Active

`config/simple_advanced.example.json` includes experimental fields for future
advanced backend or generation settings. Supported normal-generation keys are
`seed`, `top_k`, `min_p`, `present_penalty`, and `image_max_pixels`; summary generation supports
`seed` only; backend loading supports `n_batch`, `n_ubatch`, `verbosity`, and
`logits_all`. Other advanced keys remain inactive.

`tensor_split` is an advanced backend-style setting, but it is intentionally
kept outside `advanced_backend_kwargs` for now to avoid breaking existing
Simple-node JSON configs.
