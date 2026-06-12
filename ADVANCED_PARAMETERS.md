# Advanced Parameters

This page explains advanced JSON-based parameter settings for the Simple nodes.

Use these settings from a JSON file selected by the Simple node `config_path`.

## Supported Advanced Generation Settings

Candidate advanced parameters are listed in the following sample file:

- `config/simple_advanced.example.json`

Of those candidates, the only keys currently supported are `seed` in the
`advanced_generation_kwargs` section and `seed` in the
`advanced_summary_generation_kwargs` section.

```json
{
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

```json
{
  "advanced_summary_generation_kwargs": {
    "seed": 456
  }
}
```

### Why `seed` Is an Advanced Parameter

When the same model, prompt, image input, generation settings, session state,
runtime cache behavior, and backend behavior all match, a fixed seed can improve
the repeatability of stochastic sampling even when `temperature` is greater
than `0`.

However, it does not guarantee global determinism. Other factors can still make
output vary, and a fixed seed may not provide the repeatability users expect.
For that reason, `seed` is treated as an advanced parameter to try carefully
rather than as a general parameter.

### Advanced Parameters Other Than `seed`

Unsupported parameters other than `seed` that are listed in
`config/simple_advanced.example.json` are currently ignored. When `log_level` is
not `minimal`, the node prints a warning for those unsupported keys.

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
the same seed, prompt, model, and image were used. If you want to check
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
- Use the same model file, mmproj file, image input, prompt, and config.
- Make sure history and summary text are not changing the prompt.
- If summary reproducibility matters, set
  `advanced_summary_generation_kwargs.seed`.
- Compare saved history `params` to confirm the effective settings.
- Backend, hardware, and llama-cpp-python differences may still change output.

## History Records

Applied advanced generation settings are recorded in each saved turn's `params`.

For example:

```json
{
  "params": {
    "advanced_generation_kwargs": {
      "seed": 12345
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
advanced backend or generation settings. At the moment, the only advanced keys
that are read are `advanced_generation_kwargs.seed` and
`advanced_summary_generation_kwargs.seed`.
