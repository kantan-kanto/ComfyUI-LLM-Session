# Advanced Parameters

This page documents advanced JSON settings for the Simple nodes.

Use these settings from a JSON file selected by the Simple node `config_path`.
For a fuller example, see:

- `config/simple_advanced.example.json`

## Supported Advanced Generation Settings

The advanced generation section currently supports only `seed`.

```json
{
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

`seed` is passed to llama-cpp-python generation. With the same model, prompt,
image input, generation settings, and backend behavior, this can make stochastic
sampling reproducible even when `temperature` is greater than `0`.

Unsupported keys in `advanced_generation_kwargs` are ignored. When `log_level`
is not `minimal`, the node prints a warning for those unsupported keys.

## Summary Generation Settings

Summary generation has its own advanced section.

```json
{
  "advanced_summary_generation_kwargs": {
    "seed": 456
  }
}
```

The main generation seed is not automatically reused for summaries. If you need
summary reproducibility, set `advanced_summary_generation_kwargs.seed`
explicitly.

Unsupported keys in `advanced_summary_generation_kwargs` are ignored. When
`log_level` is not `minimal`, the node prints a warning for those unsupported
keys.

## Reproducibility Notes

For strict reproducibility tests, use:

```json
{
  "runtime_cache": "off",
  "reset_session": true,
  "advanced_generation_kwargs": {
    "seed": 12345
  }
}
```

Runtime caches such as `LlamaTrieCache` can change deterministic output even
when the same seed, prompt, model, and image are used. `runtime_cache: "off"`
is recommended when you want to verify exact repeatability.

The log message `Using cached model` only means the already loaded model
instance was reused. It is separate from runtime prompt/KV cache behavior.
Exact repeatability depends more directly on `runtime_cache`, session state,
prompt/history content, model, mmproj, image input, and generation settings.

For repeatability checks:

- Use the same model and mmproj.
- Use the same prompt, image input, and generation settings.
- Use `runtime_cache: "off"`.
- Use `reset_session: true` or a fresh session id.
- Confirm the saved history `params` contains the expected
  `advanced_generation_kwargs.seed`.

## History Records

Applied advanced generation settings are recorded in each saved turn's
`params`.

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

`config/simple_advanced.example.json` may contain exploratory fields for future
advanced backend or generation settings. At the moment, the only supported
advanced generation key is `seed`.

In particular, broad backend kwargs and unsupported generation kwargs are not
forwarded automatically. They should be enabled only after each option has been
validated against llama-cpp-python behavior.
