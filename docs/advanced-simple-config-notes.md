# Advanced Simple Config Notes

This note records an advanced JSON configuration sketch for the Simple nodes.
Only the seed read path has been implemented so far; the remaining options are
kept as design notes until they have a concrete validation path.

## Branch

Suggested branch name:

```text
draft/simple-advanced-config
```

## Background

The Simple nodes read a JSON config file through `config_path`. This makes them
a natural place to expose advanced options without adding more controls to the
ComfyUI node UI.

The idea was to add an example config such as:

```text
config/simple_advanced.example.json
```

This file would include the normal `simple_defaults.json` settings plus optional
advanced sections for generation-time and backend-load-time parameters.

## Proposed Shape

The proposed split was:

```json
{
  "schema_version": 1,
  "advanced_generation_kwargs": {
    "seed": null,
    "top_k": null,
    "min_p": null,
    "typical_p": null,
    "mirostat_mode": null,
    "mirostat_tau": null,
    "mirostat_eta": null
  },
  "advanced_summary_generation_kwargs": {
    "seed": null,
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "min_p": null,
    "typical_p": null
  },
  "advanced_backend_kwargs": {
    "ctx_checkpoints": null,
    "checkpoint_on_device": null,
    "verbosity": null,
    "log_filters": null
  }
}
```

`advanced_generation_kwargs` would be for `create_completion()` /
`create_chat_completion()` options such as `seed` and `top_k`.

`advanced_summary_generation_kwargs` would be for summary-generation options
needed for reproducibility. Summary generation does not inherit
`advanced_generation_kwargs` implicitly. If a user wants fixed summary sampling,
they should set `advanced_summary_generation_kwargs` explicitly. If omitted,
summaries should continue using the normal summary settings.

`advanced_backend_kwargs` would be for model-load/backend options passed to `Llama(...)`,
such as checkpoint or backend logging controls.

## Current Decision

Implement only the narrow seed path for now.

Reasons:

- The maintainer does not currently need these advanced parameters.
- Many of these options are backend-version, model, and device dependent.
- A broad pass-through surface may increase issue triage and support cost.

Implemented behavior:

- `advanced_generation_kwargs.seed` is read from Simple config and passed to
  normal generation.
- `advanced_summary_generation_kwargs.seed` is read separately and passed to
  summary generation.
- Summary generation does not inherit `advanced_generation_kwargs.seed`
  implicitly.
- Missing, `null`, or invalid seed values are omitted.
- Other advanced keys in the example are not read yet. They are ignored, with a
  warning when `log_level` is not `minimal`.
- Applied advanced kwargs are recorded in each turn's history `params`.
- For strict reproducibility with `advanced_generation_kwargs.seed`, use
  `runtime_cache: "off"`. Real-machine testing showed that `LlamaTrieCache` can
  change deterministic output even when the same seed, prompt, model, and image
  are used. Keeping the loaded model instance is not the issue by itself; the
  runtime cache mode is the relevant difference.
- Unsupported backend-kwarg fallback is not implemented for the seed-only path.
  If broader kwargs are added later, targeted fallback should be considered only
  for clearly identified unsupported-keyword `TypeError` cases.

## If Revisited

Prefer starting with a narrow feature instead of opening a broad pass-through.

The next candidates should be added only when there is a clear use case and a
focused validation path.

For further additions, use:

- an allowlist of supported keys
- type validation and clamping where appropriate
- omission of `null` values
- tests proving that Simple config values reach the intended runtime path
- documentation that the feature is advanced and backend-dependent
