# Advanced Simple Config Notes

This note records an unpublished design sketch for advanced JSON configuration
used by the Simple nodes.

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
  "generation_kwargs": {
    "seed": null,
    "top_k": null,
    "min_p": null,
    "typical_p": null,
    "tfs_z": null,
    "mirostat_mode": null,
    "mirostat_tau": null,
    "mirostat_eta": null
  },
  "backend_kwargs": {
    "ctx_checkpoints": null,
    "checkpoint_on_device": null,
    "verbosity": null,
    "log_filters": null
  }
}
```

`generation_kwargs` would be for `create_completion()` /
`create_chat_completion()` options such as `seed` and `top_k`.

`backend_kwargs` would be for model-load/backend options passed to `Llama(...)`,
such as checkpoint or backend logging controls.

## Current Decision

Do not publish or wire this into the main branch yet.

Reasons:

- The maintainer does not currently need these advanced parameters.
- Many of these options are backend-version, model, and device dependent.
- Publishing an example before implementing and validating the read path could
  imply support that does not exist yet.
- A broad pass-through surface may increase issue triage and support cost.

The current draft is useful as a design reference, but it should remain on a
non-release branch until there is a concrete user need and implementation plan.

## If Revisited

Prefer starting with a narrow feature instead of opening a broad pass-through.

The first candidate should be `generation_kwargs.seed`, because there has
already been user interest in fixed-seed generation. Add other options only when
there is a clear use case and a focused validation path.

If implemented, use:

- an allowlist of supported keys
- type validation and clamping where appropriate
- omission of `null` values
- tests proving that Simple config values reach the intended runtime path
- documentation that the feature is advanced and backend-dependent
