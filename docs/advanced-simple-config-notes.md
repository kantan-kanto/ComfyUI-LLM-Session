# Advanced Simple Config Implementation Rules

- Status: Canonical
- Last reviewed: 2026-07-05
- Update when: Advanced Simple config support expands, validation rules change, or new advanced parameter categories are added.

This document defines maintainer-facing implementation rules and review points
for adding advanced parameters to Simple-node JSON config.

`ADVANCED_PARAMETERS.md` is the user-facing reference for advanced parameters
that are actually supported. This document is for implementation decisions,
validation rules, compatibility cautions, and future expansion criteria.

## Purpose

The Simple nodes read a JSON config file through `config_path`. This makes them
a natural place to expose advanced options without adding more controls to the
ComfyUI node UI.

Advanced Simple config should remain intentionally narrow. Each new parameter
must have a concrete use case, a clear runtime destination, validation rules,
tests, and user-facing documentation after it becomes supported.

## Current Supported Surface

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
- Unsupported backend-kwarg fallback is not implemented for the seed-only path.

For strict reproducibility with `advanced_generation_kwargs.seed`, use
`runtime_cache: "off"`. Real-machine testing showed that `LlamaTrieCache` can
change deterministic output even when the same seed, prompt, model, and media
are used. Keeping the loaded model instance is not the issue by itself; the
runtime cache mode is the relevant difference.

## Candidate Config Shape

`config/simple_advanced.example.json` is the staging place for future advanced
Simple config ideas. It may include the normal `simple_defaults.json` settings
plus optional advanced sections for generation-time and backend-load-time
parameters.

The intended top-level split is:

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

Section meanings:

- `advanced_generation_kwargs`: runtime generation options for
  `create_completion()` / `create_chat_completion()`, such as `seed` and
  sampling controls.
- `advanced_summary_generation_kwargs`: summary-generation options needed for
  reproducibility or future summary-specific sampling behavior.
- `advanced_backend_kwargs`: model-load/backend options passed to `Llama(...)`,
  such as checkpoint or backend logging controls.

Example-only keys are not supported until they are explicitly read, validated,
tested, and documented.

## Design Rules For Adding Advanced Parameters

Add a new advanced parameter only when all of these are true:

1. There is a concrete maintainer or user need.
2. The target runtime API and call path are known.
3. Backend-version, model-family, and device-dependent behavior is understood
   enough to document the risk.
4. The parameter can be allowlisted rather than passed through broadly.
5. Invalid values can be omitted or rejected predictably.
6. Tests can prove that the Simple config value reaches the intended runtime
   path.

Do not add broad pass-through surfaces only because the backend accepts arbitrary
kwargs. A broad surface increases issue triage cost and can make backend-specific
failures look like node bugs.

## Validation And Compatibility Rules

Use an explicit allowlist for each advanced section.

For each supported key:

- Validate type.
- Clamp numeric values where a safe range is known.
- Omit `null` values.
- Omit missing values.
- Treat invalid values as omitted unless there is a strong reason to fail fast.
- Warn about unsupported keys when `log_level` is not `minimal`.
- Keep warning text clear that unsupported example keys are intentionally ignored.

Prefer narrow parsers such as seed-only helpers over generic dictionary
forwarding. Generic forwarding should be considered only after enough keys share
the same validation, warning, history, and fallback behavior.

## Summary Generation Rules

Summary generation has its own advanced section.

Do not implicitly copy normal generation options into summary generation. If a
user wants fixed summary sampling or other summary-specific behavior, they should
set `advanced_summary_generation_kwargs` explicitly.

When adding summary advanced parameters:

- Preserve current summary defaults unless the key is explicitly configured.
- Test normal generation and summary generation separately.
- Record applied summary advanced kwargs separately in history params.
- Update `ADVANCED_PARAMETERS.md` with summary-specific usage notes.

## Backend Fallback Rules

Backend kwargs are especially version-, build-, model-, and device-dependent.

If broader backend kwargs are added later, targeted fallback should be considered
only for clearly identified unsupported-keyword `TypeError` cases. Avoid catching
all backend failures and retrying with silently reduced options, because that can
hide real model-load or generation errors.

When fallback is implemented:

- Log or warn which key was removed unless `log_level` suppresses it.
- Retry only when the error clearly identifies an unsupported keyword.
- Preserve the original error if fallback also fails.
- Add regression tests for both supported and unsupported backend versions where
  practical.

## History Recording Rules

Applied advanced kwargs should be recorded in each turn's history `params`.

History records should show only values that were actually accepted and passed to
the runtime path. Invalid, missing, `null`, or unsupported keys should not appear
as applied parameters.

When a parameter affects summary generation, record it in a summary-specific
history field or clearly separated summary advanced kwargs entry rather than
making it look like a normal generation parameter.

## Documentation Requirements

When a new advanced parameter becomes supported:

- Update `ADVANCED_PARAMETERS.md` with user-facing usage, examples, limitations,
  and reproducibility notes.
- Update `config/simple_advanced.example.json` if it should appear in the
  example config.
- Update this file if the new parameter changes implementation policy,
  validation policy, fallback behavior, or supported advanced sections.
- Add or update tests proving the Simple config value reaches the intended
  runtime path.

Do not document a parameter as supported in `ADVANCED_PARAMETERS.md` until it is
implemented, validated, and tested.

## Initial Seed-Only Decision

The first implemented advanced Simple config path intentionally supported only
`seed`.

Reasons:

- The maintainer did not need the other advanced parameters at the time.
- Many advanced options are backend-version, model, and device dependent.
- A broad pass-through surface would increase issue triage and support cost.

This remains the preferred expansion pattern: add one narrow, well-tested
parameter or parameter group at a time instead of opening a broad pass-through.
