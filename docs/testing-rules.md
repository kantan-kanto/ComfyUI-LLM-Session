# Testing Rules

- Status: Canonical
- Last reviewed: 2026-07-05
- Update when: Test structure, coverage expectations, fixture policy, or
  regression-test rules change.

This document defines repository-wide testing rules for ComfyUI-LLM-Session.
Use it together with domain-specific test requirements in documents such as
`model-specific-parameter-flow.md` and `advanced-simple-config-notes.md`.

## Core Principles

1. Behavior preservation is the first testing priority.
2. Tests should lock user-visible behavior and compatibility-critical surfaces.
3. Service boundaries should have direct tests, not only indirect node-level coverage.
4. Refactors should be supported by existing tests or small additive tests in the same step.

## When Tests Are Required

Add or update tests when a change affects any of the following:

1. `services/*.py` classes, request objects, dependency wiring, or orchestration behavior.
2. Cache, history, session, KV state, model unload, transcript, or filesystem side effects.
3. Model-family detection, model-specific parameter flow, chat handler kwargs, or backend fallback.
4. UI input defaults, enum/choice labels, sentinel strings, or Simple-node defaults.
5. Bug fixes, especially when the broken condition can regress silently.

For service-layer changes, prefer a direct test in `tests/services/` even when a
higher-level node test already exercises the path.

## Test Placement

1. Put `core/` tests under `tests/core/`.
2. Put `services/` tests under `tests/services/`.
3. Put `infra/` tests under `tests/infra/`.
4. Keep `llm_session_nodes.py` UI/default/import/node-wrapper tests in focused
   top-level `tests/test_*.py` files.
5. For model-specific parameter behavior, also follow the Required Tests section
   in `model-specific-parameter-flow.md`.
6. For advanced Simple config behavior, also follow the documentation and test
   requirements in `advanced-simple-config-notes.md`.

## Side Effects

Side-effecting behavior should cover both success and failure paths when practical.

Examples include:

1. History save, backup, restore, quarantine, and schema migration.
2. KV state save, restore, invalidation, and cache mismatch handling.
3. Runtime cache configuration and invalidation.
4. Model unload and runtime container cleanup.
5. Transcript append and output file writes.

When a failure is intentionally tolerated, assert the returned status, recorded
error, log output, or preserved state that makes the tolerance observable.

## Fixtures And Helpers

1. Do not copy the same test helper across many files.
2. If the same helper, stub, or import setup is needed in three or more test
   files, move it to `tests/conftest.py` or a focused test helper module.
3. Keep `folder_paths` stubs, `llm_session_nodes` reload setup, and runtime
   container stubs centralized when they are shared.
4. Large dependency dictionaries should use a default builder plus explicit
   overrides instead of repeating full dictionaries in each test.
5. Test helpers should keep defaults boring and make the behavior under test
   obvious at the call site.

## Parametrized Tests

Use `pytest.mark.parametrize` when the same assertion shape covers three or more
cases.

Good candidates include:

1. Model family aliases and variant names.
2. UI input defaults and option lists.
3. Language-specific continue-rewrite behavior.
4. Supported and unsupported media/input shapes.
5. Error-message or backend-fallback cases with the same expected outcome.

Do not force a regression test into a parameter table when a named standalone
test better explains the broken condition.

## Large Test Files

Avoid growing already-large test files by default.

1. When a test file is roughly 500 lines or larger, consider placing new tests in
   a focused sibling file.
2. Split new turn-execution coverage by concern when possible: preflight,
   generation, persistence, KV state, node execution, or timing/logging.
3. Existing large files do not need immediate mechanical splitting. Prefer
   incremental extraction when new tests or helper cleanup make the boundary clear.

## Testing Private Helpers

Prefer public functions, service methods, and request/dependency contracts as the
main test surface.

Direct private-helper tests are acceptable when one of these applies:

1. The helper handles path safety, media conversion, model alias detection,
   config/default normalization, or backend compatibility fallback.
2. The helper has many boundary conditions that would be noisy to reach through a
   higher-level API.
3. The higher-level API would require heavy mocks or unrelated setup that hides
   the behavior being tested.

## Regression Test Names

Regression tests should name the condition that used to fail.

Prefer names like:

1. `test_load_history_reads_backup_when_primary_invalid`
2. `test_restore_state_invalidates_cache_on_state_data_mismatch`
3. `test_simple_config_override_preserves_model_specific_default`

Avoid vague names like `test_history_fix` or `test_kv_state_works`.

## Pytest Configuration

Keep pytest configuration strict enough to catch stale test metadata.

Recommended defaults:

1. Use `--strict-config`.
2. Use `--strict-markers`.
3. Register any marker before using it.
4. Introduce coverage measurement before introducing a hard coverage threshold.
5. Raise coverage thresholds gradually after currently-thin service areas are
   covered directly.

## Checklist

Before finishing a code change, check:

1. Did service-layer behavior get a direct test?
2. Did side-effect behavior cover success and failure paths where practical?
3. Did repeated cases use parametrization where appropriate?
4. Did new helpers avoid copy-paste across test files?
5. Did regression tests describe the broken condition in the test name?
6. Did domain-specific docs add any extra test requirements for this change?

## Related Documents

1. [`refactoring-rules.md`](refactoring-rules.md) - Refactoring safety-net and
   split criteria.
2. [`model-specific-parameter-flow.md`](model-specific-parameter-flow.md) -
   Required tests for model-specific parameter changes.
3. [`advanced-simple-config-notes.md`](advanced-simple-config-notes.md) -
   Advanced Simple config implementation and test requirements.
4. [`change-documentation-guidelines.md`](change-documentation-guidelines.md) -
   Documentation, changelog, and release-note placement rules.
