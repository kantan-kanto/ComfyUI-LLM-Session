# Verification Plan Template

Use this template for non-trivial changes and for any change where verification
scope is easy to miss. Keep completed plans concise, and replace placeholder
text with the actual commands, results, and decisions for the change.

For testing policy, see `../testing-rules.md`. For documentation policy, see
`../change-documentation-guidelines.md`.

## Change Summary

- Change:
- Behavior affected:
- Files or modules affected:
- Risk level: low / medium / high

## Automated Tests

List the pytest checks that should be run for this change.

- [ ] Focused pytest command:
  ```powershell
  python -m pytest
  ```
- [ ] Full pytest command, when practical:
  ```powershell
  python -m pytest -q
  ```
- [ ] All relevant tests passed.

Skipped automated checks and reason:

- None / reason:

## Regression Tests

Use this section for bug fixes or behavior that should not regress.

- [ ] A regression test is required.
- [ ] A regression test was added or updated.
- [ ] No regression test is needed.

Regression coverage:

- Test file:
- Scenario protected:
- Why this would have failed before, if applicable:

If no regression test was added, explain why:

- Reason:

## Contract Tests

Use this section when public interfaces, node contracts, defaults, mappings, or
service request/response shapes may change.

- [ ] Node `INPUT_TYPES` reviewed.
- [ ] Node `RETURN_TYPES` / `RETURN_NAMES` reviewed.
- [ ] Node mappings reviewed.
- [ ] Default values reviewed.
- [ ] Service dataclass or request/response shape reviewed.
- [ ] Model-specific parameter routing reviewed.
- [ ] Contract tests were added or updated where needed.
- [ ] No contract surface changed.

Contract coverage:

- Test file:
- Contract protected:

## Manual ComfyUI Tests

Use this section for behavior pytest cannot fully prove. Do not mark manual
validation as complete unless it was actually performed in ComfyUI.

- [ ] Manual ComfyUI validation is required.
- [ ] Manual ComfyUI validation is not required for this change.

Recommended checks, when relevant:

- [ ] Node appears in the ComfyUI UI.
- [ ] Basic Session Chat generation works.
- [ ] Dialogue Cycle flow works.
- [ ] Continue behavior works.
- [ ] Summary behavior works.
- [ ] KV cache behavior works.
- [ ] GGUF discovery works.
- [ ] Image-capable model path works, when available.
- [ ] User-visible errors are understandable.

Manual validation result:

- Relevant `docs/manual-test-checklist.md` sections:
  - Section:
    Reason:
- Performed by:
- Environment:
- Result:

Manual validation not performed:

- Reason:
- Remaining risk:

## Documentation Checks

Review documentation impact before completion. Update files only when the change
requires it.

- [ ] `README.md` reviewed.
- [ ] `README.ja.md` reviewed.
- [ ] `CHANGELOG.md` reviewed.
- [ ] `docs/architecture.md` reviewed.
- [ ] `docs/testing-rules.md` reviewed.
- [ ] `docs/refactoring-rules.md` reviewed.
- [ ] `docs/logging-guidelines.md` reviewed.
- [ ] `docs/change-documentation-guidelines.md` reviewed.
- [ ] No documentation update is required.

Documentation changes:

- Updated:
- Not updated because:

## Out of Scope

List checks, platforms, model types, or workflows intentionally not covered by
this verification plan.

- Out of scope:
- Reason:

## Result Summary

- Focused tests:
- Full test suite:
- Regression coverage:
- Contract coverage:
- Manual ComfyUI validation:
- Documentation:
- Remaining risks:
