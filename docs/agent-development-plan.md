# Agent Development Plan

- Status: Proposed
- Source: Adapted from `experiments/plan.md`
- Scope: Codex Extension and similar coding agents working on this repository.
- Update when: Agent workflow, testing policy, documentation routing, or release verification expectations change.

This document defines the repository-level plan for making AI-assisted
development repeatable, reviewable, and safe for ComfyUI-LLM-Session.

The goal is not to replace human ComfyUI validation. The goal is to make every
agent-assisted change arrive with enough repository context, focused tests, and
documentation discipline that manual ComfyUI checks can focus on behavior that
pytest cannot prove.

## Objectives

- Improve implementation quality for agent-assisted changes.
- Avoid repeating the same project instructions in every prompt.
- Give coding agents a stable entry point for repository rules.
- Maximize the behavior covered by pytest.
- Keep the boundary between automated pytest checks and manual ComfyUI checks
  explicit.
- Preserve project knowledge in tracked documentation instead of ignored
  experiment files.

## Operating Model

Agents are expected to:

- Understand the repository structure before editing.
- Prepare or infer an implementation plan for non-trivial changes.
- Keep edits scoped to the requested behavior.
- Add or update tests when behavior changes.
- Run focused pytest checks, and preferably the full suite before completion.
- Fix failing tests caused by the change before reporting completion.
- State any manual ComfyUI validation that remains out of scope for the agent.

Manual ComfyUI validation remains the final quality gate for UI/runtime behavior
that cannot be fully exercised in pytest.

In short:

```text
pytest = automated guardrail
ComfyUI manual validation = final behavior check
```

## Current Repository Context

The repository already has several durable documents that should be reused
instead of duplicated:

- `docs/index.md`: documentation map and routing.
- `docs/testing-rules.md`: canonical testing guidance.
- `docs/refactoring-rules.md`: refactoring boundaries and stop criteria.
- `docs/logging-guidelines.md`: canonical logging and exception-handling rules.
- `docs/change-documentation-guidelines.md`: documentation update expectations.
- `docs/architecture.md`: service/module architecture overview.

The repository also already has a meaningful pytest suite. Because of that, this
plan does not require an immediate physical test directory migration. The safer
near-term path is to strengthen naming, coverage, and documentation around the
existing `tests/core`, `tests/services`, and root-level focused tests.

## Priority 1: Stable Agent Entry Point

### 1. Add `AGENTS.md`

Status: Done.

Create a tracked `AGENTS.md` at the repository root. This should be the first
file a coding agent reads before making changes.

`AGENTS.md` should stay concise and route agents to the detailed docs instead of
copying all rules inline.

Recommended contents:

- Repository purpose and major modules.
- Required document reading order.
- Implementation planning expectations.
- Testing expectations.
- Documentation update expectations.
- Definition of Done.
- Manual ComfyUI validation boundary.

`AGENTS.md` should complement `docs/index.md`; it should not replace the docs
index. The intended split is:

- `AGENTS.md`: mandatory agent operating rules.
- `docs/index.md`: map of maintained project documentation.

### 2. Define Documentation Routing

Status: Done.

Use `docs/index.md` as the human-readable documentation map, and make
`AGENTS.md` the agent-readable entry point.

The expected routing is:

- General code change: read `AGENTS.md`, then `docs/index.md`.
- Refactoring: also read `docs/refactoring-rules.md`.
- Tests: also read `docs/testing-rules.md`.
- Logging or exception handling: also read `docs/logging-guidelines.md`.
- Public behavior, parameters, or release notes: also read
  `docs/change-documentation-guidelines.md`.
- Architecture-sensitive work: also read `docs/architecture.md`.

### 3. Add an Implementation Plan Template

Status: Not started.

Add `docs/templates/implementation-plan.md` only if repeated planning work
starts to drift. Until then, a short plan in the task thread is acceptable.

The template should include:

- Goal
- Design
- Impact
- Risks
- Verification Plan
- Docs Compliance
- Definition of Done

The template should be lightweight. It should help agents think clearly without
turning small fixes into paperwork.

### 4. Add a Verification Plan Template

Status: Done.

Add `docs/templates/verification-plan.md`.

This is higher priority than the implementation template because verification
quality directly affects whether agent-assisted changes are safe to merge.

Recommended sections:

- Automated Tests
- Regression Tests
- Contract Tests
- Manual ComfyUI Tests
- Out of Scope

## Priority 2: Testing Discipline

### 5. Clarify the Role of Pytest

Status: Done.

`docs/testing-rules.md` is the canonical place for pytest policy.

Pytest should cover:

- Pure functions and data transformations.
- Defaults and parameter contracts.
- Model family detection and model-specific behavior.
- Message construction.
- Runtime container wiring.
- Service-level orchestration.
- Regression cases for previously fixed bugs.

Pytest should not pretend to fully replace:

- ComfyUI node graph execution in a real UI session.
- GPU/runtime-specific behavior.
- llama.cpp backend compatibility across all local installations.
- Visual inspection or user workflow validation.

### 6. Preserve Current Test Layout for Now

Status: Done.

Do not immediately move tests into `tests/unit`, `tests/integration`,
`tests/regression`, and `tests/contract`.

The current layout already has useful locality:

- `tests/core`
- `tests/services`
- focused root-level tests such as defaults, model family aliases, media input,
  runtime container integration, and tensor split behavior

For now, improve test intent through:

- Clear test names.
- Regression-oriented names for bug fixes.
- Focused helper modules when setup becomes repetitive.
- References in `docs/testing-rules.md`.

Physical directories such as `tests/regression` or `tests/contract` should be
introduced only when there are enough tests of that type to make the new
structure easier to navigate.

### 7. Prioritize Regression Tests

Status: Ongoing.

When fixing a bug, add a regression test whenever the behavior can be expressed
without launching ComfyUI.

Regression tests should prefer names that describe the behavior being preserved,
for example:

```text
test_summary_override_does_not_leak_to_normal_generation
test_model_family_alias_accepts_known_variants
test_kv_state_is_not_saved_when_generation_fails
```

### 8. Expand Contract Tests Carefully

Status: Ongoing.

Contract tests are valuable for this project because ComfyUI node classes expose
structured interfaces that can break even when implementation details still
work.

Useful contract test targets include:

- `INPUT_TYPES`
- `RETURN_TYPES`
- `RETURN_NAMES`
- node class mappings
- default values shared between simple/full nodes
- service request/response dataclasses
- model-specific parameter routing

Contract tests should avoid overfitting to private implementation details unless
the private helper is intentionally stable and covered by `docs/testing-rules.md`.

## Priority 3: Manual Validation and Automation

### 9. Add a Manual ComfyUI Test Checklist

Status: Done.

Add `docs/manual-test-checklist.md` after the agent entry point and verification
template are in place.

The checklist should focus on behavior pytest cannot fully prove:

- Node appears in ComfyUI.
- Basic Session Chat generation works.
- Dialogue Cycle flow works.
- Continue/summary behavior works.
- KV cache behavior works.
- GGUF discovery works.
- Image-capable model path works when available.
- Errors are understandable in the UI/log output.

### 10. Add a Docs Compliance Checklist

Status: Partially done.

This can be a standalone document or a section of the implementation/verification
template. A documentation check section now exists in
`docs/templates/verification-plan.md`; a standalone checklist has not been added.

It should ask whether the change requires updates to:

- `README.md`
- `README.ja.md`
- `PARAMETERS.md`, if present or reintroduced
- `CHANGELOG.md`
- `docs/architecture.md`
- `docs/testing-rules.md`
- `docs/refactoring-rules.md`
- `docs/logging-guidelines.md`
- `docs/change-documentation-guidelines.md`

### 11. Defer `agent-check.ps1`

Status: Deferred.

An `agent-check.ps1` helper may be useful later, but it should come after the
workflow is stable.

Possible future checks:

- Run pytest.
- Check that `AGENTS.md` exists.
- Check that docs required by `docs/index.md` exist.
- Check obvious stale references.
- Check whether changed public behavior has a changelog or documentation update.

Do not add this script before the manual process has settled. A premature script
can make the workflow look automated while still missing the important review
questions.

Before starting `agent-check.ps1`, complete the following:

- Use `AGENTS.md` in several real coding tasks and note repeated omissions.
- Use `docs/templates/verification-plan.md` for changes that need explicit
  verification planning.
- Use `docs/manual-test-checklist.md` for at least one manual ComfyUI validation
  flow and confirm the routing matrix is practical.
- Decide whether docs compliance belongs only in the verification template or
  also needs a standalone checklist.
- Identify checks that are deterministic enough to automate, such as required
  file existence, docs index coverage, stale references, and pytest execution.
- Keep manual ComfyUI validation as a reported human activity; do not try to
  mark it passed from a script.

### 12. Define the Definition of Done

Status: Done.

Add the final Definition of Done to `AGENTS.md`, then mirror or reference it from
templates.

Recommended baseline:

- Implementation is complete and scoped.
- Relevant tests are added or updated.
- Focused pytest checks pass.
- Full pytest suite passes when practical.
- Documentation impact is reviewed.
- Manual ComfyUI validation needs are stated.
- Known limitations or out-of-scope items are reported.

## Future Work

### Strengthen the Test Harness

Possible future harness improvements:

- lightweight fake manager
- fake llama model
- fake ComfyUI node execution context
- reusable service fixtures
- test utilities for media inputs and KV state

These should be introduced incrementally when they reduce duplication or make
high-value behavior testable.

### Improve Agent Knowledge Organization

As the repository grows, consider keeping `AGENTS.md` short and moving deeper
agent guidance into focused docs. The rule should be:

- `AGENTS.md` tells agents what to read and what must be true before completion.
- `docs/*` explains the detailed policy.

## Recommended Execution Order

1. Create `AGENTS.md`.
2. Create `docs/templates/verification-plan.md`.
3. Add the Definition of Done to `AGENTS.md`.
4. Add `docs/manual-test-checklist.md`.
5. Add or extend docs compliance guidance.
6. Revisit whether an implementation template is needed.
7. Revisit whether `agent-check.ps1` is worth adding.
8. Revisit physical test directory splitting only if the current layout becomes
   hard to navigate.

## Final Target State

The desired workflow is:

1. The agent reads `AGENTS.md`.
2. The agent follows documentation routing through `docs/index.md`.
3. The agent prepares an appropriately sized implementation plan.
4. The agent implements the change.
5. The agent adds or updates relevant tests.
6. The agent runs pytest.
7. The agent reviews documentation impact.
8. The human performs any necessary ComfyUI validation.
9. The change is reported with test results, documentation notes, and remaining
   manual validation needs.

This keeps agent-assisted development fast without making it casual.
