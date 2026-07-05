# AGENTS.md

This file is the required entry point for coding agents working on
ComfyUI-LLM-Session. Keep it short: detailed policy belongs in `docs/`.

## Purpose

ComfyUI-LLM-Session is a ComfyUI custom node project for local LLM session
workflows. Agent-assisted changes should be small, tested, and consistent with
the repository's existing service/module boundaries.

Agents are expected to improve the repository without replacing human ComfyUI
validation. Treat pytest as the automated guardrail and manual ComfyUI checks as
the final behavior validation for UI/runtime workflows.

## Repository Overview

- `llm_session_nodes.py`: ComfyUI node definitions and compatibility-facing
  behavior.
- `core/`: shared defaults, runtime container, generation helpers, KV-state
  helpers, logging utilities, and turn types.
- `services/`: orchestration services for chat turns, generation execution,
  history persistence, KV-state handling, and node execution flows.
- `infra/`: persistence/storage helpers.
- `tests/`: pytest coverage for core helpers, services, contracts, defaults,
  model-specific behavior, and regressions.
- `docs/`: maintained project rules, architecture notes, and change guidance.

## Required Reading

Before making changes, read `docs/index.md` and then follow the relevant routing
below.

- General agent workflow: `docs/agent-development-plan.md`
- Refactoring: `docs/refactoring-rules.md`
- Tests or behavior verification: `docs/testing-rules.md`
- Logging, exception handling, or silent fallbacks:
  `docs/logging-guidelines.md`
- Public behavior, parameters, README, changelog, or docs changes:
  `docs/change-documentation-guidelines.md`
- Service boundaries, dependency direction, or module placement:
  `docs/architecture.md`

Do not duplicate detailed policy here. Update the relevant document in `docs/`
when a rule changes.

## Working Rules

- Prefer existing patterns, helpers, and service boundaries.
- Keep edits scoped to the requested behavior.
- Avoid broad refactors unless the task explicitly asks for them.
- Preserve public node interfaces unless the change intentionally updates them.
- Add or update tests for behavior changes whenever pytest can express the
  expected behavior.
- Do not claim manual ComfyUI validation was performed unless it actually was.
- Report skipped checks, manual validation needs, and residual risks clearly.

## Implementation Planning

For small fixes, a short implementation note is enough. For non-trivial changes,
identify:

- Goal
- Files or modules likely to change
- Expected behavior impact
- Test plan
- Documentation impact
- Manual ComfyUI validation needs

Use `docs/agent-development-plan.md` as the source for the longer-term agent
workflow plan.

## Testing Expectations

- Run focused tests for the changed area.
- Run the full pytest suite when practical.
- Add regression tests for bug fixes when the behavior can be reproduced without
  launching ComfyUI.
- Add contract tests when node interfaces, defaults, mappings, or service
  request/response shapes change.
- Keep the current test layout unless a separate change explicitly introduces a
  new structure.

Current preferred full-suite command:

```powershell
python -m pytest -q
```

Use the repository's active Python environment when one is required by the local
setup.

## Documentation Expectations

Check documentation impact before finishing. Depending on the change, update or
explicitly leave unchanged:

- `README.md`
- `README.ja.md`
- `CHANGELOG.md`
- `docs/architecture.md`
- `docs/testing-rules.md`
- `docs/refactoring-rules.md`
- `docs/logging-guidelines.md`
- `docs/change-documentation-guidelines.md`

Follow `docs/change-documentation-guidelines.md` for public-facing behavior and
release-note decisions.

When proposing commit messages, follow the commit message rules in
`docs/change-documentation-guidelines.md`:

- Prefer `area: Imperative summary`.
- Use one of the documented areas such as `docs`, `compat`, `runtime`, `ui`,
  `tests`, or `release` when it adds useful context.
- Keep the subject concise, ideally around 72 characters or less.
- Do not end the subject line with a period.
- Add a short body only when it clarifies the grouped changes.

## Manual ComfyUI Validation

Some behavior cannot be fully validated by pytest. Call this out when relevant,
especially for:

- Node appearance and UI wiring.
- Real ComfyUI graph execution.
- GPU/backend-specific llama.cpp behavior.
- Image-capable model paths.
- User-visible error presentation.
- End-to-end session, continue, summary, or KV-cache workflows.

When manual validation is needed, tell the human which sections of
`docs/manual-test-checklist.md` to run. Do not mark those checks as passed unless
a human actually performed them.

## Definition of Done

A change is ready to report when:

- The implementation is complete and scoped.
- Relevant tests were added or updated, or the reason for not doing so is clear.
- Focused pytest checks pass.
- The full pytest suite was run when practical, or the reason it was skipped is
  stated.
- Documentation impact was reviewed.
- Manual ComfyUI validation needs are stated, including the relevant
  `docs/manual-test-checklist.md` sections when manual validation is needed.
- Remaining limitations or risks are reported.
