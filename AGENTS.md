# AGENTS.md

This is the required entry point for coding agents working on
ComfyUI-LLM-Session. It routes agents to the repository's canonical guidance;
detailed policy belongs in `docs/`.

## Purpose

ComfyUI-LLM-Session is a ComfyUI custom node project for local LLM session
workflows. Keep changes scoped, preserve established service and module
boundaries, and use pytest as the automated guardrail. Manual ComfyUI checks
remain the final behavior validation for UI and runtime workflows.

## Repository Map

- `llm_session_nodes.py`: ComfyUI node definitions and compatibility-facing
  behavior.
- `core/`: shared runtime, generation, state, logging, and turn helpers.
- `services/`: chat-turn, generation, history, state, and node orchestration.
- `infra/`: persistence and storage helpers.
- `tests/`: pytest coverage for behavior, contracts, and regressions.
- `docs/`: canonical project rules, architecture notes, and change guidance.

## Required Reading and Routing

Before repository-specific analysis, planning, editing, review, or release
work, read `docs/index.md`, then apply every row below that matches the task.
Required outputs are cumulative unless a specialized document defines a more
specific format.

| Task type | Required document | Required output |
| --- | --- | --- |
| Any repository code change | `docs/agent-development-plan.md` | Implementation summary, checks run or skipped, docs impact, and manual validation needs |
| Refactoring | `docs/refactoring-rules.md` | Scope, behavior-preservation notes, and focused tests |
| Tests or behavior verification | `docs/testing-rules.md` | Focused/full pytest status and reasons for skipped checks |
| Logging, exceptions, or silent fallbacks | `docs/logging-guidelines.md` | Error-handling rationale and test status |
| Public behavior, parameters, docs, changelog, commit text, or releases | `docs/change-documentation-guidelines.md` | The documentation and final-response outputs required there |
| Service boundaries, dependency direction, or module placement | `docs/architecture.md` | Placement rationale and affected modules or services |
| Manual ComfyUI validation may be needed | `docs/manual-test-checklist.md` | Relevant checklist sections for a human to run |
| Non-trivial verification planning | `docs/templates/verification-plan.md` | A verification plan or concise equivalent |

Before acting, extract the applicable requirements from the routed documents.
For non-trivial work, record them in a short working checklist. Before the
final response, verify all required outputs, formats, prohibitions, and skipped-
check explanations.

## Cross-Cutting Rules

- Prefer existing patterns, helpers, service boundaries, and dependency
  direction.
- Keep edits scoped to the requested behavior; avoid unrelated refactoring.
- Preserve public node interfaces unless the task intentionally changes them.
- Add or update tests when changed behavior can be expressed with pytest.
- Follow `docs/testing-rules.md` for focused and full-suite test decisions, and
  report every skipped check with its reason.
- Determine documentation impact through `docs/index.md` and
  `docs/change-documentation-guidelines.md`.
- When documentation files are added, removed, or change roles, update
  `docs/index.md` in the same change.
- Do not claim manual ComfyUI validation was performed unless a human actually
  performed it. Identify the relevant sections of
  `docs/manual-test-checklist.md` when manual validation is needed.
- Report remaining limitations and risks clearly.

Do not copy detailed policy into this file. When a rule changes, update its
canonical document in `docs/`; change this file only when routing or a genuinely
cross-cutting rule changes.

## Definition of Done

A change is ready to report when:

- The requested work is complete and scoped.
- Applicable automated checks pass, or skipped checks and their reasons are
  reported.
- Documentation impact has been reviewed.
- Required human validation is identified and is not reported as completed
  unless it was actually performed.
- All routed-document output requirements are satisfied.
- Remaining limitations or risks are stated.
