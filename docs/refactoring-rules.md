# Refactoring Rules (Project-Wide)

This document defines the refactoring rules for this repository. It is intended to keep
changes consistent and maintainable over time.

## Core Principles
1. Behavior preservation is the top priority.
2. Spec changes are handled in a separate PR or step.

## Structure & Responsibility
1. Clarify responsibility boundaries.
2. Separate concerns: node definition, UI definition, session processing, and I/O.

## Naming
1. `snake_case` for functions and variables.
2. `PascalCase` for classes.
3. `UPPER_SNAKE` for constants.

## Dependencies
1. Fix the dependency direction: lower layers (I/O, utilities) must not reference higher layers.

## Side Effects
1. Isolate side effects.
2. Access to global state and file I/O should be localized in dedicated helpers/modules.

## Extension Points
1. Make extension points explicit.
2. If you add a new feature, document where to place related code (comments or docs).

## Project-Specific Notes
1. Keep UI definitions thin by using shared UI helper functions (e.g., input builder helpers).
2. Preserve UI enum/choice labels and sentinel strings to avoid breaking saved workflows.
3. Prefer small refactors that keep public node inputs stable unless a separate change is agreed.
4. Keep node execution methods thin (orchestration-focused). If a node method grows beyond roughly 150 lines, move logic to services/helpers.
5. Avoid duplicated core logic. Continue rewrite, generation retry flow, and KV state load/save logic must be implemented once and reused.
6. Preserve compatibility-critical surface by default: node class names, input keys, enum/choice labels, and sentinel strings must stay stable unless a spec-change step is explicitly separated.

## Physical Split Criteria (How to decide move vs keep)
This section defines how to decide whether helper functions should remain in the current file
or be moved to another module. Responsibility separation is a design concept; file split is
an implementation step that must be gated by safety and testability.

### 1) Preconditions to move code to another file
Move a function/group only when all of the following are true:
1. Boundary stability:
   - Inputs/outputs are explicit and small enough to reason about.
   - The function does not rely on broad hidden context from module globals.
2. Dependency direction safety:
   - The destination layer preserves allowed dependency direction.
   - The moved code does not require lower layers to import node/UI layer details.
3. Compatibility safety:
   - The move does not require changing compatibility-critical surface by default
     (node class names, input keys, enum labels, sentinel strings).
4. Test lockability:
   - Behavior can be locked via existing tests or a small additive test in the same step.
   - If behavior cannot be locked, defer move unless there is a critical maintenance reason.

### 2) Reasons to keep code in-place (defer physical split)
Keep code in the current file when one or more apply:
1. Hidden dependency concentration:
   - Strong reliance on module globals, side effects, or execution environment details.
2. High breakage blast radius:
   - Split would likely affect many call paths in one step.
3. Low observability:
   - Failures would be hard to localize after split because test signal is weak.
4. Poor ROI at current stage:
   - Additional split produces little readability/maintainability gain versus risk.

### 3) Prioritization order for physical split
When multiple candidates exist, split in this order:
1. Execution orchestration paths with clear request/dependency boundaries.
2. Reused business logic helpers with stable contracts.
3. UI/input builder helpers.
4. Utility groups with many implicit assumptions (last; only after guardrails improve).

### 4) Stop criteria for a refactoring milestone
A milestone can be considered "done enough" (local optimum) when:
1. Major runtime paths are routed through explicit service/request/dependency boundaries.
2. Regression tests cover the moved orchestration boundaries.
3. Remaining split candidates are mostly high-risk/low-ROI under behavior-preserving constraints.
4. Next highest-value work shifts to bug fixing or behavior decision tasks (tracked separately).

### 5) Decision recording expectations
When proposing additional physical split, contributors should explicitly state:
1. Why the candidate satisfies preconditions in section 1.
2. Which risks in section 2 are present and how they are mitigated.
3. What tests lock current behavior before/after move.
4. Why this split has better ROI than the next unresolved issue.

If these points are unclear, prefer defer + documentation over forced split.

## Notes
- If a potential bug is found but fixing it would change behavior, record it and handle it later.

## Documentation & Architecture
1. Keep folder-role comments in each layer package `__init__.py` (e.g., `core/`, `services/`, `infra/`).
2. Add a short top-of-file purpose comment for extracted modules so intent is clear without cross-reading.
3. Keep a one-screen dependency-direction diagram in `docs/architecture.md`, and keep `docs/refactoring-rules.md` focused on rules.
4. Track unresolved behavior issues in `docs/known-issues.md` (not in refactor commits that preserve behavior).
5. When architecture changes, update both `docs/architecture.md` and affected package/file comments in the same change.

## Line Endings
1. Normalize all text files to LF (`\\n`) and keep `.gitattributes` with `* text=auto eol=lf`.
2. Do not mix functional changes with line-ending normalization in the same commit.
3. If line-ending normalization is needed, perform it in a dedicated commit with a clear message.


## P0 Safety Net Baseline (2026-03-21)
1. Add pytest scaffolding under `tests/`.
2. Lock behavior for `core/continue_rewrite.py`.
3. Lock behavior for `core/kv_state.py`.
4. Lock behavior for `core/generation_runner.py`.
5. Lock orchestration behavior for `services/turn_execution_service.py`.

Note:
- These tests are intended to preserve current behavior during refactoring.
- KNOWN_ISSUES remain tracked separately and are not changed by P0 test commits.

## Error Handling Improvement Project (2026-03-22)

### Completed Phases

- **Phase 2**: Created `core/logging_utils.py` with logging utilities
- **Phase 3**: Improved 4 P0 error handling items with logging
- **Phase 4**: Improved 8 P1 error handling items with logging
- **Phase 5**: Documented 16 P3 acceptable silent errors
- **Phase 6**: Added 8 tests for P0/P1 error handling improvements

### Test Coverage

- Total tests: 73 (all passing)
- P0/P1 error handling tests: 8 (in `TestErrorHandlingP0P1`)
- No regressions from error handling improvements

### Files Modified

- `core/logging_utils.py` (new) - Logging utilities
- `services/turn_execution_service.py` - P0/P1 logging additions
- `core/kv_state.py` - P1 logging addition
- `llm_session_nodes.py` - P1 logging additions
- `tests/services/test_turn_execution_service.py` - P0/P1 tests
- `docs/error-handling-audit.md` - Status update
- `docs/acceptable-silent-errors.md` (new) - P3 documentation


