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

## Notes
- If a potential bug is found but fixing it would change behavior, record it and handle it later.

## Documentation & Architecture
1. Keep folder-role comments in each layer package `__init__.py` (e.g., `core/`, `services/`, `infra/`).
2. Add a short top-of-file purpose comment for extracted modules so intent is clear without cross-reading.
3. Keep a one-screen dependency-direction diagram in `ARCHITECTURE.md`, and keep `REFACTORING_RULES.md` focused on rules.
4. Track unresolved behavior issues in `KNOWN_ISSUES.md` (not in refactor commits that preserve behavior).
5. When architecture changes, update both `ARCHITECTURE.md` and affected package/file comments in the same change.

## Line Endings
1. Normalize all text files to LF (`\\n`) and keep `.gitattributes` with `* text=auto eol=lf`.
2. Do not mix functional changes with line-ending normalization in the same commit.
3. If line-ending normalization is needed, perform it in a dedicated commit with a clear message.

