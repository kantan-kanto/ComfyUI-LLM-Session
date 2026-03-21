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

## Notes
- If a potential bug is found but fixing it would change behavior, record it and handle it later.
