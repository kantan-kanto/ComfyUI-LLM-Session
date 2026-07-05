# Documentation Index

- Status: Canonical
- Last reviewed: 2026-07-05
- Update when: Files are added to or removed from `docs/`, or document roles change.

This directory contains architecture notes, implementation rules, and historical
records for ComfyUI-LLM-Session. Use this index to find the current source of
truth before changing code or documentation.

## Start Here

- [`architecture.md`](architecture.md) - Current module boundaries, top-level
  directory roles, and dependency direction.
- [`refactoring-rules.md`](refactoring-rules.md) - Current behavior-preserving
  refactoring rules, split criteria, and safety expectations.
- [`known-issues.md`](known-issues.md) - Current issue index. Check `Open`
  entries before making behavior changes.
- [`change-documentation-guidelines.md`](change-documentation-guidelines.md) -
  Rules for updating docs, README content, changelog entries, and release notes.

## Canonical Reference Docs

- [`advanced-simple-config-notes.md`](advanced-simple-config-notes.md) -
  Advanced Simple config behavior and implementation rules.
- [`logging-guidelines.md`](logging-guidelines.md) - Current logging,
  exception-handling, and silent-handler policy.
- [`model-specific-parameter-flow.md`](model-specific-parameter-flow.md) -
  Model-family parameter precedence and update checklist.
- [`testing-rules.md`](testing-rules.md) - Repository-wide test placement,
  fixture, parametrization, and regression-test rules.

## Historical Records

Historical documents preserve past review context. They are not current policy
unless a canonical document explicitly says so.

- [`acceptable-silent-errors.md`](acceptable-silent-errors.md) - Historical
  2026-03-22 silent-error allowlist.
- [`error-handling-audit.md`](error-handling-audit.md) - Historical 2026-03-22
  audit of broad exception handling.
- [`refactoring-history.md`](refactoring-history.md) - Completed refactoring
  milestones and related historical notes.

## Maintenance Rule of Thumb

When a change touches module boundaries, logging, model-specific parameters,
history behavior, tests, known issues, or release-facing behavior, check this
index first and update the affected canonical document in the same change.
