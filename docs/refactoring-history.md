# Refactoring History

- Status: Historical
- Last reviewed: 2026-07-05
- Update when: Clarifying historical status, fixing stale cross-references, or adding supersession notes.

Historical note: This document records completed refactoring milestones. For
current refactoring rules, see [`refactoring-rules.md`](refactoring-rules.md).
For current logging and exception-handling policy, see
[`logging-guidelines.md`](logging-guidelines.md).

Counts, file lists, and test totals in this document refer to the repository
state at the time of each milestone and may not match current source files.

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
