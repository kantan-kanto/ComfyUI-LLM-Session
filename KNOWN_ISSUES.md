# Known Issues

## dynamic_max_tokens does not retry on some n_ctx overflow errors
- Status: Open
- First recorded: 2026-03-21
- Scope: `LLM Session Chat` / `LLM Dialogue Cycle`
- Symptom: Generation fails immediately on the first attempt even when `dynamic_max_tokens=true`.
- Repro (example): `n_ctx=512`, long user input, `max_tokens=400`, `dynamic_max_tokens=true`.
- Observed error: `Requested tokens (...) exceed context window of 512`
- Current behavior: no adaptive retry (no second attempt with reduced `max_tokens`).
- Regression status: Unknown (baseline before refactor was not confirmed).
- Planned handling: Address after P1 refactoring completion as a separate bug-fix task.
- Notes: This is tracked separately from refactoring acceptance criteria.
