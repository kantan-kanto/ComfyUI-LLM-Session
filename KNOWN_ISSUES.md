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

## KV_cache load fails after llama-cpp-python update
- Status: Open
- First recorded: 2026-03-21
- Scope: `LLM Session Chat` (confirmed) / `LLM Dialogue Cycle` (likely same path)
- Symptom: KV cache restore does not succeed and cache state is invalidated.
- Observed log: `[LLM Session Chat] KV state: INVALIDATED (load failed)`
- Current behavior: KV_cache is effectively not working (state is not reused across turns).
- Regression status: Known pre-existing issue before P0 refactoring; no additional worsening observed in current refactoring.
- Planned handling: Track as a separate bug-fix task after refactoring milestones.
