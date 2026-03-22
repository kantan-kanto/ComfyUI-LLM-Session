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
## LLM Dialogue Cycle unloads model at end of execution
- Status: Resolved (2026-03-22)
- First recorded: 2026-03-21
- Scope: `LLM Dialogue Cycle`
- Previous behavior: model unload was always performed at the end of `chat_cycle` flow.
- Resolution: `runtime_cache` modes `KV_cache` and `LlamaTrieCache` now keep managers loaded across and after cycle execution.
- Notes: Other runtime cache modes continue deterministic unload behavior for memory cleanup.
## History .bak recovery does not restore session in some local cases
- Status: Open
- First recorded: 2026-03-22
- Scope: `LLM Session Chat` (history load path)
- Symptom: Session history is not recovered even when a `.bak` file exists.
- Repro (confirmed):
  1. Use an existing session file (example: `20260207.json`).
  2. Case A: break JSON structure in the primary file (`*.json`) and keep `*.bak`.
  3. Case B: delete the primary file (`*.json`) while `*.bak` exists.
  4. Run `LLM Session Chat` with the same `session_id`.
- Current behavior: `.bak` is not used to restore in the tested local environment.
- Regression status: Unknown (pre-refactor behavior was not conclusively verified).
- Planned handling: Address after refactoring completion as a separate bug-fix task.
- Notes: This issue is tracked separately from behavior-preserving refactor commits.

## Potential Refactoring Candidates (Not Confirmed Bugs)
- Status: Tracking
- First recorded: 2026-03-22
- Scope: Internal structure / maintainability
- Notes: The following items are architectural cleanup candidates, not confirmed defects. They should be handled as behavior-preserving refactors with focused tests.

### Session Chat / Dialogue Cycle request-building overlap
- Location: `llm_session_nodes.py` (`_build_session_chat_node_execution_request`, `_build_dialogue_cycle_node_execution_request`)
- Observation: Similar input-to-request mapping logic exists in parallel paths.
- Caution: The two paths still have intentional differences; over-aggressive unification can accidentally change node behavior.

### Session Chat / Dialogue Cycle dependency-wiring overlap
- Location: `llm_session_nodes.py` (`_build_turn_execution_dependencies`, `_build_dialogue_cycle_dependencies`)
- Observation: Dependency assembly patterns are structurally similar.
- Caution: Role-specific and orchestration-specific dependencies must remain explicit to keep layering clear.

### Simple wrapper parameter mapping overlap
- Location: `llm_session_nodes.py` (simple wrapper kwargs builders and simple node entry points)
- Observation: Multiple simple-wrapper paths perform similar default/override mapping.
- Caution: Public UI compatibility (input keys, defaults, sentinel labels) is compatibility-critical and must be preserved.

