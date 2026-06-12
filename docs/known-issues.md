# Known Issues

## dynamic_max_tokens does not retry on some n_ctx overflow errors
- Status: Resolved (2026-03-22)
- First recorded: 2026-03-21
- Scope: `LLM Session Chat` / `LLM Dialogue Cycle`
- Symptom: Generation failed immediately on the first attempt even when `dynamic_max_tokens=true`.
- Repro (example): `n_ctx=512`, long user input, `max_tokens=400`, `dynamic_max_tokens=true`.
- Observed error: `Requested tokens (...) exceed context window of 512`
- Root cause: Context-overflow detection only matched `n_ctx`-style wording and did not classify `context window` overflow messages as retryable context errors.
- Resolution: Expanded `GenerationExecutionService._is_ctx_error()` matching to include `context window` + `exceed` wording, so adaptive retry now triggers for this error family.
- Regression status: Unknown (baseline before refactor was not confirmed).
- Notes: This issue was tracked separately from refactoring acceptance criteria.
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
- Status: Resolved (2026-03-25)
- First recorded: 2026-03-22
- Scope: `LLM Session Chat` (history load path)
- Symptom: Session history is not recovered even when a `.bak` file exists.
- Repro (confirmed):
  1. Use an existing session file (example: `20260207.json`).
  2. Case A: break JSON structure in the primary file (`*.json`) and keep `*.bak`.
  3. Case B: delete the primary file (`*.json`) while `*.bak` exists.
  4. Run `LLM Session Chat` with the same `session_id`.
- Resolution:
  `load_history` now falls back to `*.bak` when the primary history file is invalid or missing, then restores and re-saves the primary `*.json` file atomically.
  Added regression tests for both repro cases:
  - primary JSON is invalid but `.bak` exists
  - primary JSON is missing and `.bak` exists
- Regression status: Fixed with focused tests in `tests/infra/test_history_store.py`.
- Notes: This issue is tracked separately from behavior-preserving refactor commits.

## Simple config model-specific overrides can be overwritten by Full-node defaults
- Status: Resolved (2026-05-17)
- First recorded: 2026-05-17
- Scope: `LLM Session Chat (Simple)` / `LLM Dialogue Cycle (Simple)` model-specific override wiring
- Symptom: A model-specific setting loaded from `config/simple_defaults.json` could be replaced when the Simple wrapper delegated to the Full node.
- Example: `gemma4.enable_thinking=true` loaded from Simple config could be overwritten by the Full-node default `enable_thinking=false`.
- Root cause: The merge helper used assignment after copying explicit overrides, so the Full-node default value replaced an already present per-model override.
- Resolution: UI-default merge now preserves explicit per-model override values and only fills missing keys.
- Regression status: Covered by `test_model_specific_config_override_wins_over_full_node_default`.
- Notes: Future model-specific parameters added through `CHAT_HANDLER_KWARGS_MAP`, `TEXT_CHAT_BUILDER_CONFIG_MAP`, or `SUMMARY_TEXT_CHAT_BUILDER_FORCE_MAP` must follow the precedence documented in `docs/architecture.md`.

## Summary updated_at can be newer than history meta saved_at
- Status: Open
- First recorded: 2026-06-12
- Scope: `LLM Session Chat` / Simple config summary generation
- Symptom: A saved history file can show `summary.updated_at` later than `meta.updated_at` and `meta.last_params.saved_at`.
- Observed case: With `max_turns=1`, `summary_chunk_turns=1`, `runtime_cache="off"`, and `advanced_summary_generation_kwargs.seed` set, summary generation completed after the history meta timestamps were written.
- Impact: Low. Summary content and `covered_until_turn_id` are saved correctly; the mismatch appears limited to metadata timestamp consistency.
- Expected behavior: When summary text is generated or compacted, history-level metadata timestamps should reflect the final persisted state.
- Planned handling: Revisit the history persistence order so summary generation and final metadata timestamp updates happen in a consistent sequence.
- Regression status: Not covered yet.

## Gemma4 text-only output can enter channel/thought token sequences
- Status: Open
- First recorded: 2026-06-12
- Scope: `LLM Session Chat` text-only generation with Gemma4-family GGUF models
- Symptom: Gemma4 text-only output can start with or enter channel/thought token sequences even when `enable_thinking=false`.
- Observed model: `gemma-4-26B-A4B-it-heretic.Q6_K.gguf`
- Observed setup: text-only mode, `runtime_cache="off"`, `stream_to_console=true`, Simple config testing.
- Observed progression across runs:
  - test2 turn1 began with `<|channel>thought` followed by `<channel|>`.
  - test2 turn2 began with repeated `<start_of_turn>model` fragments and later `_thought` / `<channel|>` fragments.
  - test3 began with `thought` followed by repeated channel delimiters.
  - test4 began with `thought` followed by repeated `<channel|>` delimiters.
  - test5 began with `<|channel>` and saved reasoning-style English planning text as the assistant response.
- Impact: Medium. Some runs still produce usable final Japanese text, but other runs can expose or persist intermediate reasoning/channel text instead of the requested final answer.
- Notes: This was observed while testing advanced Simple config seed behavior. The seed/null-seed behavior itself was correct; this issue appears specific to Gemma4 text-only output behavior.
- Regression status: Not covered yet.

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
