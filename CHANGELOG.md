# Changelog

All notable changes to ComfyUI-LLM-Session will be documented in this file.

---

## [1.2.3] - 2026-06-12

- ComfyUI Registry compatibility
  - Removed a plain URL comment from `requirements.txt` to avoid false-positive custom URL dependency flagging.
  - No runtime behavior changes from `1.2.2`.

---

## [1.2.2] - 2026-06-12

- Vision model diagnostics
  - Report the detected model family, required multimodal chat handler, installed `llama-cpp-python` version, and backend version when Vision is required but the handler is unavailable.
  - Distinguish unknown model-family aliases from known families whose handlers are missing in the installed backend.
  - Link users to the upstream JamePeng `llama-cpp-python` project for backend-specific multimodal compatibility and installation guidance, without printing install commands.

- MiniCPM-V compatibility
  - Added `minicpm-v-4.6` / `minicpm-v-4_6` / `minicpmv46` aliases.
  - Mapped MiniCPM-V-4.6 to `MiniCPMV46ChatHandler`.
  - Set the MiniCPM-V-4.6 chat-handler default to `enable_thinking=false` while preserving explicit overrides.
  - Added MiniCPM-V-4.6 Text-only prompt building with `enable_thinking` control and summary forced-off behavior.

- Advanced JSON-based parameter settings
  - Added support for configuring several advanced parameters through Simple-node JSON config files.
  - Listed additional parameters that are present for future expansion but are not active yet.
  - Added `ADVANCED_PARAMETERS.md` to document the advanced JSON-based parameter settings and their current limitations.

- Dependency and compatibility documentation
  - Keep `requirements.txt` on the conservative PyPI-compatible `llama-cpp-python>=0.3.16` dependency while noting that recent JamePeng builds may be needed for newer Vision models.
  - Updated README and compatibility notes to explain that newer multimodal model families may require backend builds with matching chat handlers.

---

## [1.2.1] - 2026-05-31

- Generation controls
  - Raised the Full UI `max_tokens` upper limit from `8192` to `32768` for long-form generation and editing workflows.
  - Documented `max_tokens` and `n_ctx` UI ranges and long-form sizing guidance.

- Session history reliability
  - Report history persistence failures through `TurnExecutionResult` while still returning successful generations.
  - Added warning logs when a response is generated but the session history could not be saved.
  - Log unreadable history files and quarantine corrupt primary history files before creating a fresh history.

- Vision model error handling
  - Stop with an explicit error instead of falling back to text-only mode when image input or an explicitly selected mmproj requires vision support.

- ComfyUI integration
  - Support ComfyUI Cancel during non-stream llama.cpp generation by forwarding interrupts to `llm.abort()`.
  - Removed startup-only debug prints from the llama.cpp import path.

- Registry and test packaging
  - Exclude development-only test files from ComfyUI Registry archives.
  - Added reproducible local test dependencies in `requirements-dev.txt`.

---

## [1.2.0] - 2026-05-17

- Added llama.cpp `tensor_split` support for Simple nodes
  - Added `tensor_split` to `config/simple_defaults.json`
  - Passes normalized split values through model loading, cache signatures, and model signatures
  - Documents common multi-GPU examples in `PARAMETERS.md`

- Added user-facing `enable_thinking` controls to Full UI nodes
  - `LLM Session Chat` and `LLM Dialogue Cycle` now expose `enable_thinking`
  - Supported model families currently include Gemma 4 and the Qwen3.5 / Qwen3.6 compatibility path
  - Full-node defaults now preserve explicit per-model Simple config overrides

- Improved Gemma 4 model-specific behavior
  - Added Gemma 4 text prompt builder support for the `enable_thinking` setting
  - Forces `enable_thinking=false` for Gemma 4 summary generation
  - Warns when Gemma 4 E2B/E4B vision models are loaded with thinking disabled, since those variants may require thinking to be enabled in the JamePeng handler

- Documented model-specific parameter flow
  - Added `docs/model-specific-parameter-flow.md`
  - Documented parameter precedence for model-family-specific overrides
  - Added regression tests for override precedence, Gemma 4 thinking flow, and `tensor_split`

- Bumped package/module version to `1.2.0`

---

## [1.1.2] - 2026-04-21

- Added Step3-VL compatibility aliases and handler mapping
  - Added `step3-vl` / `step3vl` aliases
  - Added chat-format mapping to `Step3VLChatHandler`

- Added Qwen3.6 compatibility aliases
  - Added `qwen3.6` / `qwen3_6` / `qwen36` aliases
  - Mapped Qwen3.6 aliases to existing `qwen3.5` compatibility path

- Adjusted Gemma 4 default thinking behavior
  - Set `gemma4` handler kwargs to `enable_thinking: false`
  - Added `gemma4.enable_thinking: false` to `config/simple_defaults.json` for Simple nodes

- Updated compatibility documentation entries
  - Revised Gemma 4 tested entry to `Gemma 4 (E2B / 31B)*`
  - Added `Qwen3.6 (35B-A3B)*` to tested models

- Bumped package/module version to `1.1.2`

---

## [1.1.1] - 2026-04-11

- Added Gemma 4 alias support
  - Added `gemma4` / `gemma-4` model-family aliases
  - Added `gemma4` chat-format mapping to `Gemma4ChatHandler` for compatibility with current llama-cpp chat handlers
  - Updated mmproj auto-detect family matching to include Gemma 4 aliases

- Added minimal LFM2.5-VL handler wiring
  - Added `lfm2.5-vl` aliases and `LFM25VLChatHandler` mapping
  - `lfm2-vl` / `lfm2.5-vl` are currently not validated in this repository's test environment

- Improved GGUF/mmproj filename discovery robustness
  - GGUF extension matching is now case-insensitive (`.gguf` / `.GGUF`)
  - mmproj prefix matching is now case-insensitive (`mmproj-` / `MMPROJ-`)

- ComfyUI Registry flagged-status mitigation
  - Added a bundled `LICENSE` file so `pyproject.toml` `license = {file = "LICENSE"}` always resolves during packaging/indexing
  - Bumped package/module version to `1.1.1`

---

## [1.1.0] - 2026-03-25

- Added explicit unload control
  - Added a new output node: `Unload LLM Model`
  - You can now release the currently loaded model manually (for example, after keep-loaded cache runs)

- Adjusted model unload behavior for dialogue cache modes
  - `LLM Dialogue Cycle` now keeps model managers loaded when `runtime_cache` is `KV_cache` or `LlamaTrieCache`
  - Non keep-loaded modes continue deterministic unload behavior
  - `GGUFModelManager.unload_model()` cleanup was strengthened to reduce stale runtime/cache state

- Fixed adaptive retry for context overflow errors
  - Context-overflow errors phrased as `context window ... exceed ...` are now detected as retryable ctx errors
  - Dynamic max-token retry can now recover in more overflow wording variants

- Fixed history recovery from backup files
  - `load_history` now falls back to `*.bak` when the primary history JSON is invalid or missing
  - When backup load succeeds, the primary `*.json` is restored and re-saved atomically
  - History loader log verbosity is now aligned with request `log_level`

- Internal quality improvements (no intended breaking changes)
  - Large internal refactor across `core/`, `infra/`, and `services/` layers
  - Expanded pytest coverage to lock behavior during refactoring

---

## [1.0.4] - 2026-03-16

- Revised cache behavior and improved cache safety
  - Changed the default `persistent_cache` from `LlamaDiskCache` to `off`
  - Updated `reset_session` so it clears history and per-session KV state while keeping the session's disk cache
  - Moved disk cache storage to a session-scoped layout under `history_dir/cache/<session_id>/...`, with further separation by model settings
  - Added more detailed diagnostics for cache mismatches and strengthened retry behavior after cache invalidation
  - Updated `history_dir`, `reset_session`, and `persistent_cache` descriptions and tooltips to match the new cache model

- Improved Qwen3.5 generation paths and override handling
  - Added `qwen3.5.enable_thinking` to `config/simple_defaults.json`
  - Introduced text prompt builder overrides so Qwen3.5 text-mode generation can explicitly enable or disable `<think>` output
  - Updated the compatibility documentation to reflect the current Qwen3.5 support status

- Improved backward compatibility for repetition-control parameters
  - Adjusted repetition-penalty argument handling across llama-cpp-python versions by retrying with legacy `repeat_last_n` when the newer `penalty_last_n` path is rejected

- Improved conversation summarization quality and history tracking
  - Reworked the summary prompt to preserve names, roles, constraints, and unresolved items in a more compact and consistent way
  - Strengthened the summary-compaction prompt to avoid guesswork and vague rewrites
  - Added normalization for older histories that lack `covered_until_turn_id`, so summarized ranges can be tracked safely
  - Improved post-processing that strips exposed reasoning such as `<think>` blocks and internal channel markup
  - Added a forced override to suppress Qwen3.5 thinking output during summary generation

---

## [1.0.3] - 2026-03-13

- Improved cache architecture for better safety and performance
  - The former `prompt_cache_mode` / `kv_state_mode` settings were clearly separated and reorganized into `persistent_cache` and `runtime_cache`.
  - Added layered cache support that combines fast in-memory caches (`LlamaRAMCache` / `LlamaTrieCache`) with persistent disk cache (`LlamaDiskCache`), with read-through / write-through behavior.
  - Cache keys now include `n_gpu_layers`, chat format, vision usage state, and `llama-cpp-python` runtime information, allowing incompatible caches to be detected and invalidated automatically for improved stability.
  - Added recovery logic that automatically invalidates KV state and cache, then retries once when `Failed to set llama state data` or `input_ids`-related inconsistencies occur.
  - Improved `reset_session` so that it clears not only history, but also per-session KV state and the cache directory.
  - Improved `LLMDialogueCycleNode` so that, during a single Dialogue Cycle run, models are not unloaded between A/B turns when `runtime_cache` is set to `KV_cache`.
  - If your existing settings or workflows use `prompt_cache_mode` / `kv_state_mode`, they should be updated to `persistent_cache` / `runtime_cache`.
  - The cache storage directory has changed from `prompt_cache/` to `cache/`. Existing cache data is not reused automatically.

- Generalized and expanded multimodal (Vision) model support
  - Introduced a generic mechanism for dynamically loading chat handlers, making it easier to support a wider range of Vision models available in llama-cpp-python, including Qwen, Llava, Moondream2, MiniCPM, Gemma3, and GLM4v families.
  - Improved mmproj (Vision projector) auto-detection to reduce the need for manual path selection on more model setups. If model or mmproj filenames do not follow the expected naming patterns, manual selection may still be required.
  - `LLM Session Chat` and `LLM Dialogue Cycle` now share the same Vision model initialization and cache-application strategy.

- More robust and consistent summarized conversation history management
  - Added sequential `id` values to history turns and changed summarized-range tracking to use `covered_until_turn_id`.
  - Added normalization logic that upgrades older history JSON files to the new schema, improving compatibility with existing sessions. Long-running sessions may therefore show different summarization and context reuse behavior than before.

- Improved output quality and debugging
  - Automatically strips reasoning text emitted by models, such as `<think>...</think>` blocks, from the final output for cleaner responses.
  - Strengthened debug logging (`log_level: debug`) so cache state and prompt content passed to the model can be traced in more detail.

---
## [1.0.2] - 2026-02-13

- Include model configuration fields in KV cache signature to prevent cross-model state reuse.

  - Include model_path, mmproj_path, n_ctx, and n_gpu_layers in hash material
  - Prevent KV state reuse across different models/settings under same session_id
  - Fix false KV cache HIT when switching models or context parameters
  - Apply change to both KV load (HIT check) and save paths across all session nodes

- Add configurable console streaming support across session nodes

  - Added \stream_to_console` as the last UI option for `LLM Session Chat` and `LLM Dialogue Cycle` 
  - Enabled streaming in `LLM Session Chat (Simple)` and `LLM Dialogue Cycle (Simple)` only via JSON config (`"stream_to_console": true`) 
  - Kept default behavior unchanged when streaming is not enabled 
  - Reused existing history/summarization/cache flow while adding streaming generation paths`

- Support GGUF model discovery from extra `LLM` paths via `extra_model_paths.yaml`

  - Add multi-root LLM model path resolution (`models/LLM` + extra `LLM`/`llm` paths)
  - Apply multi-root model/mmproj discovery to Session/Dialogue nodes
  - Keep path traversal protection for resolved model and mmproj paths

- Fix import-time failures for llama-cpp and improve registry compatibility

  - Broaden llama_cpp import guard from ImportError to Exception to prevent crashes from native dependency or shared library issues
  - Store import exception detail in `_LLAMA_CPP_IMPORT_ERROR` for runtime diagnostics
  - Improve availability check message to include captured import error when present
  - Ensure module can be imported safely in Comfy Registry / Manager indexing environments without optional dependencies installed

---

## [1.0.1] - 2026-02-09

- Fixed issue where non-Qwen3-VL vision models were always loaded in text-only mode even when a valid mmproj file was specified.

  - Added vision chat handler support for`Qwen2.5-VL`, `LLaVA`,  `Llama`, `GLM-4.6V`, and `Gemma-3` 
  - Enable vision mode automatically when supported model + mmproj are present

- Improved mmproj auto-detection logic

  - Auto-detect now selects mmproj files based on model family prefix (qwen2, qwen3, llava, llama, gemma-3, glm-4) instead of arbitrary alphabetical fallback
  - Prevents incorrect mmproj selection when multiple mmproj files exist in the same directory

- Added recursive GGUF discovery under `models/LLM`

  - `.gguf` files can now be selected from all subdirectories under `models/LLM`
  - Auto-detection of mmproj continues to work with nested model folders

- Expanded parameter snapshot recording in session history

  - `meta.last_run_params` now stores the most recent execution parameters (overwritten on each run)
  - `turns.params` now stores the parameters used for each individual execution

---

## [1.0.0] - 2026-01-25

- Initial release
