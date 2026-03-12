# Changelog

All notable changes to ComfyUI-LLM-Session will be documented in this file.

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
