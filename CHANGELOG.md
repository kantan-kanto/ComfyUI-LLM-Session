# Changelog

All notable changes to ComfyUI-LLM-Session will be documented in this file.

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