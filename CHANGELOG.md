# Changelog

All notable changes to ComfyUI-LLM-Session will be documented in this file.

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