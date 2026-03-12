# Changelog

All notable changes to ComfyUI-LLM-Session will be documented in this file.

---

## [1.0.3] - 2026-03-13

- cache 機構の再設計による安全性とパフォーマンスの向上
  - 従来の `prompt_cache_mode` / `kv_state_mode` を、ディスク等に保存されるpersistent_cache（永続キャッシュ）と、メモリ上で動作するruntime_cache（ランタイムキャッシュ）に明確に分離・再編しました。
  - 高速なメモリキャッシュ（`LlamaRAMCache` / `LlamaTrieCache`）と永続的なディスクキャッシュ（`LlamaDiskCache` ）を組み合わせた階層型キャッシュ機能を導入し、read-through / write-through動作に対応しました。
  - cache キーに `n_gpu_layers`、chat format、vision 利用状態、`llama-cpp-python` の runtime 情報を含め、キャッシュの互換性を自動で検知・無効化する仕組みを追加し、動作の安定性を大幅に高めました。
  - `Failed to set llama state data` や `input_ids` 関連の不整合発生時に、KV state と cache を自動的に無効化して再試行する復旧処理を追加。
  - `reset_session` 実行時に、history のみではなく、セッション単位の KV state と cache ディレクトリの両方を確実にクリアするよう改善。
  - LLMDialogueCycleNodeの動作を、1 回の Dialogue Cycle 実行中は、KV_cache 利用時に A/B 間でモデルをアンロードしないよう改善しました。
  - 既存の設定や workflow で `prompt_cache_mode` / `kv_state_mode` を使用している場合は、`persistent_cache` / `runtime_cache` へ読み替えが必要です。
　- cache 保存ディレクトリは `prompt_cache/` から `cache/` へ変更されました。旧 cache はそのままでは再利用されません。

- マルチモーダル（Vision）モデル対応の汎用化と拡張
  - llama-cpp-pythonがサポートする多様なVisionモデル（Qwen, Llava, Moondream2, MiniCPM, Gemma3, GLM4v等）に柔軟に対応するため、チャットハンドラを動的に読み込む汎用的な仕組みを導入しました。
  - mmproj（Visionプロジェクター）ファイルの自動検出ロジックを改善し、より多くのモデルでユーザーが手動でパスを指定する手間を削減しました。モデル名や mmproj 名が命名規則から外れている場合は、手動指定が必要になることがあります。
  - LLM Session Chat / LLM Dialogue Cycle で、同じ vision モデル初期化・cache 適用方針を共有。

- 要約付き対話履歴（History）管理の堅牢化、一貫性向上
  - 履歴 turn に連番 `id` を付与し、要約がどこまで反映済みかを `covered_until_turn_id` で追跡する方式へ変更。
  - 旧形式の履歴 JSON を読み込んだ際に、新しい履歴スキーマへ正規化する処理を追加し、既存セッションとの互換性を改善。長期間継続している session では、要約や文脈再利用の挙動が従来と変わる場合があります。

- モデルの出力品質とデバッグ機能の改善
  - モデルが生成する思考過程のテキスト（例: <think>...</think>タグなど）を最終的な出力から自動的に除去し、よりクリーンな回答を得られるようにしました。
  - デバッグログ(log_level: debug)を強化し、キャッシュの状態やモデルに渡されるプロンプトの内容をより詳細に追跡可能にしました。

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
