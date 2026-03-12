# Model Compatibility and Backend Notes

This document summarizes empirical compatibility results obtained during development and testing.

---

## Scope and Interpretation

- Compatibility refers to *successful text generation*, not output quality
- Performance varies significantly by model size and quantization
- Vision behavior is evaluated separately from text behavior

---

## Tested Models (Text Compatibility)

### Confirmed Working (Text)

- DeepSeek
- Gemma 2 Instruct (2B / 9B)
- Gemma 3 Instruct (4B / 12B)
- GLM-4.6V Flash*
- gpt-oss
- Llama 3.1 Instruct (8B / 70B)
- LLaVA
- MiniCPM-V 2.6
- Mistral NeMo 12B Instruct
- Nemotron-Nano*
- Phi-3 Mini Instruct
- Phi-4*
- Qwen2.5 Instruct (7B / 14B)
- Qwen2.5-VL (3B / 7B)
- Qwen3-30B-A3B
- Qwen3-VL (4B / 8B)
- Qwen3.5 (9B / 27B / 35B-A3B)*

**Note:** Official llama-cpp-python 0.3.16: `GLM-4.6V Flash` and `Phi-4` fail to load
Nemotron-Nano、Qwen3.5のロード可否は未確認

---

## MoE Models (Backend-Dependent)

- Qwen3-30B-A3B, Qwen3.5-35B-A3B: confirmed working
- Mixtral 8x7B GGUF: may fail to load depending on backend build

Mixtral failures occur at model load time and are likely backend-related.

---

## Vision Support Caveats

- Some vision-capable models may ignore image inputs without errors
- Vision compatibility depends on mmproj selection and chat template support

---

## Upgrade-related Compatibility Notes

以下は `1.0.3` 系更新に伴う、既存ユーザー向けの互換性上の注意事項です。

- Vision モデルの chat handler 判定は動的ロード方式に変更されました。対応するVisionモデルが増えています。`llama-cpp-python` 側に handler 実装が存在しない場合、その chat format は自動的に無効化されます。
- mmproj の自動検出はモデル名と mmproj 名の別名正規化に依存します。ファイル名が想定プレフィックスから外れている場合、Auto-detect が失敗することがあります。
- 一部 vision 系モデルは backend 依存が強く、handler が読み込めても画像入力が安定動作しない場合があります。私の環境では`Qwen3.5` のVisionが利用できていません。Textだけでも素晴らしいので、ぜひ使ってみて下さい。
- cache 機構の構成を変更しました。"prompt_cache_mode": "LlamaDiskCache", "runtime_cache": "LlamaTrieCache"をデフォルト設定としました。既存のUI設定はそのままだとエラーがでますので、選択し直して下さい。jsonでの既存設定項目は無視されますので、項目名"prompt_cache_mode"、"runtime_cache "とそれぞれの設定値を追加記載して下さい。 なお私の環境では、最新（0.3.30+）の`llama-cpp-python`でKV_cacheが利用できなくなっています。
- cache 互換性判定が強化されたため、旧バージョンで作成された cache や runtime 条件の異なる cache は、自動的に無効化されることがあります。これは不具合ではなく安全側の動作です。
- `reset_session` 実行時は cache も削除されるため、更新前よりも「完全な新規セッション」に近い挙動になります。

---

## Backend Compatibility by llama-cpp-python Version

**Important:** Model compatibility varies by llama-cpp-python version. Based on my testing environment:

| Version | confirmed <br> models <br> (Text)| Qwen2.5-VL <br> LLaVA <br> Llama-3.1 <br> MiniCPM-V 2.6 <br> (Vision) | Qwen3-VL <br> Gemma 3 <br> GLM-4.6V <br> (Vision) |
|---------|-------------------|-------------------|-------------------|
| 0.3.16 (official) | ✅* | ✅ | ❌ |
| 0.3.21+ (JamePeng fork) | ✅ | ✅ | ✅ |

**Note:** Official llama-cpp-python 0.3.16: `GLM-4.6V Flash` and `Phi-4` fail to load

**Recommended Installation (JamePeng fork for Qwen3-VL support):**  
Please follow the build and installation instructions provided in the JamePeng fork repository, as this fork requires a custom build and cannot be reliably installed via a simple `pip install`.

0.3.30+ (JamePeng fork)でQwen3.5に対応していますが、私の環境ではVisonの利用は成功していません。text_only_modeのみ利用できています。

**Source:** https://github.com/JamePeng/llama-cpp-python

⚠️ **Disclaimer:** Your results may differ depending on system configuration, GPU drivers, and other factors. If you encounter issues, please verify your environment setup and consider reporting compatibility details.

---

## Chat Template Compatibility (System Role)

Some models do not support the `system` role.

The node automatically retries by merging system messages into user messages.

---

## Disclaimer

These results are environment-dependent and provided for reference only.
