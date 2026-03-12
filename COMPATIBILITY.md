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

The following notes are compatibility-related cautions for existing users upgrading to the `1.0.3` series.

- Vision model chat handler detection now uses dynamic loading. Supported Vision model coverage has increased, but if the required handler implementation does not exist in your `llama-cpp-python` build, that chat format is disabled automatically.
- mmproj auto-detection now depends on normalized aliases for both model and mmproj filenames. If filenames fall outside the expected prefix patterns, Auto-detect may fail.
- Some Vision-family models remain highly backend-dependent. Even when the handler loads successfully, image input may still be unstable or unsupported in practice. In my environment, `Qwen3.5` Vision is not working yet, although text-only use works well.
- The cache configuration model has changed. The new default settings are `persistent_cache = "LlamaDiskCache"` and `runtime_cache = "LlamaTrieCache"`. Existing UI settings should be reselected manually if needed. Older JSON config keys are ignored, so update your config files to use the new option names and values. In my environment, `KV_cache` no longer works with the latest `llama-cpp-python` (`0.3.30+`).
- Cache compatibility checks are now stricter. Cache data created by older versions or under different runtime conditions may be invalidated automatically. This is expected safety behavior, not a defect.
- `reset_session` now also clears cache data, so its behavior is closer to starting a completely fresh session than in previous versions.

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
