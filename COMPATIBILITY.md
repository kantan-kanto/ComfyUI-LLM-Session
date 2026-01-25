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

- Phi-3 Mini Instruct
- Gemma 2 Instruct (2B / 9B)
- Gemma 3 Instruct (4B / 12B)
- Llama 3.1 Instruct (8B / 70B)
- Mistral NeMo 12B Instruct
- Qwen 2.5 Instruct (7B / 14B)
- Qwen3-VL (4B / 8B)
- Qwen2.5-VL-7B
- Shisa v2

---

## MoE Models (Backend-Dependent)

- Qwen3-30B-A3B: confirmed working
- Mixtral 8x7B GGUF: may fail to load depending on backend build

Mixtral failures occur at model load time and are likely backend-related.

---

## Vision Support Caveats

- Some vision-capable models may ignore image inputs without errors
- Vision compatibility depends on mmproj selection and chat template support

---

## Backend Compatibility by llama-cpp-python Version

**Important:** Model compatibility varies by llama-cpp-python version. Based on my testing environment:

| Version | confirmed working <br> models (Text)| Qwen2.5-V, Gemma 3 <br> (Vision)| Qwen3-VL <br> (Vision) | 
|---------|-------------------|---------------------|----------|
| 0.3.16 (official) | ✅ | ❌ | ❌ |
| 0.3.21+ (JamePeng fork) | ✅ | ❌* | ✅ |

***Note:** Vision input support may vary depending on your environment and configuration. In my setup, I have not been able to get vision input working with Qwen2.5-VL and Gemma 3 even with the JamePeng fork.

**Recommended Installation (JamePeng fork for Qwen3-VL support):**  
Please follow the build and installation instructions provided in the JamePeng fork repository, as this fork requires a custom build and cannot be reliably installed via a simple `pip install`.

**Source:** https://github.com/JamePeng/llama-cpp-python

**My Environment Results:**
- Official llama-cpp-python 0.3.16: Qwen2.5-VL text-only, no vision input, Qwen3-VL fails to load
- JamePeng fork 0.3.21+: Qwen3-VL works with vision input, Qwen2.5-VL and Gemma 3 text works but vision input still unavailable

⚠️ **Disclaimer:** Your results may differ depending on system configuration, GPU drivers, and other factors. If you encounter issues, please verify your environment setup and consider reporting compatibility details.

---

## Chat Template Compatibility (System Role)

Some models do not support the `system` role.

The node automatically retries by merging system messages into user messages.

---

## Disclaimer

These results are environment-dependent and provided for reference only.

