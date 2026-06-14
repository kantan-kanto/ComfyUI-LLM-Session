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
- Gemma 4 (E2B / E4B / 12B / 26B-A4B / 31B)*
- GLM-4.6V Flash*
- gpt-oss
- Llama 3.1 Instruct (8B / 70B)
- LLaVA
- MiniCPM-V 2.6
- MiniCPM-V 4.6*
- Mistral NeMo 12B Instruct
- Nemotron-Nano*
- Phi-3 Mini Instruct
- Phi-4
- Qwen2.5 Instruct (7B / 14B)
- Qwen2.5-VL (3B / 7B / 32B)
- Qwen3-30B-A3B
- Qwen3-VL (4B / 8B)
- Qwen3.5 (9B / 27B / 35B-A3B)*
- Qwen3.6 (27B / 35B-A3B)*

**Note:** Entries marked with `*` either do not work on official llama-cpp-python 0.3.16 or have not been tested on it.

---

## MoE Models (Backend-Dependent)

- Qwen3-30B-A3B, Qwen3.5/3.6-35B-A3B, and Gemma4-26B-A4B: confirmed working
- Mixtral 8x7B GGUF: may fail to load depending on backend build

Mixtral failures occur at model load time and are likely backend-related.

---

## Vision Support Caveats

- Some vision-capable models may ignore image inputs without errors
- `LLM Session Chat` and `LLM Session Chat (Simple)` send IMAGE batches as
  multiple image message parts. Whether every frame/image is used depends on the
  active backend handler and model.
- Vision compatibility depends on mmproj selection and chat template support
- Vision chat handler detection uses dynamic loading. If a required handler is
  unavailable in your `llama-cpp-python` build, execution stops with a
  diagnostic error instead of silently falling back to text-only mode.
- mmproj auto-detection depends on normalized model-family aliases in model and
  mmproj filenames. If filenames fall outside the expected alias patterns,
  Auto-detect may fail.

---

## Gemma 4 AUDIO Input

ComfyUI `AUDIO` input is currently routed only for models detected as Gemma 4.
The node encodes the AUDIO object as WAV and sends it as an OpenAI-style
`input_audio` message part. Other model families reject AUDIO explicitly.

This is intended for Gemma 4 builds whose chat handler actually consumes audio
waveform input, especially the Gemma 4 12B Unified path. Model-card support and
GGUF/backend support are separate requirements: use a recent JamePeng
`llama-cpp-python` build and verify that its `Gemma4ChatHandler` accepts
`input_audio` in your environment. In current testing, the handler still
requires a valid mmproj path; leaving mmproj unset fails during handler
initialization.

---

## Backend Compatibility by llama-cpp-python Version

**Important:** Model compatibility varies by llama-cpp-python version. Based on my testing environment:

| Version | confirmed <br> models <br> (Text)| Qwen2.5-VL <br> LLaVA <br> Llama-3.1 <br> MiniCPM-V 2.6 <br> (Vision) | Qwen3-VL <br> Qwen3.5/3.6 <br> Gemma 3/4 <br> GLM-4.6V <br> MiniCPM-V 4.6 <br> (Vision) |
|---------|-------------------|-------------------|-------------------|
| 0.3.16 (official) | ✅* | ✅ | ❌ |
| 0.3.40+ (JamePeng fork) | ✅ | ✅ | ✅ |

**Note:** Entries marked with `*` either do not work on official llama-cpp-python 0.3.16 or have not been tested on it.

**Recommended Backend Notes:**  
For newer Vision model families, please follow the build and installation information provided by the upstream JamePeng llama-cpp-python project and choose a build appropriate for your OS, Python version, and acceleration backend.

`0.3.33+` (JamePeng fork) works for `Qwen3.5` Vision in my environment. Earlier `0.3.30+` builds added support for `Qwen3.5`, but Vision mode was not yet working reliably for me at that stage.

**Source:** https://github.com/JamePeng/llama-cpp-python

## Disclaimer

These results are environment-dependent and provided for reference only. Your
results may differ depending on system configuration, GPU drivers, and other
factors. If you encounter issues, please verify your environment setup and
consider reporting compatibility details.
