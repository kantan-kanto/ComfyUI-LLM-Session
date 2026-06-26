# Benchmarks

This page contains informal local benchmark results for model runs with
ComfyUI-LLM-Session. These numbers are intended as rough guidance only. Model
speed depends heavily on hardware, drivers, Python packages, llama.cpp /
llama-cpp-python versions, model format, quantization, prompt length, and
whether the run includes vision processing.

The raw console logs are not published here. The table below is a summarized
view of the local results. All rows used the same image input and the same text
prompt.

## Test Setup

### Local Test Environment

These results were measured on a single local Windows machine.

| Item | Value |
|---|---|
| OS | Microsoft Windows 11 Home 10.0.26200, 64-bit |
| CPU | Intel Core Ultra 5 225H, 14 cores / 14 threads |
| RAM | 32 GB |
| GPU offload | Disabled during GGUF model inference (`n_gpu_layers=0`) |

### Node Settings

Unless otherwise noted, these runs used the Simple node defaults from
`config/simple_defaults.json`:

| Setting | Value |
|---|---:|
| `n_ctx` | `4096` |
| `n_gpu_layers` | `0` |
| `max_tokens` | `512` |
| `dynamic_max_tokens` | `true` |
| `min_generation_tokens` | `96` |
| `safety_margin_tokens` | `64` |
| `max_turns` | `6` |
| `temperature` | `0.7` |
| `top_p` | `0.9` |
| `repeat_penalty` | `1.12` |
| `repeat_last_n` | `256` |
| `runtime_cache` | `LlamaTrieCache` |
| `persistent_cache` | `off` |
| `summarize_old_history` | `true` |

Most GGUF rows below logged `token_limit=448`, which is consistent with
`max_tokens=512` and `safety_margin_tokens=64`.

All benchmark runs used image input. Vision-capable GGUF runs used the matching
`mmproj` file for the model.

## Local Speed Results

Rows are sorted by dense-equivalent size from smallest to largest. Each row used
the same image and prompt. `Prompt time` is the full ComfyUI prompt execution
time logged for the run. `Generation time` and `tok/s` come from the LLM Session
generation metrics where available.

| Size | Dense-equivalent size | Model | Runtime | Generation time | Tokens | tok/s | Predicted tok/s | Prompt time |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 2B | 1.0B | `gemma-4-E2B-it-heretic-Q8_K_XL.gguf` | GGUF, `n_gpu_layers=0` | 40.67 s | 292 | 7.18 | 7.24 | 47.58 s |
| 1.3B | 1.3B | `MiniCPM-V-4_6-Q8_0.gguf` | GGUF, `n_gpu_layers=0` | 23.43 s | 130 | 5.55 | 5.57 | 37.64 s |
| 4B | 1.4B | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | GGUF, `n_gpu_layers=0` | 81.82 s | 429 | 5.24 | 5.17 | 85.22 s |
| 4B | 1.5B | `gemma-4-E4B-it-heretic-Q8_K_XL.gguf` | GGUF, `n_gpu_layers=0` | 94.89 s | 448 | 4.72 | 4.83 | 100.70 s |
| 26B | 3.2B | `gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf` | GGUF, `n_gpu_layers=0` | 62.94 s | 141 | 2.24 | 2.26 | 132.99 s |
| 35B | 6.5B | `Qwen3.6-35B-A3B-uncensored-heretic.i1-Q4_K_M.gguf` | GGUF, `n_gpu_layers=0` | 88.17 s | 99 | 1.12 | 1.11 | 99.19 s |
| 9B | 9B | `Qwen3.5-9B-ultra-uncensored-heretic-v2-Q8_0.gguf` | GGUF, `n_gpu_layers=0` | 162.88 s | 110 | 0.68 | 0.80 | 169.21 s |
| 12B | 12B | `gemma-4-12B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf` | GGUF, `n_gpu_layers=0` | 137.37 s | 127 | 0.92 | 0.60 | 144.75 s |
| 27B | 27B | `Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q5_K_M.gguf` | GGUF, `n_gpu_layers=0` | 461.07 s | 125 | 0.27 | 0.27 | 471.95 s |
| 31B | 31B | `gemma-4-31B-it-heretic.Q5_K_M.gguf` | GGUF, `n_gpu_layers=0` | 412.16 s | 126 | 0.31 | 0.23 | 426.83 s |

## External Reference Result

The following result was produced by ComfyUI's official Generate Text node, not
by a ComfyUI-LLM-Session node. It is kept only as a rough local reference and
should not be compared directly with the GGUF rows above.

| Size | Dense-equivalent size | Model | Runtime | Generation time | Tokens | tok/s | Predicted tok/s | Prompt time |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 9B | 9B | `qwen3.5_9b_bf16.safetensors` | Generate Text node, CPU, `torch.float16` | 564 s at 97/512 tokens | 97/512 | 0.17 | 0.80 | 598.70 s |

## Notes

- These are single-machine local measurements, not a controlled benchmark
  suite.
- `Dense-equivalent size` is set equal to `Size` for dense models. For MoE
  models, it estimates the dense model size that would produce a similar tok/s
  under the rough local dense-model approximation `tok/s = 7.24 / size`.
- `Predicted tok/s` applies `tok/s = 7.24 / size` to the dense-equivalent size.
- The XPU run for `qwen3.5_9b_bf16.safetensors` was omitted because the model
  did not recognize the input image correctly and the generated text was
  corrupted.
- The external `qwen3.5_9b_bf16.safetensors` CPU row comes from progress-bar
  output rather than the LLM Session generation summary.
- Raw local paths for image files, model files, and `mmproj` files are
  intentionally omitted.
