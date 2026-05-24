# Parameters and Advanced Usage

This document describes all configurable parameters used by **ComfyUI-LLM-Session**.
It is intended as a reference for advanced users.

---

## General Session Parameters

### history_dir
Directory used to store:
- conversation history
- automatic summaries
- session-scoped prompt cache files

Using the same `history_dir` allows conversations to persist across executions.

### reset_session
Overwrites the history and summary files associated with the session name, and resets per-session KV state.

`reset_session` does not delete `LlamaDiskCache`. Reusing the same `session_id` keeps the same disk-cache namespace.

---

## Generation Parameters

### max_tokens
Maximum number of tokens generated per turn.

- Small values: faster, may truncate output
- Large values: slower, more complete responses
- Full UI range: `1` to `32768`
- Simple-node default: `512` unless overridden in `config/simple_defaults.json`

Typical values:
- CPU: 256–512
- Long-form generation or editing: 2048–8192, if the model and memory budget allow it

---

### n_ctx
Maximum context length passed to the model.

- Full UI range: `512` to `131072`
- Simple-node default: `4096` unless overridden in `config/simple_defaults.json`
- `n_ctx` must be large enough to hold the prompt, conversation history, chat template overhead, safety margin, and the requested `max_tokens`.

Typical values:
- CPU: 4096
- `max_tokens=8192`: use at least `n_ctx=16384` for short prompts; `24576` or `32768` is safer when conversation history or long source text is included.

---

### n_gpu_layers
Number of model layers offloaded to GPU.

Default:
- `0` (matches `llama-cpp-python` default)

Values:
- `0`: CPU-only
- `-1`: all layers are offloaded
- positive integer: offload up to that many layers

Recommended usage:
- Increase gradually and find a stable value for your VRAM budget.
- If model loading fails, reduce `n_gpu_layers`.

---

### tensor_split
Advanced Simple-node setting for multi-GPU llama.cpp / llama-cpp-python environments.

This setting is read from `config/simple_defaults.json` only. Leave it as `null` or omit it to use the llama.cpp default behavior.

Examples:

```json
"tensor_split": [1.0, 0.0]
```

2 visible GPUs:
- `[1.0, 0.0]`: use GPU 0 for llama.cpp model offload
- `[0.0, 1.0]`: use GPU 1 for llama.cpp model offload
- `[1.0, 1.0]`: split across GPU 0 and GPU 1

3 visible GPUs:
- `[1.0, 0.0, 0.0]`: use GPU 0
- `[0.0, 1.0, 0.0]`: use GPU 1
- `[0.0, 0.0, 1.0]`: use GPU 2
- `[1.0, 1.0, 1.0]`: split across all three GPUs

Notes:
- `tensor_split` applies to GPUs visible to the current ComfyUI process.
- If `CUDA_VISIBLE_DEVICES` is set, physical GPU numbers may differ from the internal GPU order. For example, `CUDA_VISIBLE_DEVICES=1,0` makes physical GPU 1 appear as internal GPU 0.
- `n_gpu_layers` must be greater than `0` or `-1` for GPU offload to happen.

---

### temperature / top_p
Sampling parameters controlling randomness.

- Lower values: more deterministic
- Higher values: more diverse

---

### repeat_penalty / repeat_last_n
Used to reduce repetitive (echo) outputs.

Typical values:
- repeat_penalty: 1.1–1.15
- repeat_last_n: 256–512

---

### enable_thinking
Controls thinking / reasoning output for supported chat formats.

Default:
- `false`

Currently supported formats:
- Gemma 4
- Qwen3.5 / Qwen3.6 compatibility path

Full nodes expose this as an optional UI setting. Simple nodes can override it in `config/simple_defaults.json`:

```json
"gemma4": {
  "enable_thinking": false
},
"qwen3.5": {
  "enable_thinking": false
}
```

When disabled, the node asks the supported model/chat handler not to expose thinking output. This behavior still depends on the model, chat handler, and `llama-cpp-python` build, so some models may still emit internal channel text.

Gemma 4 note:

Gemma 4 may emit internal channel delimiters such as:

```text
<|channel>thought
...
<channel|>
final answer...
```

In this case, the text after the last `<channel|>` appears to be the final answer intended for the user. The node therefore treats the text after the last channel delimiter as the displayed final output.

If `max_tokens` is too small, generation may stop before Gemma 4 emits `<channel|>` and the final answer. In that case, the node cannot remove the earlier internal text. Increase `max_tokens` if you see unfinished thinking/channel output.

---

## Summarization Parameters

### summarize_old_history
When enabled, older conversation turns are automatically summarized.

---

### summary_chunk_turns
Number of turns accumulated before summarization is triggered.

Typical values:
- CPU: 5–7

---

## Cache Parameters

### persistent_cache
Controls persistent disk cache.

- LlamaDiskCache
- off 

`LlamaDiskCache` is scoped per `session_id`. Inside each session cache root, cache entries are further separated by model settings so incompatible configurations do not share the same disk cache directory.

Availability depends on the `llama-cpp-python` build. If disk cache support is unstable in your environment, use `off`.

---

### runtime_cache
Controls fast in-memory caches usage during conversation.

- KV_cache
- LlamaRAMCache
- LlamaTrieCache (recommended)
- off

---

## Logging Parameters

### log_level
Controls verbosity of internal logs.

- minimal
- timing (recommended)
- debug

---

### suppress_backend_logs
Suppresses verbose llama.cpp backend logs.

---

## Simple Node Overrides

### config_path
Optional JSON file used to override internal defaults in Simple nodes.

If the config is missing or invalid, built-in safe defaults are used.

---

### force_text_only (Dialogue Cycle Simple)
Forces pure text-only execution.

- Disables vision pathways
- Ignores mmproj auto-detection
- Improves reproducibility across environments

---

## continue / continue 2

If a response is truncated, enter `continue`, `continue 2`, etc.
The model will resume from the previous output.

---

## Suppressing Echo / Repeated Output

Some models may repeat the input prompt verbatim.

### Recommended Method: assistant prefix

Append a clear delimiter to the end of the user prompt:

```
---
Answer:
```

or:

```
---
Answer (English, ~800 words):
```

This clearly signals the start of the assistant response and reduces echo.

### stream_to_console

Streams the LLM output to the console in real time.

### Additional Helpful Settings

- temperature = 0.7–0.9
- top_p = 0.9
- repeat_penalty = 1.1–1.15
- repeat_last_n = 256–512

---

## Notes

- Not all parameters are required for every model
- Tune only when issues appear
- Simple nodes intentionally hide most parameters
