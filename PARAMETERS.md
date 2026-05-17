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

Typical values:
- CPU: 256–512

---

### n_ctx
Maximum context length passed to the model.

Typical values:
- CPU: 4096

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
