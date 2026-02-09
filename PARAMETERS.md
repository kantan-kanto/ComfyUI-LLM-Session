# Parameters and Advanced Usage

This document describes all configurable parameters used by **ComfyUI-LLM-Session**.
It is intended as a reference for advanced users.

---

## General Session Parameters

### history_dir
Directory used to store:
- conversation history
- automatic summaries
- prompt cache files

Using the same `history_dir` allows conversations to persist across executions.

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

### kv_state_mode
Controls KV cache usage during conversation.

- memory (recommended)
- off

---

### prompt_cache_mode
Controls prompt evaluation cache.

- disk (recommended)
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

