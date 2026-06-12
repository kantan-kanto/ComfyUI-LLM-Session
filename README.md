# ComfyUI-LLM-Session
[en | [ja](README.ja.md)]

**Version:** 1.2.3
**License:** GPL-3.0

A local LLM execution environment that runs entirely inside **ComfyUI**, 
without external runtimes such as Ollama.

Supports GGUF models via llama.cpp, including many popular open-weight LLMs
such as Llama, Mistral, Qwen, DeepSeek, GLM, Gemma, LLaVA and gpt-oss.

In addition to user–model chat, it also supports model-to-model dialogue
for **observation, experimentation, and analysis**.

---

<details>
<summary><strong>Upgrade Notes for Existing Users</strong></summary>

The following notes are intended for existing users upgrading to the `1.1.x` / `1.2.x` series.

- Cache-related setting names have changed. The previous `prompt_cache_mode` / `kv_state_mode` options have been reorganized into `persistent_cache` / `runtime_cache`.
- The cache storage directory name has changed from `prompt_cache/` to `cache/`. Existing cache data is not migrated automatically.
- `reset_session` now clears history and per-session KV state, while keeping the session's disk cache so the same session can restart efficiently.
- Older history JSON files are normalized automatically, but the tracking model for summarized ranges has changed. Long-lived sessions may therefore behave somewhat differently from previous versions.
- When using Vision models, both mmproj auto-detection and handler selection logic have changed. Even combinations that worked before may need to be rechecked depending on backend behavior and filename conventions.
- `LLM Dialogue Cycle` now keeps model managers loaded when `runtime_cache` is `KV_cache` or `LlamaTrieCache`.
- Added `Unload LLM Model` output node for explicit manual VRAM release after keep-loaded runs.
- History loading now restores from `*.bak` when the primary history JSON is invalid or missing.
- Adaptive retry now recognizes additional context-overflow error wording (`context window ... exceed ...`).

For details, see the `1.1.x` and `1.2.x` sections in [CHANGELOG.md](CHANGELOG.md). For Vision / backend-specific differences, see [COMPATIBILITY.md](COMPATIBILITY.md).

</details>

---

## What This Project Is

- A **file-based session system** for local GGUF LLMs
- A set of ComfyUI nodes for **persistent multi-turn conversations**
- A tool for **observing model behavior**, convergence, and failure modes
- Fully self-contained inside ComfyUI (no server, no daemon)

---

## Provided Nodes

### LLM Session Chat
Standard session-based chat node with full parameter control.

### LLM Session Chat (Simple)
A minimal UI version with safe defaults, designed for quick testing and demos.

- Session persistence included
- Parameters fixed internally
- Optional external config override (JSON)

### LLM Dialogue Cycle
Model-to-model dialogue execution without graph cycles.

### LLM Dialogue Cycle (Simple)
Minimal version focused on **role-based dialogue observation**.

### Unload LLM Model
A utility output node that manually unloads the current LLM from VRAM.
Set `unload_now=true` and queue the node to release model memory.
After running, set it back to false to avoid repeated unloads.

---

## Key Design Concepts

- **Session-first design**: sessions live beyond a single execution
- **File-based persistence**: no external state or database
- **Observation-oriented**: transcripts are first-class outputs
- **Deterministic turn execution**: suitable for analysis

---

## Installation

### 1. Clone Repository

Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kantan-kanto/ComfyUI-LLM-Session.git
```

### 2. Install Dependencies

```bash
cd ComfyUI-LLM-Session
pip install -r requirements.txt
```

**Alternative manual installation:**
```bash
pip install pillow numpy
```

### 3. Install llama-cpp-python (Notes)

Model compatibility depends on the llama-cpp-python build.
Vision support varies significantly by backend and environment.

- Official releases work for many text-only workflows.
- Newer multimodal model families can require chat handlers that are only
  available in recent backend builds.
- If a Vision request fails because the required multimodal chat handler is
  unavailable, check the upstream JamePeng llama-cpp-python project and choose
  a build that matches your OS, Python version, and acceleration backend:
  https://github.com/JamePeng/llama-cpp-python

See [COMPATIBILITY.md](COMPATIBILITY.md) for detailed environment test results.

### 4. Place Models

Place your GGUF models in `ComfyUI/models/LLM/`:
```
ComfyUI/models/LLM/
├── Qwen3VL-4B-Q8_0.gguf
├── mmproj-qwen3vl-4b-f16.gguf
└── ...
```

**Notes for Vision models**

When using Vision-capable models, please follow these rules:

- Place the **model** and **mmproj** GGUF files in the **same folder**.
- The model filename must start with one of the following prefixes and end with `.gguf`:

  `llava-1-5, llava15, llava-v1.5, llava-1-6, llava16, llava-v1.6, moondream2, nanollava, llama-3, llama3, minicpm-v-2.6, minicpm-v-2_6, minicpmv26, minicpm-v-4.0, minicpm-v-4_0, minicpmv40, minicpm-v-4.5, minicpm-v-4_5, minicpmv45, minicpm-v-4.6, minicpm-v-4_6, minicpmv46, gemma3, gemma-3, gemma_3, gemma4, gemma-4, gemma_4, glm4.1v, glm4_1v, glm41v, glm-4.1v, glm4.6v, glm4_6v, glm46v, glm-4.6v, granitedocling, granite-docling, lfm2-vl, lfm2vl, lfm2.5-vl, lfm2.5vl, lfm2_5-vl, lfm2_5vl, paddleocr, qwen2.5-vl, qwen2_5-vl, qwen25vl, qwen3-vl, qwen3vl, qwen3.5, qwen3_5, qwen35, qwen3.6, qwen3_6, qwen36, step3-vl, step3vl`
  
- The mmproj filename must start with `mmproj-` and end with `.gguf`.
- If exactly one file matching
  `mmproj-*[llava-1-5|llava15|llava-v1.5|llava-1-6|llava16|llava-v1.6|moondream2|nanollava|llama-3|llama3|minicpm-v-2.6|minicpm-v-2_6|minicpmv26|minicpm-v-4.0|minicpm-v-4_0|minicpmv40|minicpm-v-4.5|minicpm-v-4_5|minicpmv45|minicpm-v-4.6|minicpm-v-4_6|minicpmv46|gemma3|gemma-3|gemma_3|gemma4|gemma-4|gemma_4|glm4.1v|glm4_1v|glm41v|glm-4.1v|glm4.6v|glm4_6v|glm46v|glm-4.6v|granitedocling|granite-docling|lfm2-vl|lfm2vl|lfm2.5-vl|lfm2.5vl|lfm2_5-vl|lfm2_5vl|paddleocr|qwen2.5-vl|qwen2_5-vl|qwen25vl|qwen3-vl|qwen3vl|qwen3.5|qwen3_5|qwen35|qwen3.6|qwen3_6|qwen36|step3-vl|step3vl]*.gguf`
  exists in the folder, it can be selected automatically via Auto-detect.
- Filename matching is **case-insensitive**.
- Both GGUF extension matching (for example `.gguf`, `.GGUF`) and `mmproj` prefix matching (for example `mmproj-`, `MMPROJ-`) are case-insensitive.

---

### Simple Node Settings (Quick Notes)

- **history_dir**: Conversations persist as long as the same directory is used.
- **config_path**: Optional JSON file to override internal defaults in Simple nodes.
  For fixed seed generation and other advanced Simple-node JSON settings, see
  [ADVANCED_PARAMETERS.md](ADVANCED_PARAMETERS.md).
- **force_text_only** (Dialogue Cycle Simple): Forces pure text mode to avoid mmproj / vision handler differences and improve reproducibility.
- **reset_session** (Dialogue Cycle Simple): Overwrites the history and summary files associated with the session name, and resets per-session KV state. The session's disk cache is kept.
- **tensor_split** (`config/simple_defaults.json`): Optional llama.cpp multi-GPU split. For example, `"tensor_split": [1.0, 0.0]` keeps llama.cpp model offload on visible GPU 0 in a 2-GPU setup. Leave it `null` or omit it for default behavior.

### Generation Limits

- Full UI nodes allow `max_tokens` up to `32768` and `n_ctx` up to `131072`.
- Defaults remain conservative: `max_tokens=512` and `n_ctx=4096`.
- For long-form generation, raise `n_ctx` together with `max_tokens`; for example, `max_tokens=8192` usually needs at least `n_ctx=16384`, and `24576` or `32768` is safer with history or long source text.

### Cache Scope Notes

- `persistent_cache = LlamaDiskCache` stores disk cache data under `history_dir/cache/<session_id>/`.
- Disk cache is isolated per session id, then split by model settings inside that session cache root.
- `reset_session` does not delete disk cache. To start with a different cache namespace, use a different `session_id`.
- `LLM Dialogue Cycle` keeps model managers loaded during and after execution when `runtime_cache` is `KV_cache` or `LlamaTrieCache`.
- For other runtime cache modes, `LLM Dialogue Cycle` unloads managers between turns and at execution end.
- To release VRAM explicitly after a keep-loaded run, execute `Unload LLM Model` manually.

See [PARAMETERS.md](PARAMETERS.md) for the full list of settings and advanced usage.

---

## Example Workflow

A ready-to-run example workflow is included:

```
examples/example_workflow.json
```

### Purpose

This workflow demonstrates **session persistence across executions** using **LLM Session Chat (Simple)**.

### How to Use

1. Load [example_workflow.json](examples/example_workflow.json) in ComfyUI
2. Set your GGUF model path
3. Set `history_dir` to any writable directory
4. Run the workflow once (Turn 1)
5. Replace the text input with the Turn 2 prompt shown below
6. Run the workflow again

### Turn 1 Prompt (included in JSON)

```
Please prepare an explanation about the key points
to consider when using a local LLM in real-world scenarios.
Do not output the explanation yet.
```

### Turn 2 Prompt (replace text input)

```
Now, please provide the explanation you prepared earlier.
Write in clear English and separate the content into paragraphs.
```

The second response depends on context prepared during the first run.

---

## Screenshots

### LLM Session Chat (Simple)

![LLM Session Chat Simple](images/LLM_Session_Chat_Simple.png)

Demonstrates session persistence across executions.

### LLM Dialogue Cycle (Simple)

![LLM Dialogue Cycle Simple](images/LLM_Dialogue_Cycle_Simple.png)

Demonstrates model-to-model dialogue with separate roles and a full transcript output.

---

## Model Compatibility (Tested)

The following GGUF instruction models have been tested.

### Text Chat (Confirmed)

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

### MoE Models

- MoE models can work depending on backend support
- Qwen3-30B-A3B, Qwen3.5/3.6-35B-A3B, and Gemma4-26B-A4B confirmed working
- Mixtral GGUF may fail to load depending on llama.cpp / llama-cpp-python build

### Vision Models

- Vision support depends on model + mmproj + backend
- Some vision-capable models may ignore images without errors
- Text-only operation is always supported

---

## Chat Template Compatibility

Some GGUF models do not support the `system` role.

This project automatically falls back by merging system messages into user messages when needed.

This behavior is generic and not model-specific.

---

## Performance Notes

- Performance strongly depends on model size and quantization
- Large models may be extremely slow on CPU
- Long sessions benefit from summarization and history management

---

## Examples Directory

```
examples/
 ├─ example_workflow.json
```

---

## License

This project is licensed under the **GNU General Public License v3.0**.

**Copyright (C) 2026 kantan-kanto**  
GitHub: https://github.com/kantan-kanto

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

**Note:** GPL-3.0 is required due to llama-cpp-python dependency.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas needing help:
- Testing on different hardware configurations
- Documenting vision input compatibility across environments
- Additional workflow examples
- Performance optimizations

---

## Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Examples**: Check [examples/](examples/) for workflow templates

---

## Release Notes

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Current Version: 1.2.2 / 1.2.3

- `1.2.3` is a Registry compatibility re-release of `1.2.2`.
- Removed a plain URL comment from `requirements.txt` to avoid false-positive dependency flagging by the ComfyUI Registry scanner.
- Improved Vision model diagnostics when a required multimodal chat handler is unavailable.
- Added MiniCPM-V-4.6 aliases, chat-handler mapping, and Text-only prompt support.
- Added advanced JSON-based parameter settings for Simple-node config files.
- Added [ADVANCED_PARAMETERS.md](ADVANCED_PARAMETERS.md) for advanced JSON settings and current limitations.
- Kept the default `llama-cpp-python` dependency conservative while documenting backend compatibility notes for newer Vision models.
