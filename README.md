# ComfyUI-LLM-Session
[en | [ja](README.ja.md)]

**Version:** 1.0.0
**License:** GPL-3.0

A local LLM execution environment that runs entirely inside **ComfyUI**, 
without external runtimes such as Ollama.

Supports GGUF models via llama.cpp, including many popular open-weight LLMs
such as Qwen, Llama, Mistral NeMo, Gemma, and Phi-3 Mini.

In addition to user–model chat, it also supports model-to-model dialogue
for **observation, experimentation, and analysis**.

---

## What This Project Is

- A **file-based session system** for local GGUF LLMs
- A set of ComfyUI nodes for **persistent multi-turn conversations**
- A tool for **observing model behavior**, convergence, and failure modes
- Fully self-contained inside ComfyUI (no server, no daemon)

---

## What This Project Is NOT

- Not a chat UI
- Not an API server
- Not an Ollama replacement
- Not an end-user application

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

- Official releases work for text-only usage
- Qwen3-VL requires a custom build (e.g. JamePeng fork)

See [COMPATIBILITY.md](COMPATIBILITY.md) for detailed environment test results.

### 4. Place Models

Place your GGUF models in `ComfyUI/models/LLM/`:
```
ComfyUI/models/LLM/
├── Qwen3VL-4B-Q8_0.gguf
├── mmproj-qwen3vl-4b-f16.gguf
└── ...
```

---

### Simple Node Settings (Quick Notes)

- **history_dir**: Conversations persist as long as the same directory is used.
- **config_path**: Optional JSON file to override internal defaults in Simple nodes.
- **force_text_only** (Dialogue Cycle Simple): Forces pure text mode to avoid mmproj / vision handler differences and improve reproducibility.

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

- Llama 3.1 (8B / 70B)
- Qwen 2.5 (7B / 14B)
- Qwen3-VL (4B / 8B)
- Gemma 2 Instruct (2B / 9B)
- Gemma 3 Instruct (4B / 12B)
- Mistral NeMo 12B
- Phi-3 Mini
- Shisa v2

### MoE Models

- MoE models can work depending on backend support
- Qwen3-30B-A3B confirmed working
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
### 1.0.0
- Initial release
