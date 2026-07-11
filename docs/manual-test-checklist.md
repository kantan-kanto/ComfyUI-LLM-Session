# Manual ComfyUI Test Checklist

- Status: Canonical
- Last reviewed: 2026-07-11
- Update when: Manual ComfyUI validation scope, triage, or checklist steps change.
- Use when: A change may affect ComfyUI UI wiring, real graph execution, model
  loading, backend behavior, or user-visible runtime behavior.
- Do not use when: The change is documentation-only, test-only, or fully covered
  by pytest with no ComfyUI runtime impact.

This checklist is not required for every change. Use the triage and routing
matrix to select only the sections relevant to the change.

Agents must not mark manual checks as passed unless a human actually performed
them. When manual validation is needed, agents should tell the human which
sections of this file to run.

## How To Use

1. Complete the triage section.
2. Use the routing matrix to choose the relevant manual sections.
3. Run only the sections that apply to the change.
4. Record the result in the change notes, PR, or verification plan.

## Triage

Check the categories affected by the change:

- [ ] Node registration, category, or display name
- [ ] Node inputs, defaults, widgets, return types, or return names
- [ ] Session Chat execution
- [ ] Dialogue Cycle execution
- [ ] Continue behavior
- [ ] Summary behavior
- [ ] KV cache behavior
- [ ] GGUF discovery or model loading
- [ ] Multimodal or image-capable model path
- [ ] User-visible error handling or log output
- [ ] Documentation-only change
- [ ] Test-only change

Manual ComfyUI validation decision:

- [ ] Required
- [ ] Not required

Reason:

- 

## Routing Matrix

| Change type | Manual sections to run |
| --- | --- |
| Node registration, category, or display name | Minimal Common Checks, Node Interface Checks |
| Node inputs, defaults, widgets, return types, or return names | Minimal Common Checks, Node Interface Checks |
| Session Chat behavior | Minimal Common Checks, Session Chat Checks, Error Handling Checks if affected |
| Dialogue Cycle behavior | Minimal Common Checks, Dialogue Cycle Checks, KV Cache Checks if affected |
| Continue or summary behavior | Minimal Common Checks, Continue And Summary Checks |
| KV cache behavior | Minimal Common Checks, KV Cache Checks |
| GGUF discovery or model loading | Minimal Common Checks, Model And GGUF Discovery Checks |
| Multimodal or image-capable model path | Minimal Common Checks, Multimodal Checks |
| User-visible errors or runtime logging | Minimal Common Checks, Error Handling Checks |
| Documentation-only change | Usually no manual ComfyUI check |
| Test-only change | Usually no manual ComfyUI check |

## Result Format

Use this format when reporting manual validation:

```text
Manual ComfyUI validation: performed / required but not performed / not required
Sections run:
- Minimal Common Checks
- ...
Environment:
Model used:
Result:
Notes:
```

## Minimal Common Checks

Required when:

- Any manual ComfyUI validation is required.

Skip when:

- Manual ComfyUI validation is not required.

Required checks:

- [ ] ComfyUI starts successfully.
- [ ] ComfyUI-LLM-Session imports without startup errors.
- [ ] The affected node appears in the expected ComfyUI category.
- [ ] The affected node can be placed in a graph.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Node Interface Checks

Required when:

- Node registration, category, display name, inputs, defaults, widgets, return
  types, or return names changed.

Skip when:

- The change does not affect ComfyUI node UI or public node contracts.

Required checks:

- [ ] The node display name is correct.
- [ ] The node category is correct.
- [ ] Required inputs are visible.
- [ ] Optional inputs are visible when expected.
- [ ] Default values look correct.
- [ ] Return sockets appear as expected.
- [ ] Existing workflows can still load when compatibility should be preserved.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Session Chat Checks

Required when:

- Session Chat execution, prompt handling, message construction, generation
  orchestration, or session behavior changed.

Skip when:

- The change does not affect Session Chat runtime behavior.

Required checks:

- [ ] A minimal Session Chat workflow runs.
- [ ] A simple text prompt produces a response.
- [ ] The response is visible in the expected output.
- [ ] Session/history behavior matches the change intent.
- [ ] Re-running the node does not produce unexpected errors.

Optional checks:

- [ ] Simple node wrapper behavior still works.
- [ ] Full node behavior still works.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Dialogue Cycle Checks

Required when:

- Dialogue Cycle request construction, multi-turn flow, role handling, or
  conversation loop behavior changed.

Skip when:

- The change does not affect Dialogue Cycle runtime behavior.

Required checks:

- [ ] A minimal Dialogue Cycle workflow runs.
- [ ] The first turn produces the expected style of response.
- [ ] A follow-up turn uses prior context when expected.
- [ ] Outputs are routed to the expected sockets.
- [ ] Re-running the graph does not produce unexpected errors.

Optional checks:

- [ ] Simple node wrapper behavior still works.
- [ ] Full node behavior still works.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Continue And Summary Checks

Required when:

- Continue behavior, summary behavior, prompt rewriting, summary override, or
  related defaults changed.

Skip when:

- The change does not affect continue or summary paths.

Required checks:

- [ ] Continue behavior produces a coherent continuation when enabled.
- [ ] Normal generation still works when continue behavior is disabled.
- [ ] Summary behavior runs when enabled.
- [ ] Summary-related outputs or history effects match the change intent.
- [ ] Summary settings do not leak into unrelated generation paths.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## KV Cache Checks

Required when:

- KV cache save/load behavior, session identity, history interaction, or
  cache-related UI defaults changed.

Skip when:

- The change does not touch KV cache state or cache-related controls.

Required checks:

- [ ] Cache can be saved when expected.
- [ ] Cache can be loaded when expected.
- [ ] Generation still works when cache is disabled.
- [ ] Cache-related errors are understandable.
- [ ] Cache behavior across repeated turns matches the change intent.

Optional checks:

- [ ] Cache behavior with summary enabled.
- [ ] Cache behavior after changing session identifiers.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Model And GGUF Discovery Checks

Required when:

- GGUF discovery, model path handling, model loading, chat handler selection, or
  model-family routing changed.

Skip when:

- The change does not affect model discovery or loading.

Required checks:

- [ ] Expected GGUF files appear in selectable model inputs.
- [ ] Case-insensitive discovery behavior works when relevant.
- [ ] A supported text model loads.
- [ ] A minimal generation works after model load.
- [ ] Model loading errors are understandable when a model is missing or invalid.

Optional checks:

- [ ] Model-family-specific parameter behavior matches the change intent.
- [ ] Unload model behavior still works when relevant.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Multimodal Checks

Required when:

- Image input handling, vision model routing, multimodal chat handler selection,
  or image-capable model behavior changed.

Skip when:

- The change only affects text-only generation paths.
- No image-capable model is available locally. Record this as not performed, not
  passed.

Required checks:

- [ ] Image input can be connected to the affected node.
- [ ] A supported image-capable model can be selected.
- [ ] A minimal image prompt produces a response.
- [ ] Text-only generation still works when no image is provided.
- [ ] Missing multimodal dependency or handler errors are understandable.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Error Handling Checks

Required when:

- User-visible error messages, logging, exception handling, fallback behavior, or
  validation changed.

Skip when:

- The change does not affect runtime errors, logs, or validation paths.

Required checks:

- [ ] The expected error path can be triggered.
- [ ] The UI or console message is understandable.
- [ ] The log includes enough detail for troubleshooting.
- [ ] Sensitive or excessively noisy details are not exposed.
- [ ] Normal successful execution still works.

Result:

- [ ] Passed
- [ ] Failed
- [ ] Not applicable

Notes:

- 

## Final Manual Validation Summary

- Manual ComfyUI validation: performed / required but not performed / not required
- Sections selected:
- Sections completed:
- Environment:
- Model used:
- Result:
- Remaining risk:
