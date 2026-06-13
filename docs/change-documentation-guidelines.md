# Change Documentation Guidelines

This document defines how to document and describe changes in this repository.
Use it when adding features, changing behavior, updating compatibility notes,
writing commit messages, preparing changelog entries, or drafting GitHub release
notes.

## Document Roles

### `README.md` / `README.ja.md`

Entry points for first-time users.

Use these files for:

- Project overview
- Provided nodes
- Installation notes
- Minimal model placement and first-use guidance
- Short links to detailed reference documents
- Important constraints users must know before trying the project

Avoid placing long compatibility notes, detailed runtime behavior, or full
parameter explanations in the READMEs. Prefer short summaries with links to the
appropriate reference document.

Short duplication is acceptable when it helps first-time users quickly decide
whether the project is relevant or current. For example, the README may include:

- A compact tested-model summary, while detailed backend results and caveats
  remain canonical in `COMPATIBILITY.md`.
- A short current-version summary, while full release history remains canonical
  in `CHANGELOG.md`.

### `PARAMETERS.md`

Reference for settings and runtime behavior.

Use this file for:

- UI parameters
- Simple-node default behavior
- `config_path` behavior
- Cache, history, session, retry, fallback, and logging behavior
- Practical guidance for how settings affect execution
- Model-agnostic behavior such as chat template fallback

If the question is "what does this setting do?" or "what happens at runtime?",
the answer usually belongs here.

### `ADVANCED_PARAMETERS.md`

Reference for advanced Simple-node JSON settings.

Use this file for:

- Settings available only through Simple-node JSON config
- Advanced generation options
- Experimental or future-facing JSON keys
- Carefully-use settings such as fixed seeds
- Notes about reproducibility and backend-sensitive advanced behavior

If a setting is intentionally not exposed in the UI and is configured through
JSON, document it here.

### `COMPATIBILITY.md`

Reference for model and backend compatibility.

Use this file for:

- Tested model families
- Vision model behavior
- mmproj and chat handler compatibility
- Backend-dependent behavior
- `llama-cpp-python` version or fork differences
- Empirical compatibility results and caveats

Avoid general usage notes, cache operation details, or migration notes unless
they directly affect model/backend compatibility.

### `CHANGELOG.md`

Release history.

Use this file for:

- Every user-visible release change
- New features
- Bug fixes
- Breaking or migration-relevant changes
- Compatibility changes
- Documentation updates

The changelog should summarize what changed and point to the relevant detailed
document when needed.

## Commit Message Guidelines

Use commit messages as short, precise records of individual changes.

### Format

Prefer this format:

```text
area: Imperative summary
```

Examples:

```text
docs: Clarify Simple-node config behavior
compat: Add MiniCPM-V 4.6 aliases
runtime: Fix cache invalidation after reset
release: Prepare v1.2.4
```

The `area:` prefix is recommended when it adds useful context. For very small
or obvious changes, a plain imperative summary is acceptable.

### Style

- Use imperative mood: `Add`, `Fix`, `Update`, `Clarify`, `Refine`,
  `Prepare`.
- Keep the first line concise, ideally around 72 characters or less.
- Do not end the subject line with a period.
- Mention the user-visible behavior when possible.
- Keep unrelated changes in separate commits when practical.

### Suggested Areas

- `docs`: documentation-only changes
- `compat`: model, backend, Vision, or llama-cpp-python compatibility
- `runtime`: execution, cache, history, session, or fallback behavior
- `ui`: node inputs, labels, tooltips, or exposed parameters
- `tests`: tests and test infrastructure
- `release`: version, changelog, and release preparation

## GitHub Release Guidelines

GitHub release notes should be easier to read than the changelog. They should
highlight what users need to know first, then group details by category.

### Release Title

Use this format:

```text
vX.Y.Z – Short Theme
```

Examples:

```text
v1.2.4 – Documentation Cleanup and Backend Compatibility Notes
v1.3.0 – New Simple Config Controls and Vision Diagnostics
```

Use an en dash (`–`) between the version and theme.

### Release Notes Template

Use only the sections that apply.

```md
This release focuses on ...

### Highlights

- ...

### Added

- ...

### Changed

- ...

### Fixed

- ...

### Compatibility

- ...

### Documentation

- ...

### Tests

- ...

### Upgrade Notes

- No action required.
```

### Section Meanings

- `Highlights`: the most important user-facing summary
- `Added`: new features, nodes, parameters, or config keys
- `Changed`: behavior changes that are not bug fixes
- `Fixed`: bug fixes
- `Compatibility`: model, Vision, backend, or llama-cpp-python changes
- `Documentation`: README, parameter docs, compatibility docs, or guidance
- `Tests`: new or changed regression coverage
- `Upgrade Notes`: required user actions or migration notes

Omit empty sections. If no action is required, say so clearly under
`Upgrade Notes` only when users might otherwise wonder.

## Changelog Guidelines

`CHANGELOG.md` should be more complete and factual than GitHub release notes.

- Record every user-visible change.
- Group entries by release version.
- Use categories similar to release notes when helpful.
- Link or point to detailed docs rather than duplicating long explanations.
- Include documentation-only changes when they affect user guidance.

## Placement Rules

- If first-time users need it before trying the project, put a short note in
  `README.md` and `README.ja.md`.
- If it is a setting or runtime behavior, put it in `PARAMETERS.md`.
- If it is a Simple-node JSON-only setting, put it in `ADVANCED_PARAMETERS.md`.
- If it depends on model family, Vision support, mmproj, chat handlers, or
  `llama-cpp-python` builds, put it in `COMPATIBILITY.md`.
- If it describes when a change happened, put it in `CHANGELOG.md`.
- If a topic spans multiple documents, keep the README short and link to the
  detailed reference document.
- If a README summary intentionally duplicates reference content, keep it short,
  user-facing, and linked to the canonical document.

## Common Examples

### Adding a New Node

- `README.md` / `README.ja.md`: add a short entry under the provided nodes.
- `PARAMETERS.md`: document any settings or runtime behavior.
- `CHANGELOG.md`: record the new node.

### Adding a New UI Parameter

- `PARAMETERS.md`: document the parameter, valid values, defaults, and caveats.
- `README.md` / `README.ja.md`: add only if it is important for first-time use.
- `CHANGELOG.md`: record the addition.

### Adding a New Simple JSON Advanced Key

- `ADVANCED_PARAMETERS.md`: document the key and usage guidance.
- `PARAMETERS.md`: link or mention it only if it relates to a general setting.
- `CHANGELOG.md`: record the addition.

### Adding or Testing a New Model Family

- `COMPATIBILITY.md`: record tested behavior, backend requirements, and caveats.
- `README.md`: update model family aliases only if users need them for model or
  mmproj placement.
- `CHANGELOG.md`: record the compatibility update.

### Changing Cache, History, or Session Behavior

- `PARAMETERS.md`: document the new behavior and operational implications.
- `README.md` / `README.ja.md`: add a short upgrade note only if existing users
  must take action.
- `CHANGELOG.md`: record the change.

### Changing Installation Requirements

- `README.md` / `README.ja.md`: update the installation path.
- `COMPATIBILITY.md`: update only if backend or model compatibility changes.
- `CHANGELOG.md`: record the dependency or installation change.

## Style Notes

- Keep README sections short and user-facing.
- Keep reference documents specific and factual.
- Prefer stable descriptions over implementation details unless the behavior is
  user-visible.
- Do not duplicate long explanations across documents. Link to the canonical
  reference instead.
- When updating English README content that affects first-time use, update
  `README.ja.md` with the same intent.
- Keep commit messages and release notes consistent in vocabulary:
  - Use `Vision`, `backend`, `Simple node`, `Simple-node JSON config`, and
    `llama-cpp-python` consistently.
  - Prefer `compatibility` for model/backend support and `runtime behavior` for
    execution semantics.
  - Prefer `release` over `version bump` when describing release preparation.

## AI Agent Checklist

When a user asks an AI assistant to update documentation according to this
guide, the assistant should complete the following checklist.

### 1. Review the Change Scope

- Identify the actual user-visible change, feature, fix, compatibility update,
  or documentation cleanup.
- Determine which documents are canonical for the change using the document
  roles and placement rules above.
- Check whether English and Japanese README content should be kept in sync.
- Avoid editing unrelated sections only for style preferences.

### 2. Update the Documentation

- Put short first-use guidance in `README.md` / `README.ja.md`.
- Put settings and runtime behavior in `PARAMETERS.md`.
- Put Simple-node JSON-only advanced settings in `ADVANCED_PARAMETERS.md`.
- Put model/backend compatibility in `COMPATIBILITY.md`.
- Put release-history facts in `CHANGELOG.md`.
- Remove duplicate explanations when a single canonical reference is better.
- Ensure cross-document links point to the canonical reference document.

### 3. Verify the Result

- Review the affected sections for stale wording, duplicated guidance, and
  inconsistent terminology.
- Check that headings still form a natural reading order.
- Check local Markdown links when files were moved, renamed, or newly linked.
- Confirm that README content remains concise and that detailed explanations
  live in the reference documents.

### 4. Output a Commit Message

After updating the documentation, provide one recommended commit message using
the commit message guidelines above.

Prefer this format:

```text
docs: Imperative summary of the documentation update
```

Example:

```text
docs: Organize Simple-node settings and compatibility references
```

If the change includes release preparation, use:

```text
release: Prepare vX.Y.Z
```

### 5. Output GitHub Release Notes

Provide draft GitHub release notes using the release guidelines above. Include
only sections that apply.

For documentation-only work, this compact form is usually enough:

```md
This release focuses on documentation clarity and maintenance guidance.

### Documentation

- ...

### Upgrade Notes

- No action required.
```

For feature or behavior changes, include the relevant `Added`, `Changed`,
`Fixed`, `Compatibility`, `Documentation`, `Tests`, and `Upgrade Notes`
sections.

### 6. Final Response Requirements

In the final response, include:

- A concise summary of the documentation updates.
- The recommended commit message.
- Draft GitHub release notes.
- Any verification that was performed, or a note if no tests were run because
  the change was documentation-only.
