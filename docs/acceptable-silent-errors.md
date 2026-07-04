# Acceptable Silent Errors

- Status: Historical
- Last reviewed: 2026-07-05
- Update when: Clarifying historical status, fixing stale cross-references, or adding supersession notes.

Historical note: This document is the silent-error allowlist from the 2026-03-22
audit. Line-number references below refer to the code layout at the time of that
audit and may not match current source files.

**Date**: 2026-03-22  
**Purpose**: Document `except Exception: pass` patterns that are intentionally kept silent

This document lists error handling locations where silent failure was considered
acceptable and intentional during the 2026-03-22 audit.

For the current policy on when a silent handler is acceptable, see
[`logging-guidelines.md`](logging-guidelines.md#when-silence-is-acceptable).

---

## P3 Silent Errors by Category

### 1. Logging Infrastructure Failures

| Location | Code | Rationale |
|----------|------|-----------|
| `llm_session_nodes.py:106-107` | `_simple_config_log()` failure | Logging itself failed; logging the failure would be circular and add no value |

**Why silent is acceptable**: If the logging function itself fails, attempting to log that failure would either fail again or create infinite recursion. The failure of optional logging is not user-impacting.

---

### 2. Numeric Conversion Fallbacks

| Location | Code | Rationale |
|----------|------|-----------|
| `llm_session_nodes.py:184-185` | `int(x)` conversion failure | Falls back to default value; conversion failure is expected for invalid user input |
| `llm_session_nodes.py:190-191` | `float(x)` conversion failure | Falls back to default value; conversion failure is expected for invalid user input |
| `llm_session_nodes.py:204-205` | `_as_bool()` failure | Falls back to default value; conversion failure is expected for invalid user input |
| `llm_session_nodes.py:953-954` | `_coerce_int()` failure | Falls back to default value; same pattern as above |
| `infra/history_store.py:68-70` | `_coerce_int()` failure | Falls back to default value; same pattern as above |

**Why silent is acceptable**: These are defensive fallbacks for potentially invalid configuration values. The fallback to default values is the intended behavior, and logging every invalid value would create noise without actionable information.

---

### 3. Backend/Version Detection Failures

| Location | Code | Rationale |
|----------|------|-----------|
| `llm_session_nodes.py:1840-1841` | `backend_fn()` failure | Backend version retrieval is best-effort; failure does not affect core functionality |
| `llm_session_nodes.py:1842-1843` | `import llama_cpp` failure | Import failure is handled gracefully; logging would duplicate other error paths |

**Why silent is acceptable**: These are informational queries about the backend environment. Failure to retrieve version information does not affect the ability to run inference.

---

### 4. Debug Information Retrieval Failures

| Location | Code | Rationale |
|----------|------|-----------|
| `llm_session_nodes.py:2079-2080` | `_cache_debug_label()` failure | Debug-only function; failure does not affect production behavior |
| `llm_session_nodes.py:2094-2095` | `type(state).__name__` failure | Debug info retrieval; failure does not affect core functionality |
| `llm_session_nodes.py:2100-2101` | `len(state)` failure | Debug info retrieval; failure does not affect core functionality |
| `llm_session_nodes.py:2105-2106` | `nbytes` retrieval failure | Debug info retrieval; failure does not affect core functionality |
| `llm_session_nodes.py:2111-2112` | `llama_state_size` retrieval failure | Debug info retrieval; failure does not affect core functionality |
| `llm_session_nodes.py:2122-2123` | `llama_state_size` retrieval failure | Debug info retrieval; failure does not affect core functionality |
| `llm_session_nodes.py:2128-2129` | `len(raw)` failure | Debug info retrieval; failure does not affect core functionality |
| `llm_session_nodes.py:2147-2148` | `fn(ctx_obj)` failure | Debug info retrieval; failure does not affect core functionality |

**Why silent is acceptable**: All of these are used exclusively for debug output and diagnostics. Failure to retrieve debug information does not affect the core functionality of the application. Adding logging here would create noise during normal operation when debug info is not needed.

---

## Summary

| Category | Count | Files |
|----------|-------|-------|
| Logging Infrastructure | 1 | `llm_session_nodes.py` |
| Numeric Conversion Fallbacks | 5 | `llm_session_nodes.py`, `infra/history_store.py` |
| Backend/Version Detection | 2 | `llm_session_nodes.py` |
| Debug Information Retrieval | 8 | `llm_session_nodes.py` |
| **Total** | **16** | 2 files |

## Related Documents

- [`error-handling-audit.md`](error-handling-audit.md) - Full audit of all `except Exception:` occurrences
- [`logging-guidelines.md`](logging-guidelines.md#when-silence-is-acceptable) - Current silent-handler policy and review triggers
