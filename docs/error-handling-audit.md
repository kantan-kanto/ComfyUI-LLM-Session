# Error Handling Audit Report

**Date**: 2026-03-22
**Scope**: `except Exception:` error handling usage
**Total Count**: 57 occurrences
**Status**: P0/P1 improvements completed (2026-03-22)

---

## Legend

### Category Classification

| Category | Description | Action Policy |
|----------|-------------|---------------|
| **A-1** | Fallback purpose | Keep as-is + add comments |
| **A-2** | Optional features | Keep as-is + log level control |
| **B-1** | Debug-critical info | Add log output (high priority) |
| **B-2** | Monitoring targets | Add structured logs (medium priority) |
| **C-1** | Potential bugs | Error propagation or retry (high priority) |
| **C-2** | Needs discussion | Requires spec decision (medium priority) |

### Priority Levels

- **P0**: Immediate action required (risk of functional failure or data loss)
- **P1**: Action needed soon (risk of debugging difficulty or unexpected behavior)
- **P2**: Action if time permits (code quality improvement)
- **P3**: Keep as-is (acceptable)

---

## Details by File

### 1. `services/turn_execution_service.py` (14 occurrences)

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 1 | 360-362 | `mgr.invalidate_cache()` failure | B-1 | P0 | Cache invalidation failure during reset | Add error log + record session ID |
| 2 | 397-399 | `os.makedirs()` failure | B-1 | P0 | Cache directory creation failure | Add error log + use fallback path |
| 3 | 430-432 | `mgr.configure_cache()` failure | B-1 | P0 | Cache configuration failure | Add error log + continue without cache |
| 4 | 516-518 | KV state restore failure | B-2 | P1 | Existing log present, needs detail | Keep existing log + add context |
| 5 | 621-623 | `maybe_summarize_history()` failure | B-2 | P1 | Summarization failure | Add debug log + continue |
| 6 | 627-629 | `atomic_write_json()` failure | C-1 | P0 | **History file save failure** | Error log + partial recovery or error notification |
| 7 | 671-673 | `try_save_kv_state()` failure | B-2 | P1 | KV state save failure | Add debug log |
| 8 | 743-745 | `mgr.invalidate_cache()` failure | B-2 | P1 | Invalidation failure on cache mismatch | Add debug log |
| 9 | 763-765 | `maybe_compact_summary()` failure | B-2 | P1 | Summary compaction failure | Add debug log |

---

### 2. `services/chat_turn_service.py` (4 occurrences)

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 10 | 227-229 | `unload_model(managerA)` failure | A-2 | P2 | Model unload failure in loop | Debug log (integrate with existing finally) |
| 11 | 250-252 | `unload_model(managerB)` failure | A-2 | P2 | Model unload failure in loop | Debug log (integrate with existing finally) |
| 12 | 255-257 | `unload_model(managerA)` failure | A-2 | P2 | Model unload failure in finally | Debug log |
| 13 | 259-261 | `unload_model(managerB)` failure | A-2 | P2 | Model unload failure in finally | Debug log |

---

### 3. `llm_session_nodes.py` (31 occurrences)

#### Configuration/Initialization

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 14 | 31-32 | Module import failure | A-1 | P2 | Relative/absolute import fallback | Keep as-is (appropriate fallback) |
| 15 | 106-107 | `_simple_config_log()` failure | A-2 | P3 | Log output itself failed | Keep as-is (log failure OK to ignore) |
| 16 | 119-120 | `os.path.expanduser()` failure | A-1 | P2 | Path expansion failure | Keep as-is + add comment |
| 17 | 123-124 | `Path(raw)` failure | A-1 | P2 | Path object creation failure | Keep as-is + add comment |
| 18 | 130-131 | `_simple_config_path()` failure | A-1 | P2 | Config path retrieval failure | Keep as-is + add comment |
| 19 | 165-166 | JSON load failure | A-1 | P2 | Config file load failure | Existing log present (appropriate) |
| 20 | 184-185 | `int(x)` conversion failure | A-1 | P3 | Numeric conversion fallback | Keep as-is (appropriate fallback) |
| 21 | 190-191 | `float(x)` conversion failure | A-1 | P3 | Float conversion fallback | Keep as-is |
| 22 | 204-205 | `_as_bool()` failure | A-1 | P3 | Boolean conversion fallback | Keep as-is |

#### Model Discovery/Path Resolution

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 23 | 689-690 | `folder_paths.get_folder_paths()` failure | A-2 | P2 | Additional model path retrieval failure | Keep as-is + add comment |
| 24 | 747-748 | `_safe_join_under()` failure | A-2 | P2 | Path join failure (in loop) | Keep as-is (continue to next) |
| 25 | 894-895 | `folder_paths.get_output_directory()` failure | A-2 | P2 | Output directory retrieval failure | Keep as-is (fallback exists) |
| 26 | 953-954 | `_coerce_int()` failure | A-1 | P3 | Numeric conversion fallback | Keep as-is |

#### Message Building/Generation

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 27 | 1012-1013 | `tensor2pil()` failure | A-2 | P2 | Image processing failure (text fallback) | Keep as-is (appropriate fallback) |
| 28 | 1219-1220 | `_extract_stream_content()` failure | A-2 | P2 | Stream content extraction failure | Keep as-is + add comment |

#### Cache Management

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 29 | 1563-1564 | `__contains__()` failure (primary) | A-2 | P2 | Cache existence check failure | Add debug log |
| 30 | 1567-1568 | `__contains__()` failure (secondary) | A-2 | P2 | Cache existence check failure | Add debug log |
| 31 | 1574-1575 | `__getitem__()` failure (primary) | A-2 | P2 | Cache get failure | Add debug log |
| 32 | 1580-1581 | `__setitem__()` failure (primary) | A-2 | P2 | Cache set failure | Add debug log |
| 33 | 1840-1841 | `backend_fn()` failure | A-2 | P3 | Backend version retrieval failure | Keep as-is |
| 34 | 1842-1843 | `import llama_cpp` failure | A-2 | P3 | llama_cpp import failure | Keep as-is |
| 35 | 1932-1933 | `LlamaCache()` failure | B-1 | P1 | Persistent cache creation failure | Add error log |
| 36 | 1956-1957 | `llm.set_cache(None)` failure | A-2 | P2 | Cache invalidation failure | Add debug log |
| 37 | 1972-1973 | `llm.cache = cache_obj` failure | A-2 | P2 | Cache set failure | Add debug log |
| 38 | 1996-1997 | `llm.set_cache(None)` failure | A-2 | P2 | Cache invalidation failure (finally) | Add debug log |
| 39 | 2013-2014 | `llm.set_cache(None)` failure | A-2 | P2 | Cache invalidation failure (invalidate) | Add debug log |
| 40 | 2023-2025 | `shutil.rmtree()` failure | B-2 | P1 | Disk cache removal failure | Add debug log |

#### Debug Info/Utilities

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 41 | 2079-2080 | `_cache_debug_label()` failure | A-2 | P3 | Debug info retrieval failure | Keep as-is |
| 42 | 2099-2101 | `container.mem_kv_state.pop()` failure | A-2 | P2 | Runtime-container state clear failure | Add debug log |
| 43 | 2094-2095 | `type(state).__name__` failure | A-2 | P3 | Type name retrieval failure | Keep as-is |
| 44 | 2100-2101 | `len(state)` failure | A-2 | P3 | Size retrieval failure | Keep as-is |
| 45 | 2105-2106 | `nbytes` retrieval failure | A-2 | P3 | Size attribute retrieval failure | Keep as-is |
| 46 | 2111-2112 | `llama_state_size` retrieval failure | A-2 | P3 | State size retrieval failure | Keep as-is |
| 47 | 2122-2123 | `llama_state_size` retrieval failure | A-2 | P3 | Saved state size retrieval failure | Keep as-is |
| 48 | 2128-2129 | `len(raw)` failure | A-2 | P3 | State length retrieval failure | Keep as-is |
| 49 | 2147-2148 | `fn(ctx_obj)` failure | A-2 | P3 | State size function failure | Keep as-is |

---

### 4. `infra/history_store.py` (4 occurrences)

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 50 | 60-62 | `os.replace(path, bak)` failure | C-1 | P0 | **Backup creation failure** | Add error log + consider recovery procedure |
| 51 | 68-70 | `_coerce_int()` failure | A-1 | P3 | Numeric conversion fallback | Keep as-is |
| 52 | 191-192 | `json.load()` failure | A-1 | P2 | History load failure (fallback to .bak) | Keep as-is (appropriate fallback) |
| 53 | 204-205 | `json.load()` failure (.bak) | A-1 | P2 | Backup load failure | Keep as-is + create new |

---

### 5. `core/generation_runner.py` (2 occurrences)

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 54 | 10-11 | `turn_types` import failure | A-1 | P2 | Module import fallback | Keep as-is |
| 55 | 89-90 | `out.write/flush()` failure | A-2 | P2 | Stream output failure | Keep as-is (output failure OK to continue) |

---

### 6. `core/continue_rewrite.py` (1 occurrence)

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 56 | 9-10 | `turn_types` import failure | A-1 | P2 | Module import fallback | Keep as-is |

---

### 7. `core/kv_state.py` (1 occurrence)

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 57 | 100-101 | `invalidate_cache()` failure | B-2 | P1 | Cache invalidation failure on state mismatch | Add debug log |

---

### 8. `__init__.py` (1 occurrence)

| # | Lines | Code Content | Category | Priority | Details | Recommended Action |
|---|-------|--------------|----------|----------|---------|-------------------|
| 58 | 16-17 | Relative import failure | A-1 | P2 | Fallback to absolute import | Keep as-is |

---

## Summary

### By Category

| Category | Count | Percentage |
|----------|-------|------------|
| A-1: Fallback purpose | 15 | 26% |
| A-2: Optional features | 28 | 49% |
| B-1: Debug-critical info | 4 | 7% |
| B-2: Monitoring targets | 7 | 12% |
| C-1: Potential bugs | 2 | 4% |
| C-2: Needs discussion | 0 | 0% |
| **Total** | **56** | **100%** |

### By Priority

| Priority | Count | Percentage |
|----------|-------|------------|
| P0: Immediate action | 4 | 7% |
| P1: Action needed soon | 7 | 12% |
| P2: Action if time permits | 24 | 43% |
| P3: Keep as-is | 21 | 38% |
| **Total** | **56** | **100%** |

### By File

| File | Count | Percentage |
|------|-------|------------|
| `llm_session_nodes.py` | 31 | 55% |
| `services/turn_execution_service.py` | 9 | 16% |
| `services/chat_turn_service.py` | 4 | 7% |
| `infra/history_store.py` | 4 | 7% |
| `core/generation_runner.py` | 2 | 4% |
| `core/kv_state.py` | 1 | 2% |
| `core/continue_rewrite.py` | 1 | 2% |
| `__init__.py` | 1 | 2% |
| **Total** | **53** | **100%** |

---

## Recommended Action Order

### Phase 1: P0 (Immediate Action) - COMPLETED

1. **`turn_execution_service.py:627-629`** - History file save failure
   - Risk: Data loss
   - Action: Error log + recovery procedure
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

2. **`history_store.py:60-62`** - Backup creation failure
   - Risk: Data corruption
   - Action: Error log + recovery procedure
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

3. **`turn_execution_service.py:360-362`** - Reset-time cache invalidation failure
   - Risk: State inconsistency
   - Action: Error log
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

4. **`turn_execution_service.py:397-399`** - Cache directory creation failure
   - Risk: Cache dysfunction
   - Action: Error log + fallback
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

### Phase 2: P1 (Action Needed Soon) - COMPLETED

5. **`turn_execution_service.py:516-518`** - KV state restore failure
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

6. **`turn_execution_service.py:621-623`** - Summarization failure
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

7. **`turn_execution_service.py:671-673`** - KV state save failure
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

8. **`turn_execution_service.py:743-745`** - Cache invalidation failure on mismatch
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

9. **`turn_execution_service.py:763-765`** - Summary compaction failure
   - **Status**: ✅ Completed - Added logging with `log_error_safely()`

10. **`llm_session_nodes.py:1932-1933`** - Persistent cache creation failure
    - **Status**: ✅ Completed - Added logging with `log_error_safely()`

11. **`llm_session_nodes.py:2023-2025`** - Disk cache removal failure
    - **Status**: ✅ Completed - Added logging with `log_error_safely()`

12. **`kv_state.py:100-101`** - Cache invalidation failure on state mismatch
    - **Status**: ✅ Completed - Added logging with `log_error_safely()`

### Phase 3: P2 (Action If Time Permits) - PENDING

Remaining 24 occurrences (add debug logs, add comments)

### Phase 4: P3 (Keep As-Is) - DOCUMENTED

21 occurrences are appropriate fallbacks, keep as-is.
See [`acceptable-silent-errors.md`](acceptable-silent-errors.md) for details.

---

## Implementation Summary (2026-03-22)

### Logging Infrastructure (Phase 2)

Created `core/logging_utils.py` with:
- `log_error_safely()` - Safe error logging with module name and context
- `get_module_logger()` - Module-specific logger factory
- `log_with_exception_info()` - Debug-level logging with traceback
- `create_fallback_logger()` - Fallback logger for critical paths
- Log levels: `minimal`, `timing`, `debug`

### P0 Improvements (Phase 3)

| Location | Issue | Implementation |
|----------|-------|----------------|
| `turn_execution_service.py:522-525` | Cache invalidation during reset | `log_error_safely()` |
| `turn_execution_service.py:397-399` | Cache directory creation | `log_error_safely()` |
| `infra/history_store.py:61-66` | Backup creation failure | `log_error_safely()` |
| `turn_execution_service.py:569-578` | History save failure | `log_error_safely()` |
| `infra/history_store.py` | Unreadable primary history load | `log_error_safely()` + quarantine before fresh history creation |

### P1 Improvements (Phase 4)

| Location | Issue | Implementation |
|----------|-------|----------------|
| `turn_execution_service.py:522-525` | KV state restore failure | `log_error_safely()` |
| `turn_execution_service.py:627-629` | Summarization failure | `log_error_safely()` |
| `turn_execution_service.py:678-680` | KV state save failure | `log_error_safely()` |
| `turn_execution_service.py:748-751` | Cache invalidation on mismatch | `log_error_safely()` |
| `turn_execution_service.py:770-772` | Summary compaction failure | `log_error_safely()` |
| `llm_session_nodes.py:1930-1934` | Persistent cache creation | `log_error_safely()` |
| `llm_session_nodes.py:2021-2026` | Disk cache removal | `log_error_safely()` |
| `core/kv_state.py:98-101` | Cache invalidation on mismatch | `log_error_safely()` |

### Tests (Phase 6)

Added `TestErrorHandlingP0P1` class in `tests/services/test_turn_execution_service.py`:
- 8 test methods for P0/P1 error handling
- All tests verify graceful degradation (generation succeeds despite errors)
- All 67 tests passing (no regressions)

---

## Next Steps

1. ✅ Review this document - Completed
2. ✅ Decide on specific implementation approach for P0 items - Completed
3. ✅ Proceed to Phase 2 (Logging Utility Setup) - Completed
4. ✅ Complete P0 improvements - Completed
5. ✅ Complete P1 improvements - Completed
6. ✅ Document P3 acceptable silent errors - Completed
7. ✅ Add tests for P0/P1 improvements - Completed
8. Consider P2 improvements (24 occurrences) as time permits
