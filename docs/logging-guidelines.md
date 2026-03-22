# Logging Guidelines

This document describes the logging conventions and best practices for ComfyUI-LLM-Session.

## Log Levels

The project uses three log levels:

| Level | Description | When to Use |
|-------|-------------|-------------|
| `minimal` | Only critical errors | Production environments, minimal console output |
| `timing` | Timing information + critical errors (default) | Normal development and usage |
| `debug` | Full debug output including tracebacks | Debugging issues, development |

## Logging Utilities

### Core Module: `core.logging_utils`

The `core.logging_utils` module provides common logging helpers:

```python
from core.logging_utils import (
    get_module_logger,
    log_error_safely,
    log_with_exception_info,
    create_fallback_logger,
    handle_exception_safely,
    LOG_LEVEL_MINIMAL,
    LOG_LEVEL_TIMING,
    LOG_LEVEL_DEBUG,
)
```

### Getting a Module Logger

```python
# Create a logger for your module
log = get_module_logger("MyModuleName")

# Use the logger
log("Operation completed", LOG_LEVEL_TIMING)
log("Debug detail", LOG_LEVEL_DEBUG)
```

### Logging Errors in Except Blocks

**Recommended pattern:**

```python
from core.logging_utils import log_error_safely, LOG_LEVEL_MINIMAL

try:
    # some operation
    result = do_something()
except Exception as e:
    log_error_safely("MyModule", e, "Failed to do something", LOG_LEVEL_MINIMAL)
    return None  # or appropriate fallback
```

**Alternative pattern with fallback logger:**

```python
from core.logging_utils import create_fallback_logger, LOG_LEVEL_MINIMAL

# Create fallback logger at module level
_log_error = create_fallback_logger("MyModule", LOG_LEVEL_MINIMAL)

def my_function():
    try:
        # some operation
        result = do_something()
    except Exception as e:
        _log_error(e, "Failed to do something")
        return None
```

### Logging with Full Exception Info

For debug-level logging with traceback information:

```python
from core.logging_utils import log_with_exception_info, LOG_LEVEL_DEBUG

try:
    # some operation
    result = do_something()
except Exception:
    log_with_exception_info("MyModule", "Failed to do something", level=LOG_LEVEL_DEBUG)
    return None
```

## Error Handling Patterns

### Pattern 1: Fallback with Logging

Use when the operation has a safe fallback:

```python
from core.logging_utils import log_error_safely, LOG_LEVEL_TIMING

def load_optional_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error_safely("ConfigLoader", e, "Failed to load optional config", LOG_LEVEL_TIMING)
        return {}  # safe fallback
```

### Pattern 2: Critical Error with Propagation

Use when the operation must succeed:

```python
from core.logging_utils import log_error_safely, LOG_LEVEL_MINIMAL

def load_required_history(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error_safely("HistoryLoader", e, "Failed to load required history", LOG_LEVEL_MINIMAL)
        raise  # re-raise for caller to handle
```

### Pattern 3: Silent Error (Acceptable)

Use only for truly optional operations where logging would add noise:

```python
def cleanup_temp_file(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass  # Acceptable: temp file cleanup is best-effort
```

**Note:** Silent errors (Pattern 3) should only be used when:
1. The operation is truly optional
2. Failure has no impact on user experience
3. Logging would add noise without value
4. The error is documented in `error-handling-audit.md`

## Best Practices

### DO:

1. **Use module-specific loggers** for clear log attribution
2. **Include context** in error messages (what operation was being attempted)
3. **Use appropriate log levels** (minimal for errors, timing for info, debug for verbose)
4. **Log before falling back** so users know when fallbacks are triggered
5. **Keep log messages concise** but informative

### DON'T:

1. **Don't use bare `except:`** - always use `except Exception:`
2. **Don't log in loops** without rate limiting
3. **Don't log sensitive information** (API keys, personal data)
4. **Don't use print() directly** - use logging utilities
5. **Don't silently swallow errors** without documentation

## Migration Guide

### Replacing Existing Patterns

**Old pattern:**
```python
def _simple_config_log(message: str, log_level: str) -> None:
    try:
        if str(log_level).strip().lower() != "minimal":
            print(f"[LLM Session Simple] {message}")
    except Exception:
        pass
```

**New pattern:**
```python
from core.logging_utils import get_module_logger, LOG_LEVEL_TIMING

log = get_module_logger("LLM Session Simple")

# Usage
log("Config loaded", LOG_LEVEL_TIMING)
```

**Old pattern:**
```python
try:
    # operation
except Exception:
    pass  # Silent failure
```

**New pattern:**
```python
from core.logging_utils import log_error_safely, LOG_LEVEL_MINIMAL

try:
    # operation
except Exception as e:
    log_error_safely("MyModule", e, "operation description", LOG_LEVEL_MINIMAL)
    # fallback or re-raise
```

## Configuration

### Setting Log Levels

```python
from core.logging_utils import set_global_log_level, set_module_log_level

# Set global log level
set_global_log_level("debug")

# Set module-specific log level
set_module_log_level("MyModule", "minimal")
```

### Log Level Hierarchy

Log levels follow this hierarchy (most verbose to least verbose):
1. `debug` - Shows all messages
2. `timing` - Shows timing + minimal messages
3. `minimal` - Shows only critical errors

## File Locations

- `core/logging_utils.py` - Logging utility implementations
- `docs/logging-guidelines.md` - This document
- `docs/error-handling-audit.md` - Audit of all error handling locations
