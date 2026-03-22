# logging_utils.py
"""
Logging utilities for ComfyUI-LLM-Session.

Provides common logging helpers with support for log levels:
- minimal: Only critical errors
- timing: Timing information + critical errors
- debug: Full debug output
"""

import os
import sys
from datetime import datetime
from typing import Optional, Callable, Any, Dict


# Log level constants
LOG_LEVEL_MINIMAL = "minimal"
LOG_LEVEL_TIMING = "timing"
LOG_LEVEL_DEBUG = "debug"

# Global log level (can be overridden per-module)
_global_log_level: str = LOG_LEVEL_TIMING

# Module-specific log level overrides
_module_log_levels: Dict[str, str] = {}


def set_global_log_level(level: str) -> None:
    """Set the global log level for all modules."""
    global _global_log_level
    _global_log_level = level


def set_module_log_level(module_name: str, level: str) -> None:
    """Set log level for a specific module."""
    _module_log_levels[module_name] = level


def get_log_level(module_name: Optional[str] = None) -> str:
    """Get the effective log level for a module."""
    if module_name and module_name in _module_log_levels:
        return _module_log_levels[module_name]
    return _global_log_level


def is_log_enabled(level: str, module_name: Optional[str] = None) -> bool:
    """
    Check if logging at the given level is enabled.
    
    Level hierarchy (by importance):
    - minimal: Most important (always enabled)
    - timing: Medium importance
    - debug: Least important (only when debug level is set)
    
    When effective level is:
    - debug: All levels enabled
    - timing: timing and minimal enabled
    - minimal: only minimal enabled
    """
    effective_level = get_log_level(module_name)
    
    # minimal is always enabled (most important)
    if level == LOG_LEVEL_MINIMAL:
        return True
    
    # timing is enabled when effective level is timing or debug
    if level == LOG_LEVEL_TIMING:
        return effective_level in (LOG_LEVEL_TIMING, LOG_LEVEL_DEBUG)
    
    # debug is only enabled when effective level is debug
    if level == LOG_LEVEL_DEBUG:
        return effective_level == LOG_LEVEL_DEBUG
    
    return False


def get_module_logger(module_name: str) -> Callable[[str, str], None]:
    """
    Get a logger function for a specific module.
    
    Returns a function that takes (message, level) and logs appropriately.
    
    Args:
        module_name: Name of the module (used for log prefix)
    
    Returns:
        A callable that accepts (message: str, level: str) -> None
    """
    def log(message: str, level: str = LOG_LEVEL_TIMING) -> None:
        # Check if this log level is enabled for the module
        # The level parameter indicates the importance of the message
        # minimal = most important (always logged)
        # timing = medium importance (logged when level is timing or debug)
        # debug = least important (only logged when level is debug)
        if not _should_log(level, module_name):
            return
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}][{module_name}] {message}")
        except Exception:
            # Never fail during logging
            pass
    
    return log


def _should_log(message_level: str, module_name: Optional[str] = None) -> bool:
    """
    Internal helper to check if a message at the given level should be logged.
    
    This is different from is_log_enabled - it checks if the effective log level
    allows messages at message_level to be output.
    
    Level hierarchy (by importance):
    - minimal: Most important (always output)
    - timing: Medium importance (output when effective level is timing or debug)
    - debug: Least important (only output when effective level is debug)
    """
    effective_level = get_log_level(module_name)
    
    # minimal messages are always output
    if message_level == LOG_LEVEL_MINIMAL:
        return True
    
    # timing messages are output when effective level is timing or debug
    if message_level == LOG_LEVEL_TIMING:
        return effective_level in (LOG_LEVEL_TIMING, LOG_LEVEL_DEBUG)
    
    # debug messages are only output when effective level is debug
    if message_level == LOG_LEVEL_DEBUG:
        return effective_level == LOG_LEVEL_DEBUG
    
    return False


def log_error_safely(
    module_name: str,
    error: Exception,
    context: str = "",
    level: str = LOG_LEVEL_MINIMAL
) -> None:
    """
    Log an error safely without raising exceptions.
    
    Args:
        module_name: Name of the module for log prefix
        error: The exception that occurred
        context: Optional context message describing what was being attempted
        level: Log level (default: minimal for errors)
    """
    if not _should_log(level, module_name):
        return
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        error_type = type(error).__name__
        error_msg = str(error)
        
        if context:
            print(f"[{timestamp}][{module_name}] ERROR: {context} - {error_type}: {error_msg}")
        else:
            print(f"[{timestamp}][{module_name}] ERROR: {error_type}: {error_msg}")
    except Exception:
        # Never fail during logging
        pass


def log_with_exception_info(
    module_name: str,
    context: str,
    exc_info: Any = None,
    level: str = LOG_LEVEL_DEBUG
) -> None:
    """
    Log with full exception information (type, message, traceback).
    
    Args:
        module_name: Name of the module for log prefix
        context: Context message describing what was being attempted
        exc_info: Exception instance or sys.exc_info() tuple
        level: Log level (default: debug for verbose output)
    """
    if not _should_log(level, module_name):
        return
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if exc_info is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        elif isinstance(exc_info, tuple) and len(exc_info) == 3:
            exc_type, exc_value, exc_tb = exc_info
        elif isinstance(exc_info, Exception):
            exc_type, exc_value, exc_tb = type(exc_info), exc_info, None
        else:
            exc_type, exc_value, exc_tb = type(exc_info), exc_info, None
        
        if exc_value:
            print(f"[{timestamp}][{module_name}] {context}: {exc_type.__name__}: {exc_value}")
            if level == LOG_LEVEL_DEBUG and exc_tb:
                import traceback
                tb_lines = traceback.format_tb(exc_tb)
                for tb_line in tb_lines[-3:]:  # Last 3 frames only
                    print(f"  {tb_line.strip()}")
    except Exception:
        # Never fail during logging
        pass


def create_fallback_logger(module_name: str, fallback_level: str = LOG_LEVEL_MINIMAL) -> Callable[[Exception, str], None]:
    """
    Create a fallback logger for use in except blocks.
    
    This is designed for the pattern:
        try:
            # some operation
        except Exception as e:
            log_error(e, "operation description")
    
    Args:
        module_name: Name of the module for log prefix
        fallback_level: Log level for fallback messages
    
    Returns:
        A callable that accepts (error: Exception, context: str) -> None
    """
    def log_fallback(error: Exception, context: str = "") -> None:
        log_error_safely(module_name, error, context, fallback_level)
    
    return log_fallback


def init_logging_from_config(log_level: Optional[str] = None) -> None:
    """
    Initialize logging based on configuration.
    
    Args:
        log_level: Log level string (minimal/timing/debug)
    """
    if log_level:
        set_global_log_level(log_level)


# Convenience function for the common pattern of logging in except blocks
def handle_exception_safely(
    module_name: str,
    error: Exception,
    operation: str,
    should_raise: bool = False,
    log_level: str = LOG_LEVEL_MINIMAL
) -> Optional[Exception]:
    """
    Handle an exception safely with logging.
    
    Args:
        module_name: Name of the module for log prefix
        error: The exception that occurred
        operation: Description of what was being attempted
        should_raise: If True, re-raise the exception after logging
        log_level: Log level for the error message
    
    Returns:
        The exception if should_raise is False, None if should_raise is True
        (return value is for convenience in except blocks)
    """
    log_error_safely(module_name, error, f"Failed to {operation}", log_level)
    if should_raise:
        raise error
    return error
