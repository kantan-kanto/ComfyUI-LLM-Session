"""Tests for core.logging_utils module."""

import pytest
from io import StringIO
import sys

from core.logging_utils import (
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_MINIMAL,
    LOG_LEVEL_TIMING,
    create_fallback_logger,
    get_log_level,
    handle_exception_safely,
    init_logging_from_config,
    is_log_enabled,
    log_error_safely,
    log_with_exception_info,
    set_global_log_level,
    set_module_log_level,
    get_module_logger,
)


class TestLogLevels:
    """Tests for log level constants and configuration."""

    def test_log_level_constants(self):
        """Test that log level constants are defined correctly."""
        assert LOG_LEVEL_MINIMAL == "minimal"
        assert LOG_LEVEL_TIMING == "timing"
        assert LOG_LEVEL_DEBUG == "debug"

    def test_set_global_log_level(self):
        """Test setting global log level."""
        set_global_log_level(LOG_LEVEL_DEBUG)
        assert get_log_level() == LOG_LEVEL_DEBUG

        set_global_log_level(LOG_LEVEL_TIMING)
        assert get_log_level() == LOG_LEVEL_TIMING

        set_global_log_level(LOG_LEVEL_MINIMAL)
        assert get_log_level() == LOG_LEVEL_MINIMAL

    def test_set_module_log_level(self):
        """Test setting module-specific log level."""
        set_global_log_level(LOG_LEVEL_TIMING)
        set_module_log_level("TestModule", LOG_LEVEL_DEBUG)

        assert get_log_level("TestModule") == LOG_LEVEL_DEBUG
        assert get_log_level("OtherModule") == LOG_LEVEL_TIMING

    def test_module_level_overrides_global(self):
        """Test that module-specific level overrides global level."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        set_module_log_level("OverrideModule", LOG_LEVEL_DEBUG)

        assert get_log_level("OverrideModule") == LOG_LEVEL_DEBUG
        assert get_log_level("OtherModule") == LOG_LEVEL_MINIMAL


class TestIsLogEnabled:
    """Tests for is_log_enabled function."""

    def test_debug_enables_all_levels(self):
        """Test that debug level enables all log levels."""
        set_global_log_level(LOG_LEVEL_DEBUG)
        assert is_log_enabled(LOG_LEVEL_DEBUG)
        assert is_log_enabled(LOG_LEVEL_TIMING)
        assert is_log_enabled(LOG_LEVEL_MINIMAL)

    def test_timing_enables_timing_and_minimal(self):
        """Test that timing level enables timing and minimal levels."""
        set_global_log_level(LOG_LEVEL_TIMING)
        assert not is_log_enabled(LOG_LEVEL_DEBUG)
        assert is_log_enabled(LOG_LEVEL_TIMING)
        assert is_log_enabled(LOG_LEVEL_MINIMAL)

    def test_minimal_only_enables_minimal(self):
        """Test that minimal level only enables minimal level."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        assert not is_log_enabled(LOG_LEVEL_DEBUG)
        assert not is_log_enabled(LOG_LEVEL_TIMING)
        assert is_log_enabled(LOG_LEVEL_MINIMAL)

    def test_module_specific_level(self):
        """Test is_log_enabled with module-specific level."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        set_module_log_level("TestModule", LOG_LEVEL_DEBUG)

        assert is_log_enabled(LOG_LEVEL_DEBUG, "TestModule")
        assert not is_log_enabled(LOG_LEVEL_DEBUG, "OtherModule")


class TestGetModuleLogger:
    """Tests for get_module_logger function."""

    def test_logger_creation(self):
        """Test that logger is created correctly."""
        log = get_module_logger("TestModule")
        assert callable(log)

    def test_logger_respects_level(self, capsys):
        """Test that logger respects log level."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        log = get_module_logger("TestModule")

        # Should not log timing message at minimal level
        log("Timing message", LOG_LEVEL_TIMING)
        captured = capsys.readouterr()
        assert "Timing message" not in captured.out

        # Should log minimal message
        log("Minimal message", LOG_LEVEL_MINIMAL)
        captured = capsys.readouterr()
        assert "Minimal message" in captured.out
        assert "[TestModule]" in captured.out

    def test_logger_includes_timestamp_and_module(self, capsys):
        """Test that logger includes timestamp and module name."""
        set_global_log_level(LOG_LEVEL_DEBUG)
        log = get_module_logger("MyTestModule")

        log("Test message", LOG_LEVEL_DEBUG)
        captured = capsys.readouterr()

        assert "[MyTestModule]" in captured.out
        # Timestamp format: [HH:MM:SS]
        assert "[" in captured.out
        assert "]" in captured.out

    def test_logger_does_not_raise_on_failure(self, capsys):
        """Test that logger does not raise exceptions even on failure."""
        set_global_log_level(LOG_LEVEL_DEBUG)
        log = get_module_logger("TestModule")

        # Should not raise even if print fails internally
        log("Test message", LOG_LEVEL_DEBUG)
        # If we get here without exception, test passes


class TestLogErrorSafely:
    """Tests for log_error_safely function."""

    def test_log_error_basic(self, capsys):
        """Test basic error logging."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        error = ValueError("Test error message")

        log_error_safely("TestModule", error, "Test context")
        captured = capsys.readouterr()

        assert "TestModule" in captured.out
        assert "ValueError" in captured.out
        assert "Test error message" in captured.out
        assert "Test context" in captured.out

    def test_log_error_without_context(self, capsys):
        """Test error logging without context."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        error = RuntimeError("Error without context")

        log_error_safely("TestModule", error)
        captured = capsys.readouterr()

        assert "TestModule" in captured.out
        assert "RuntimeError" in captured.out
        assert "Error without context" in captured.out

    def test_log_error_respects_level(self, capsys):
        """Test that error logging respects log level."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        error = Exception("Should not be logged")

        # Should not log at timing level when global is minimal
        log_error_safely("TestModule", error, "Context", level=LOG_LEVEL_TIMING)
        captured = capsys.readouterr()
        assert "Should not be logged" not in captured.out

    def test_log_error_does_not_raise(self):
        """Test that log_error_safely does not raise exceptions."""
        set_global_log_level(LOG_LEVEL_DEBUG)
        error = Exception("Test error")

        # Should not raise even if internal logging fails
        log_error_safely("TestModule", error, "Context")


class TestLogWithExceptionInfo:
    """Tests for log_with_exception_info function."""

    def test_log_with_exc_info_tuple(self, capsys):
        """Test logging with exception info tuple."""
        set_global_log_level(LOG_LEVEL_DEBUG)

        try:
            raise ValueError("Test exception")
        except Exception:
            log_with_exception_info("TestModule", "Test context")
            captured = capsys.readouterr()

            assert "TestModule" in captured.out
            assert "ValueError" in captured.out
            assert "Test exception" in captured.out

    def test_log_with_exc_info_exception(self, capsys):
        """Test logging with exception instance."""
        set_global_log_level(LOG_LEVEL_DEBUG)
        error = TypeError("Type error test")

        log_with_exception_info("TestModule", "Test context", error)
        captured = capsys.readouterr()

        assert "TypeError" in captured.out
        assert "Type error test" in captured.out

    def test_log_with_exc_info_respects_level(self, capsys):
        """Test that exception info logging respects log level."""
        set_global_log_level(LOG_LEVEL_MINIMAL)

        try:
            raise RuntimeError("Should not be logged")
        except Exception:
            log_with_exception_info("TestModule", "Context", level=LOG_LEVEL_DEBUG)
            captured = capsys.readouterr()
            assert "Should not be logged" not in captured.out


class TestCreateFallbackLogger:
    """Tests for create_fallback_logger function."""

    def test_fallback_logger_creation(self):
        """Test fallback logger creation."""
        log_error = create_fallback_logger("TestModule")
        assert callable(log_error)

    def test_fallback_logger_usage(self, capsys):
        """Test fallback logger usage."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        log_error = create_fallback_logger("TestModule", LOG_LEVEL_MINIMAL)

        error = Exception("Fallback test")
        log_error(error, "Fallback context")
        captured = capsys.readouterr()

        assert "TestModule" in captured.out
        assert "Fallback test" in captured.out
        assert "Fallback context" in captured.out


class TestHandleExceptionSafely:
    """Tests for handle_exception_safely function."""

    def test_handle_exception_returns_error(self, capsys):
        """Test that handle_exception_safely returns the error."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        error = ValueError("Handle test")

        result = handle_exception_safely("TestModule", error, "test operation")
        captured = capsys.readouterr()

        assert result is error
        assert "TestModule" in captured.out

    def test_handle_exception_raises(self, capsys):
        """Test that handle_exception_safely raises when should_raise=True."""
        set_global_log_level(LOG_LEVEL_MINIMAL)
        error = ValueError("Handle raise test")

        with pytest.raises(ValueError) as exc_info:
            handle_exception_safely("TestModule", error, "test operation", should_raise=True)

        assert exc_info.value is error
        captured = capsys.readouterr()
        assert "TestModule" in captured.out


class TestInitLoggingFromConfig:
    """Tests for init_logging_from_config function."""

    def test_init_with_log_level(self):
        """Test initialization with log level."""
        init_logging_from_config(LOG_LEVEL_DEBUG)
        assert get_log_level() == LOG_LEVEL_DEBUG

        init_logging_from_config(LOG_LEVEL_TIMING)
        assert get_log_level() == LOG_LEVEL_TIMING

    def test_init_with_none(self):
        """Test initialization with None does not change level."""
        set_global_log_level(LOG_LEVEL_DEBUG)
        init_logging_from_config(None)
        assert get_log_level() == LOG_LEVEL_DEBUG
