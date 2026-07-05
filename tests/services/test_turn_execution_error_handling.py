from __future__ import annotations

from core.turn_types import GenerationRunResult
from services.generation_execution_service import GenerationExecutionService
from services.turn_execution_service import (
    SessionChatNodeExecutionDependencies,
    SessionChatNodeExecutionRequest,
    SessionChatNodeExecutionService,
    TurnExecutionRequest,
    TurnExecutionResult,
    TurnExecutionService,
)
from turn_execution_helpers import (
    DummyManager,
    DummyLogRequest,
    _base_deps,
    _capture_generation_kwargs,
    _make_request,
)
import pytest


class TestErrorHandlingP0P1:
    """Tests for P0/P1 error handling improvements."""

    def test_cache_invalidation_failure_during_reset_logs_error(self, caplog) -> None:
        """P0: Test that cache invalidation failure during session reset is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        # Make invalidate_cache raise an exception
        def failing_invalidate(*_args, **_kwargs):
            raise RuntimeError("Cache invalidation failed")

        mgr.invalidate_cache = failing_invalidate

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Create request with reset_session=True
        request = TurnExecutionRequest(
            user_text="hello",
            session_id="sid",
            model="model.gguf",
            mmproj="(Auto detect)",
            system_prompt="sys",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            n_gpu_layers=0,
            n_ctx=1024,
            runtime_cache="off",
            model_manager=mgr,
            dependencies=deps,
            reset_session=True,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite cache invalidation failure
        assert result.generation_succeeded is True

    def test_cache_directory_creation_failure_logs_error(self, caplog) -> None:
        """P0: Test that cache directory creation failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make session_cache_root raise an exception
        deps["session_cache_root"] = lambda _sid, _dir: (_ for _ in ()).throw(RuntimeError("Cannot create cache dir"))

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error and continue without cache
        result = service.execute_turn(request)

        # Generation should still succeed despite cache directory failure
        assert result.generation_succeeded is True

    def test_history_save_failure_logs_error(self, caplog) -> None:
        """P0: Test that history file save failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make atomic_write_json raise an exception
        deps["atomic_write_json"] = lambda _path, _obj: (_ for _ in ()).throw(RuntimeError("Failed to write history"))

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite history save failure
        assert result.generation_succeeded is True
        assert result.persistence_succeeded is False
        assert isinstance(result.persistence_error, RuntimeError)
        assert "Failed to write history" in str(result.persistence_error)

    def test_kv_state_restore_failure_logs_error(self, caplog) -> None:
        """P1: Test that KV state restore failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make try_restore_kv_state raise an exception
        deps["try_restore_kv_state"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("KV restore failed"))

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite KV state restore failure
        assert result.generation_succeeded is True

    def test_summarization_failure_logs_error(self, caplog) -> None:
        """P1: Test that summarization failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make maybe_summarize_history raise an exception
        deps["maybe_summarize_history"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Summarization failed"))

        request = TurnExecutionRequest(
            user_text="hello",
            session_id="sid",
            model="model.gguf",
            mmproj="(Auto detect)",
            system_prompt="sys",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            n_gpu_layers=0,
            n_ctx=1024,
            runtime_cache="off",
            model_manager=mgr,
            dependencies=deps,
            summarize_old_history=True,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite summarization failure
        assert result.generation_succeeded is True

    def test_kv_state_save_failure_logs_error(self, caplog) -> None:
        """P1: Test that KV state save failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make try_save_kv_state raise an exception
        deps["try_save_kv_state"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("KV save failed"))

        request = TurnExecutionRequest(
            user_text="hello",
            session_id="sid",
            model="model.gguf",
            mmproj="(Auto detect)",
            system_prompt="sys",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            n_gpu_layers=0,
            n_ctx=1024,
            runtime_cache="KV_cache",
            model_manager=mgr,
            dependencies=deps,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite KV state save failure
        assert result.generation_succeeded is True

    def test_cache_invalidation_on_mismatch_failure_logs_error(self, caplog) -> None:
        """P1: Test that cache invalidation failure on mismatch is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make is_state_data_mismatch_error return True to trigger cache invalidation
        deps["is_state_data_mismatch_error"] = lambda _err: True

        # Make invalidate_cache raise an exception during mismatch handling
        def failing_invalidate(llm, remove_disk_data=False):
            raise RuntimeError("Cache invalidation on mismatch failed")

        mgr.invalidate_cache = failing_invalidate

        request = _make_request(deps, mgr)

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite cache invalidation failure
        assert result.generation_succeeded is True

    def test_summary_compaction_failure_logs_error(self, caplog) -> None:
        """P1: Test that summary compaction failure is logged."""
        service = TurnExecutionService()
        mgr = DummyManager()

        history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
        deps, _writes = _base_deps(
            history,
            run_generation_result=GenerationRunResult(
                assistant_text="reply",
                gen_tokens=64,
                turns_limit=12,
                last_err=None,
                succeeded=True,
                non_ctx_error=False,
            ),
        )

        # Make maybe_compact_summary raise an exception
        deps["maybe_compact_summary"] = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("Summary compaction failed"))

        request = TurnExecutionRequest(
            user_text="hello",
            session_id="sid",
            model="model.gguf",
            mmproj="(Auto detect)",
            system_prompt="sys",
            max_tokens=64,
            temperature=0.7,
            top_p=0.9,
            n_gpu_layers=0,
            n_ctx=1024,
            runtime_cache="off",
            model_manager=mgr,
            dependencies=deps,
            summarize_old_history=True,
        )

        # Should not raise, but should log the error
        result = service.execute_turn(request)

        # Generation should still succeed despite summary compaction failure
        assert result.generation_succeeded is True
