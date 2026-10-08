from __future__ import annotations

from dataclasses import replace

from core.turn_types import GenerationRunResult
from services.turn_execution_service import TurnExecutionService
from turn_execution_helpers import (
    DummyManager,
    _base_deps,
    _capture_generation_kwargs,
    _make_request,
)


def _run_turn(advanced_backend_kwargs):
    mgr = DummyManager()
    history = {"turns": [], "summary": {"enabled": False, "text": ""}, "meta": {}}
    deps, _writes = _base_deps(
        history,
        run_generation_result=GenerationRunResult(
            assistant_text="assistant reply",
            gen_tokens=64,
            turns_limit=12,
            last_err=None,
            succeeded=True,
            non_ctx_error=False,
        ),
    )
    observed_generation = _capture_generation_kwargs(deps)
    request = replace(_make_request(deps, mgr), advanced_backend_kwargs=advanced_backend_kwargs)

    result = TurnExecutionService().execute_turn(request)

    assert result.generation_succeeded is True
    return mgr, history, observed_generation


def test_execute_turn_passes_backend_kwargs_to_model_load_only() -> None:
    mgr, history, observed_generation = _run_turn({"n_batch": 2048, "n_ubatch": 1024})

    assert mgr.last_load_kwargs["advanced_backend_kwargs"] == {"n_batch": 2048, "n_ubatch": 1024}
    assert "n_ubatch" not in str(observed_generation)
    assert history["turns"][0]["params"]["advanced_backend_kwargs"] == {"n_batch": 2048, "n_ubatch": 1024}


def test_execute_turn_does_not_record_unset_backend_kwargs() -> None:
    mgr, history, _observed_generation = _run_turn(None)

    assert mgr.last_load_kwargs["advanced_backend_kwargs"] is None
    assert "advanced_backend_kwargs" not in history["turns"][0]["params"]
