from __future__ import annotations

from core.kv_state import build_kv_state_signature, try_restore_kv_state, try_save_kv_state


class DummyLLM:
    def __init__(self, *, fail_on_load: bool = False) -> None:
        self.fail_on_load = fail_on_load
        self.loaded_payload = None

    def load_state(self, payload):
        if self.fail_on_load:
            raise RuntimeError("Failed to set llama state data")
        self.loaded_payload = payload

    def save_state(self):
        return {"blob": "state"}


def test_build_kv_state_signature_is_stable_for_same_inputs() -> None:
    def _ctx_turns(history, max_turns=None):
        return history.get("turns", [])

    history = {"turns": [{"id": 1}], "summary": {"enabled": True, "text": "sum"}}
    sig1 = build_kv_state_signature(
        history=history,
        max_turns=3,
        summarize_old_history=True,
        system_prompt="sys",
        model_path="model.gguf",
        mmproj_path=None,
        n_ctx=4096,
        n_gpu_layers=0,
        get_context_turns=_ctx_turns,
    )
    sig2 = build_kv_state_signature(
        history=history,
        max_turns=3,
        summarize_old_history=True,
        system_prompt="sys",
        model_path="model.gguf",
        mmproj_path=None,
        n_ctx=4096,
        n_gpu_layers=0,
        get_context_turns=_ctx_turns,
    )
    assert sig1 == sig2


def test_try_restore_kv_state_skips_load_when_state_size_mismatch() -> None:
    mem = {"sid": {"signature": "sig", "state": {"blob": "saved"}}}
    llm = DummyLLM()
    cleared = []

    try_restore_kv_state(
        session_id="sid",
        signature="sig",
        llm=llm,
        mem_kv_state=mem,
        log_prefix="[test]",
        log_level="minimal",
        model_path="model.gguf",
        n_ctx=1024,
        n_gpu_layers=0,
        clear_kv_state_for_session=lambda sid: cleared.append(sid),
        is_state_data_mismatch_error=lambda _e: False,
        invalidate_cache=lambda _llm, _remove_disk: None,
        saved_llama_state_size=lambda _state: 10,
        current_llama_state_size=lambda _llm: 20,
        kv_state_debug_info=lambda _state: "dbg",
        include_error_in_invalidate_message=False,
    )

    assert cleared == ["sid"]
    assert llm.loaded_payload is None


def test_try_restore_kv_state_invalidates_on_load_mismatch_error() -> None:
    mem = {"sid": {"signature": "sig", "state": {"blob": "saved"}}}
    llm = DummyLLM(fail_on_load=True)
    cleared = []
    invalidated = []

    try_restore_kv_state(
        session_id="sid",
        signature="sig",
        llm=llm,
        mem_kv_state=mem,
        log_prefix="[test]",
        log_level="minimal",
        model_path="model.gguf",
        n_ctx=1024,
        n_gpu_layers=0,
        clear_kv_state_for_session=lambda sid: cleared.append(sid),
        is_state_data_mismatch_error=lambda e: "llama state" in str(e),
        invalidate_cache=lambda _llm, remove_disk_data: invalidated.append(remove_disk_data),
        saved_llama_state_size=lambda _state: None,
        current_llama_state_size=lambda _llm: None,
        kv_state_debug_info=lambda _state: "dbg",
        include_error_in_invalidate_message=False,
    )

    assert cleared == ["sid"]
    assert invalidated == [True]


def test_try_save_kv_state_persists_signature_and_state() -> None:
    mem = {}
    llm = DummyLLM()

    try_save_kv_state(
        session_id="sid",
        signature="sig2",
        llm=llm,
        mem_kv_state=mem,
        log_prefix="[test]",
        log_level="minimal",
        kv_state_debug_info=lambda _state: "dbg",
        log_saved_when_not_minimal=False,
        log_unsupported_when_not_minimal=False,
    )

    assert mem["sid"]["signature"] == "sig2"
    assert mem["sid"]["state"] == {"blob": "state"}
