from __future__ import annotations

from types import SimpleNamespace

from services.kv_state_service import KvStateService


class DummyManager:
    def __init__(self, *, fail_invalidate: bool = False) -> None:
        self.fail_invalidate = fail_invalidate
        self.invalidate_calls = []

    def invalidate_cache(self, llm, *, remove_disk_data: bool):
        self.invalidate_calls.append((llm, remove_disk_data))
        if self.fail_invalidate:
            raise OSError("cannot invalidate")


def _request(**overrides):
    values = {
        "runtime_cache": "KV_cache",
        "media": None,
        "max_turns": 4,
        "summarize_old_history": True,
        "system_prompt": "sys",
        "n_ctx": 4096,
        "n_gpu_layers": 12,
        "tensor_split": "1,1",
        "session_id": "sid",
        "log_prefix": "[Test]",
        "log_level": "timing",
        "include_error_in_invalidate_message": True,
        "kv_log_saved_when_not_minimal": True,
        "kv_log_unsupported_when_not_minimal": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _deps(*, restore_calls=None, save_calls=None):
    restore_calls = restore_calls if restore_calls is not None else []
    save_calls = save_calls if save_calls is not None else []

    def _build_kv_state_signature(**kwargs):
        return "sig:" + kwargs["model_path"]

    return {
        "build_kv_state_signature": _build_kv_state_signature,
        "try_restore_kv_state": lambda **kwargs: restore_calls.append(kwargs),
        "try_save_kv_state": lambda **kwargs: save_calls.append(kwargs),
        "is_state_data_mismatch_error": lambda err: "state" in str(err),
        "saved_llama_state_size": lambda _state: 10,
        "current_llama_state_size": lambda _llm: 10,
        "kv_state_debug_info": lambda _state: "debug",
        "get_context_turns": lambda history, max_turns=None: history.get("turns", []),
        "mem_kv_state": {},
        "cache_debug_label": lambda _mgr: "cache-label",
    }


def test_restore_state_calls_try_restore_for_kv_cache() -> None:
    restore_calls = []
    deps = _deps(restore_calls=restore_calls)
    mgr = DummyManager()
    llm = object()
    cleared = []

    callbacks = KvStateService().restore_state(
        request=_request(),
        deps=deps,
        mgr=mgr,
        llm=llm,
        history={"turns": [{"id": 1}]},
        model_path="model.gguf",
        mmproj_path=None,
        clear_kv_state_for_session=cleared.append,
    )

    assert callbacks == (
        deps["is_state_data_mismatch_error"],
        deps["kv_state_debug_info"],
        deps["get_context_turns"],
    )
    assert len(restore_calls) == 1
    call = restore_calls[0]
    assert call["session_id"] == "sid"
    assert call["signature"] == "sig:model.gguf"
    assert call["llm"] is llm
    assert call["mem_kv_state"] is deps["mem_kv_state"]
    call["invalidate_cache"](llm, remove_disk_data=True)
    assert mgr.invalidate_calls == [(llm, True)]


def test_restore_state_skips_try_restore_when_cache_off_or_media_present() -> None:
    for request in [_request(runtime_cache="off"), _request(media=object())]:
        restore_calls = []

        KvStateService().restore_state(
            request=request,
            deps=_deps(restore_calls=restore_calls),
            mgr=DummyManager(),
            llm=object(),
            history={"turns": []},
            model_path="model.gguf",
            mmproj_path=None,
            clear_kv_state_for_session=lambda _sid: None,
        )

        assert restore_calls == []


def test_restore_state_swallows_restore_failure() -> None:
    def _raise_restore(**_kwargs):
        raise RuntimeError("restore failed")

    deps = _deps()
    deps["try_restore_kv_state"] = _raise_restore

    callbacks = KvStateService().restore_state(
        request=_request(),
        deps=deps,
        mgr=DummyManager(),
        llm=object(),
        history={"turns": []},
        model_path="model.gguf",
        mmproj_path=None,
        clear_kv_state_for_session=lambda _sid: None,
    )

    assert callbacks[0] is deps["is_state_data_mismatch_error"]


def test_save_state_calls_try_save_for_kv_cache() -> None:
    save_calls = []
    deps = _deps(save_calls=save_calls)
    llm = object()

    KvStateService().save_state(
        request=_request(),
        deps=deps,
        llm=llm,
        history={"turns": [{"id": 1}]},
        model_path="model.gguf",
        mmproj_path="vision.mmproj",
        kv_state_debug_info=deps["kv_state_debug_info"],
        get_context_turns=deps["get_context_turns"],
    )

    assert len(save_calls) == 1
    call = save_calls[0]
    assert call["session_id"] == "sid"
    assert call["signature"] == "sig:model.gguf"
    assert call["llm"] is llm
    assert call["log_saved_when_not_minimal"] is True
    assert call["log_unsupported_when_not_minimal"] is False


def test_save_state_skips_when_cache_off_or_media_present() -> None:
    for request in [_request(runtime_cache="off"), _request(media=object())]:
        save_calls = []

        KvStateService().save_state(
            request=request,
            deps=_deps(save_calls=save_calls),
            llm=object(),
            history={"turns": []},
            model_path="model.gguf",
            mmproj_path=None,
            kv_state_debug_info=lambda _state: "debug",
            get_context_turns=lambda history, max_turns=None: history.get("turns", []),
        )

        assert save_calls == []


def test_save_state_swallows_save_failure() -> None:
    def _raise_save(**_kwargs):
        raise RuntimeError("save failed")

    deps = _deps()
    deps["try_save_kv_state"] = _raise_save

    KvStateService().save_state(
        request=_request(),
        deps=deps,
        llm=object(),
        history={"turns": []},
        model_path="model.gguf",
        mmproj_path=None,
        kv_state_debug_info=lambda _state: "debug",
        get_context_turns=lambda history, max_turns=None: history.get("turns", []),
    )


def test_on_state_cache_mismatch_clears_and_invalidates_cache(capsys) -> None:
    cleared = []
    mgr = DummyManager()
    llm = object()

    KvStateService().on_state_cache_mismatch(
        request=_request(),
        deps=_deps(),
        clear_kv_state_for_session=cleared.append,
        mgr=mgr,
        llm=llm,
        err=RuntimeError("state mismatch"),
    )

    output = capsys.readouterr().out
    assert cleared == ["sid"]
    assert mgr.invalidate_calls == [(llm, True)]
    assert "cache-label" in output
    assert "cache invalidated" in output


def test_on_state_cache_mismatch_swallows_invalidation_failure() -> None:
    cleared = []

    KvStateService().on_state_cache_mismatch(
        request=_request(log_level="minimal"),
        deps=_deps(),
        clear_kv_state_for_session=cleared.append,
        mgr=DummyManager(fail_invalidate=True),
        llm=object(),
        err=RuntimeError("state mismatch"),
    )

    assert cleared == ["sid"]
