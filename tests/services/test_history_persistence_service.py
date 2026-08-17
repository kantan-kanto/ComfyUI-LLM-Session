from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.history_persistence_service import HistoryPersistenceService


class FakeImageBatch:
    shape = (2, 4, 4, 3)


def _request(**overrides):
    values = {
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "dynamic_max_tokens": True,
        "include_media_and_stream_in_turn_params": True,
        "media": None,
        "stream_to_console": False,
        "advanced_generation_kwargs": {},
        "advanced_summary_generation_kwargs": {},
        "user_text": "hello",
        "persistent_cache": "off",
        "runtime_cache": "off",
        "max_turns": 12,
        "summarize_old_history": False,
        "summary_chunk_turns": 3,
        "max_tokens_summary": 256,
        "summary_max_chars": 1200,
        "system_prompt": "sys",
        "log_level": "timing",
        "text_chat_builder_overrides": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _generation_result(**overrides):
    values = {"gen_tokens": 11, "turns_limit": 4}
    values.update(overrides)
    return SimpleNamespace(**values)


def _deps(*, writes=None, summarize_calls=None):
    writes = writes if writes is not None else []
    summarize_calls = summarize_calls if summarize_calls is not None else []

    def _maybe_summarize_history(**kwargs):
        summarize_calls.append(kwargs)
        history = dict(kwargs["history"])
        history["summary"] = {"enabled": True, "text": "summary"}
        return history

    return {
        "next_turn_id": lambda history: len(history.get("turns", [])) + 1,
        "now_iso": lambda: "2026-07-05T12:00:00+09:00",
        "maybe_summarize_history": _maybe_summarize_history,
        "atomic_write_json": lambda path, history: writes.append((path, history)),
    }


def test_persist_history_appends_turn_and_writes_file() -> None:
    writes = []
    service = HistoryPersistenceService()

    result = service.persist_history_and_summary(
        request=_request(
            media=FakeImageBatch(),
            stream_to_console=True,
            advanced_generation_kwargs={"seed": 123},
            official_sampling_profile="qwen3.8-thinking",
        ),
        deps=_deps(writes=writes),
        history={"turns": []},
        assistant_text="answer",
        generation_result=_generation_result(),
        llm=object(),
        hist_path="history.json",
        model_path="model.gguf",
        mmproj_path=None,
    )

    assert result.persistence_succeeded is True
    assert result.persistence_error is None
    assert writes == [("history.json", result.history)]
    turn = result.history["turns"][0]
    assert turn["id"] == 1
    assert turn["user"]["text"] == "hello"
    assert turn["assistant"]["text"] == "answer"
    assert turn["params"]["image_used"] is True
    assert turn["params"]["image_count"] == 2
    assert turn["params"]["streamed"] is True
    assert turn["params"]["advanced_generation_kwargs"] == {"seed": 123}
    assert turn["params"]["official_sampling_profile"] == "qwen3.8-thinking"
    assert result.history["meta"]["last_params"]["runtime_cache"] == "off"
    assert result.history["system_prompt"] == "sys"


def test_persist_history_summarizes_before_write_when_enabled() -> None:
    writes = []
    summarize_calls = []
    service = HistoryPersistenceService()

    result = service.persist_history_and_summary(
        request=_request(
            summarize_old_history=True,
            max_turns=2,
            log_level="debug",
            advanced_summary_generation_kwargs={"seed": 456},
        ),
        deps=_deps(writes=writes, summarize_calls=summarize_calls),
        history={"turns": []},
        assistant_text="answer",
        generation_result=_generation_result(turns_limit=None),
        llm="llm",
        hist_path="history.json",
        model_path="model.gguf",
        mmproj_path="vision.mmproj",
    )

    assert result.persistence_succeeded is True
    assert result.history["summary"] == {"enabled": True, "text": "summary"}
    assert writes == [("history.json", result.history)]
    assert summarize_calls[0]["model"] == "llm"
    assert summarize_calls[0]["max_turns"] == 2
    assert summarize_calls[0]["suppress_logs"] is False
    assert summarize_calls[0]["mmproj_path"] == "vision.mmproj"
    assert summarize_calls[0]["advanced_generation_kwargs"] == {"seed": 456}


def test_persist_history_reports_write_failure_without_losing_history() -> None:
    service = HistoryPersistenceService()
    failure = OSError("locked")

    def _raise_write(_path, _history):
        raise failure

    deps = _deps()
    deps["atomic_write_json"] = _raise_write

    result = service.persist_history_and_summary(
        request=_request(),
        deps=deps,
        history={"turns": []},
        assistant_text="answer",
        generation_result=_generation_result(),
        llm=object(),
        hist_path="history.json",
        model_path="model.gguf",
        mmproj_path=None,
    )

    assert result.persistence_succeeded is False
    assert result.persistence_error is failure
    assert result.history["turns"][0]["assistant"]["text"] == "answer"


@pytest.mark.parametrize(
    ("media", "expected"),
    [
        (None, {"image_used": False, "image_count": 0, "audio_used": False}),
        (FakeImageBatch(), {"image_used": True, "image_count": 2, "audio_used": False}),
        (
            {"waveform": object(), "sample_rate": 16000},
            {"image_used": False, "image_count": 0, "audio_used": True},
        ),
        (object(), {"image_used": False, "image_count": 0, "audio_used": False}),
    ],
)
def test_persist_history_records_media_description(media, expected) -> None:
    service = HistoryPersistenceService()

    result = service.persist_history_and_summary(
        request=_request(media=media),
        deps=_deps(),
        history={"turns": []},
        assistant_text="answer",
        generation_result=_generation_result(),
        llm=object(),
        hist_path="history.json",
        model_path="model.gguf",
        mmproj_path=None,
    )

    params = result.history["turns"][0]["params"]
    for key, value in expected.items():
        if isinstance(value, bool):
            assert params[key] is value
        else:
            assert params[key] == value


def test_persist_history_requires_dependencies() -> None:
    service = HistoryPersistenceService()

    with pytest.raises(KeyError, match="next_turn_id"):
        service.persist_history_and_summary(
            request=_request(),
            deps={},
            history={"turns": []},
            assistant_text="answer",
            generation_result=_generation_result(),
            llm=object(),
            hist_path="history.json",
            model_path="model.gguf",
            mmproj_path=None,
        )
