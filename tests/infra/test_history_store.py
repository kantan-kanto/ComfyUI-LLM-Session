from __future__ import annotations

import json
import os

from infra import history_store
from core.logging_utils import LOG_LEVEL_TIMING, set_global_log_level


def test_normalize_history_schema_assigns_missing_ids_and_bounds_summary() -> None:
    hist = {
        "schema_version": 1,
        "summary": {"enabled": False, "text": "sum", "covered_until_turn_id": 99},
        "turns": [
            {"id": 0, "user": {"text": "u1"}},
            {"assistant": {"text": "a2"}},
            {"id": 3, "user": {"text": "u3"}},
        ],
    }

    normalized = history_store.normalize_history_schema(hist)

    ids = [t["id"] for t in normalized["turns"]]
    assert ids == [1, 2, 3]
    assert normalized["summary"]["enabled"] is True
    assert normalized["summary"]["covered_until_turn_id"] == 3


def test_load_history_reads_backup_when_primary_invalid(tmp_path) -> None:
    default_dir = str(tmp_path)
    path = history_store.history_path("sid", None, default_dir)

    with open(path, "w", encoding="utf-8") as f:
        f.write("{ invalid json")

    bak_payload = {
        "schema_version": 1,
        "meta": {},
        "system_prompt": "old",
        "summary": {"enabled": False, "text": "", "updated_at": "", "covered_until_turn_id": 0},
        "turns": [{"id": 1, "user": {"text": "hi"}, "assistant": {"text": "ok"}}],
    }
    with open(path + ".bak", "w", encoding="utf-8") as f:
        json.dump(bak_payload, f)

    hist, loaded_path = history_store.load_history(
        session_id="sid",
        history_dir=None,
        system_prompt="",
        model_sig=None,
        reset_session=False,
        default_dir=default_dir,
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
    )

    assert loaded_path == path
    assert hist["turns"][0]["id"] == 1
    assert hist["system_prompt"] == "old"


def test_load_history_reads_backup_when_primary_missing(tmp_path) -> None:
    default_dir = str(tmp_path)
    path = history_store.history_path("sid", None, default_dir)

    bak_payload = {
        "schema_version": 1,
        "meta": {},
        "system_prompt": "from-bak",
        "summary": {"enabled": False, "text": "", "updated_at": "", "covered_until_turn_id": 0},
        "turns": [{"id": 1, "user": {"text": "hi"}, "assistant": {"text": "ok"}}],
    }
    with open(path + ".bak", "w", encoding="utf-8") as f:
        json.dump(bak_payload, f)

    hist, loaded_path = history_store.load_history(
        session_id="sid",
        history_dir=None,
        system_prompt="",
        model_sig=None,
        reset_session=False,
        default_dir=default_dir,
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
    )

    assert loaded_path == path
    assert hist["system_prompt"] == "from-bak"
    assert hist["turns"][0]["id"] == 1
    assert os.path.exists(path)


def test_load_history_restores_primary_file_from_backup(tmp_path) -> None:
    default_dir = str(tmp_path)
    path = history_store.history_path("sid", None, default_dir)

    with open(path, "w", encoding="utf-8") as f:
        f.write("{ invalid json")

    bak_payload = {
        "schema_version": 1,
        "meta": {},
        "system_prompt": "from-bak",
        "summary": {"enabled": False, "text": "", "updated_at": "", "covered_until_turn_id": 0},
        "turns": [{"id": 1, "user": {"text": "hi"}, "assistant": {"text": "ok"}}],
    }
    with open(path + ".bak", "w", encoding="utf-8") as f:
        json.dump(bak_payload, f)

    history_store.load_history(
        session_id="sid",
        history_dir=None,
        system_prompt="",
        model_sig=None,
        reset_session=False,
        default_dir=default_dir,
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
    )

    with open(path, "r", encoding="utf-8") as f:
        restored = json.load(f)

    assert restored["system_prompt"] == "from-bak"
    assert restored["turns"][0]["id"] == 1


def test_load_history_debug_level_emits_restore_debug_log(tmp_path, capsys) -> None:
    default_dir = str(tmp_path)
    path = history_store.history_path("sid", None, default_dir)

    with open(path, "w", encoding="utf-8") as f:
        f.write("{ invalid json")

    bak_payload = {
        "schema_version": 1,
        "meta": {},
        "system_prompt": "from-bak",
        "summary": {"enabled": False, "text": "", "updated_at": "", "covered_until_turn_id": 0},
        "turns": [{"id": 1, "user": {"text": "hi"}, "assistant": {"text": "ok"}}],
    }
    with open(path + ".bak", "w", encoding="utf-8") as f:
        json.dump(bak_payload, f)

    set_global_log_level(LOG_LEVEL_TIMING)
    history_store.load_history(
        session_id="sid",
        history_dir=None,
        system_prompt="",
        model_sig=None,
        log_level="debug",
        reset_session=False,
        default_dir=default_dir,
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
    )
    out = capsys.readouterr().out
    assert "Success restore history file" in out


def test_load_history_reset_session_creates_fresh_history(tmp_path) -> None:
    default_dir = str(tmp_path)

    hist, path = history_store.load_history(
        session_id="sid",
        history_dir=None,
        system_prompt="sys",
        model_sig={"model_file": "m.gguf"},
        reset_session=True,
        default_dir=default_dir,
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
    )

    assert hist["system_prompt"] == "sys"
    assert hist["meta"]["model_signature"]["model_file"] == "m.gguf"
    assert hist["turns"] == []

    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["system_prompt"] == "sys"
