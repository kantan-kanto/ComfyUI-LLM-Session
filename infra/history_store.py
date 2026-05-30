# Filesystem helpers for history/transcript paths and atomic JSON writes.
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    from ..core.logging_utils import (
        log_error_safely,
        get_module_logger,
        set_module_log_level,
        LOG_LEVEL_MINIMAL,
        LOG_LEVEL_TIMING,
        LOG_LEVEL_DEBUG,
    )
except Exception:
    from core.logging_utils import (
        log_error_safely,
        get_module_logger,
        set_module_log_level,
        LOG_LEVEL_MINIMAL,
        LOG_LEVEL_TIMING,
        LOG_LEVEL_DEBUG,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_session_name(session_id: str) -> str:
    session_id = (session_id or "default").strip()
    safe = "".join(c for c in session_id if c.isalnum() or c in ("-", "_", "."))
    return safe or "default"


def session_base_dir(history_dir: Optional[str], default_dir: str) -> str:
    base = (history_dir or "").strip() or default_dir
    ensure_dir(base)
    return base


def history_path(session_id: str, history_dir: Optional[str], default_dir: str) -> str:
    base = session_base_dir(history_dir, default_dir)
    return os.path.join(base, f"{safe_session_name(session_id)}.json")


def transcript_path(session_id: str, history_dir: Optional[str], default_dir: str) -> str:
    base = session_base_dir(history_dir, default_dir)
    return os.path.join(base, f"{safe_session_name(session_id)}.txt")


def append_transcript_lines(path: str, lines: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")


def session_cache_root(session_id: str, history_dir: Optional[str], default_dir: str) -> str:
    base = session_base_dir(history_dir, default_dir)
    cache_root = os.path.join(base, "cache", safe_session_name(session_id))
    ensure_dir(cache_root)
    return cache_root


def atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    bak = path + ".bak"
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        try:
            os.replace(path, bak)
        except Exception as e:
            # P0: Log backup creation failure - this can cause data corruption
            log_error_safely("HistoryStore", e, f"Failed to create backup file: {bak}", LOG_LEVEL_MINIMAL)
            # Continue anyway - the tmp file will replace the original, but backup won't be available
    os.replace(tmp, path)


def _atomic_restore_json(path: str) -> Optional[str]:
    ensure_dir(os.path.dirname(path))
    bak = path + ".bak"
    if os.path.exists(bak):
        try:
            os.replace(bak, path)
            return path
        except Exception as e:            
            log_error_safely("HistoryStore", e, f"Failed to restore from backup file: {bak}", LOG_LEVEL_MINIMAL)
    return None


def _safe_timestamp_for_filename(now_iso: Callable[[], str]) -> str:
    try:
        raw = str(now_iso() or "")
    except Exception:
        raw = ""
    safe = "".join(c if c.isalnum() else "-" for c in raw).strip("-")
    return safe or "unknown-time"


def _quarantine_corrupt_history(path: str, now_iso: Callable[[], str]) -> str:
    suffix = _safe_timestamp_for_filename(now_iso)
    quarantine_path = f"{path}.corrupt-{suffix}"
    if os.path.exists(quarantine_path):
        i = 1
        while os.path.exists(f"{quarantine_path}-{i}"):
            i += 1
        quarantine_path = f"{quarantine_path}-{i}"
    os.replace(path, quarantine_path)
    return quarantine_path


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


@dataclass(frozen=True)
class HistoryLoadAttempt:
    history: Optional[Dict[str, Any]]
    path: str
    error: Optional[Exception] = None


def normalize_history_schema(hist: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort upgrade of older history files to the current in-memory shape."""
    hist.setdefault("meta", {})
    hist.setdefault("system_prompt", "")
    hist.setdefault("turns", [])

    summary = hist.setdefault("summary", {})
    summary.setdefault("enabled", False)
    summary.setdefault("text", "")
    summary.setdefault("updated_at", "")

    turns = hist.get("turns") or []
    normalized_turns = []
    next_id = 1
    max_id = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        turn = dict(t)
        tid = _coerce_int(turn.get("id"), 0)
        if tid <= 0:
            tid = next_id
        next_id = max(next_id, tid + 1)
        max_id = max(max_id, tid)
        turn["id"] = tid
        normalized_turns.append(turn)

    hist["turns"] = normalized_turns

    covered = _coerce_int(summary.get("covered_until_turn_id"), -1)
    if covered < 0:
        # Old histories may already contain a rolling summary but lack the boundary marker.
        # Preserve the existing summary text and start tracking coverage from now on.
        covered = 0
    summary["covered_until_turn_id"] = min(max(0, covered), max_id)
    summary["enabled"] = bool(summary.get("enabled") or str(summary.get("text") or "").strip())
    return hist


def new_history(
    session_id: str,
    system_prompt: str,
    model_sig: Optional[Dict[str, Any]],
    now_iso: Callable[[], str],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "meta": {
            "session_id": session_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "model_signature": model_sig or {},
        },
        "system_prompt": system_prompt or "",
        "summary": {
            "enabled": False,
            "text": "",
            "updated_at": "",
            "covered_until_turn_id": 0,
        },
        "turns": [],
    }


def next_turn_id(history: Dict[str, Any]) -> int:
    turns = history.get("turns") or []
    max_id = 0
    for t in turns:
        if isinstance(t, dict):
            max_id = max(max_id, _coerce_int(t.get("id"), 0))
    return max_id + 1


def get_context_turns(history: Dict[str, Any], max_turns: Optional[int] = None) -> List[Dict[str, Any]]:
    turns = history.get("turns") or []
    covered = _coerce_int((history.get("summary") or {}).get("covered_until_turn_id"), 0)
    pending = [
        t for t in turns
        if isinstance(t, dict) and _coerce_int(t.get("id"), 0) > covered
    ]
    if max_turns is not None:
        mt = max(0, _coerce_int(max_turns, 0))
        if mt > 0:
            pending = pending[-mt:]
        else:
            pending = []
    return pending


def _try_load_history_file(
    path: str,
    system_prompt: str,
    model_sig: Optional[Dict[str, Any]],
) -> HistoryLoadAttempt:
    if not os.path.exists(path):
        return HistoryLoadAttempt(history=None, path=path)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            if not isinstance(hist, dict) or hist.get("schema_version") != 1:
                raise ValueError("Unsupported history schema")
            hist = normalize_history_schema(hist)
            if system_prompt:
                hist["system_prompt"] = system_prompt
            if model_sig:
                hist.setdefault("meta", {}).setdefault("model_signature", {}).update(model_sig)
            return HistoryLoadAttempt(history=hist, path=path)
        except Exception as e:
            return HistoryLoadAttempt(history=None, path=path, error=e)
        

def load_history(
    *,
    session_id: str,
    history_dir: Optional[str],
    system_prompt: str,
    model_sig: Optional[Dict[str, Any]] = None,
    log_level: Optional[str] = None,
    reset_session: bool = False,
    default_dir: str,
    now_iso: Callable[[], str],
) -> tuple[Dict[str, Any], str]:
    """Load session history JSON, or create new. Returns (history, path)."""
    if isinstance(log_level, str) and log_level in {LOG_LEVEL_MINIMAL, LOG_LEVEL_TIMING, LOG_LEVEL_DEBUG}:
        # Keep infra logger filtering aligned with UI-selected verbosity.
        set_module_log_level("Load History", log_level)
        set_module_log_level("HistoryStore", log_level)

    log = get_module_logger("Load History")
    log(f"load_history (reset session: {reset_session})", LOG_LEVEL_DEBUG)
    path = history_path(session_id, history_dir, default_dir)

    if reset_session:
        hist = new_history(session_id, system_prompt, model_sig=model_sig, now_iso=now_iso)
        atomic_write_json(path, hist)
        return hist, path
    
    primary_attempt = _try_load_history_file(path, system_prompt, model_sig)
    primary_exists = os.path.exists(path)
    if primary_attempt.history is not None:
        log(f"Success load history: {path}", LOG_LEVEL_TIMING)
        return primary_attempt.history, path
    if primary_attempt.error is not None:
        log_error_safely(
            "HistoryStore",
            primary_attempt.error,
            f"Failed to load history file: {path}",
            LOG_LEVEL_MINIMAL,
        )

    bak = path + ".bak"
    backup_attempt = _try_load_history_file(bak, system_prompt, model_sig)
    if backup_attempt.history is not None:
        log(f"Success load backup history: {bak}", LOG_LEVEL_TIMING)
        try:
            _atomic_restore_json(path)
            atomic_write_json(path, backup_attempt.history)
            log(f"Success restore history file: {path}", LOG_LEVEL_DEBUG)
        except Exception as e:
            # P0: Log history file save failure - this is critical as it can cause data loss
            log_error_safely("HistoryStore", e, f"Failed to restore history file: {path}", LOG_LEVEL_DEBUG)
        return backup_attempt.history, path
    else:
        if backup_attempt.error is not None:
            log_error_safely(
                "HistoryStore",
                backup_attempt.error,
                f"Failed to load backup history file: {bak}",
                LOG_LEVEL_MINIMAL,
            )
        log(f"Failed to load backup history: {bak}", LOG_LEVEL_TIMING)

    if primary_exists:
        try:
            quarantined = _quarantine_corrupt_history(path, now_iso)
            log(f"Quarantined unreadable history file: {quarantined}", LOG_LEVEL_TIMING)
        except Exception as e:
            log_error_safely(
                "HistoryStore",
                e,
                f"Failed to quarantine unreadable history file; refusing to overwrite: {path}",
                LOG_LEVEL_MINIMAL,
            )
            raise
    
    hist = new_history(session_id, system_prompt, model_sig=model_sig, now_iso=now_iso)
    atomic_write_json(path, hist)
    return hist, path
