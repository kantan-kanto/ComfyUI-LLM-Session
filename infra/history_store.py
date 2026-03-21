# Filesystem helpers for history/transcript paths and atomic JSON writes.
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


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
        except Exception:
            pass
    os.replace(tmp, path)
