# Service for turn history append, optional summarization, and history file persistence.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

try:
    from ..core.logging_utils import log_error_safely, LOG_LEVEL_MINIMAL
except Exception:
    from core.logging_utils import log_error_safely, LOG_LEVEL_MINIMAL


@dataclass(frozen=True)
class HistoryPersistenceResult:
    history: Dict[str, Any]
    persistence_succeeded: bool = True
    persistence_error: Optional[Exception] = None


class HistoryPersistenceService:
    def _dep(self, deps: Mapping[str, Any], key: str) -> Any:
        if key not in deps:
            raise KeyError(f"Missing dependency: {key}")
        return deps[key]

    def persist_history_and_summary(
        self,
        *,
        request: Any,
        deps: Mapping[str, Any],
        history: Dict[str, Any],
        assistant_text: str,
        generation_result: Any,
        llm: Any,
        hist_path: str,
        model_path: str,
        mmproj_path: Optional[str],
    ) -> HistoryPersistenceResult:
        next_turn_id = self._dep(deps, "next_turn_id")
        now_iso = self._dep(deps, "now_iso")

        turn_params: Dict[str, Any] = {
            "max_tokens_req": int(request.max_tokens),
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "repeat_penalty": float(request.repeat_penalty) if request.repeat_penalty is not None else None,
            "repeat_last_n": int(request.repeat_last_n) if request.repeat_last_n is not None else None,
            "dynamic_max_tokens": bool(request.dynamic_max_tokens),
            "max_tokens_used": int(generation_result.gen_tokens),
            "turns_limit_used": int(generation_result.turns_limit) if generation_result.turns_limit is not None else None,
        }
        if request.include_image_and_stream_in_turn_params:
            turn_params["image_used"] = request.image is not None
            turn_params["streamed"] = bool(request.stream_to_console)
        if isinstance(request.advanced_generation_kwargs, dict) and request.advanced_generation_kwargs:
            turn_params["advanced_generation_kwargs"] = dict(request.advanced_generation_kwargs)
        if isinstance(request.advanced_summary_generation_kwargs, dict) and request.advanced_summary_generation_kwargs:
            turn_params["advanced_summary_generation_kwargs"] = dict(request.advanced_summary_generation_kwargs)

        history.setdefault("turns", []).append(
            {
                "id": next_turn_id(history),
                "t": now_iso(),
                "user": {
                    "text": request.user_text or "",
                    "image_note": "",
                },
                "assistant": {
                    "text": assistant_text or "",
                },
                "params": turn_params,
            }
        )

        history.setdefault("meta", {})["updated_at"] = now_iso()
        history.setdefault("meta", {})["last_params"] = {
            "persistent_cache": (request.persistent_cache or "off"),
            "runtime_cache": (request.runtime_cache or "off"),
            "max_turns": int(request.max_turns) if request.max_turns is not None else None,
            "summarize_old_history": bool(request.summarize_old_history),
            "summary_chunk_turns": int(request.summary_chunk_turns),
            "max_tokens_summary": int(request.max_tokens_summary),
            "summary_max_chars": int(request.summary_max_chars),
            "saved_at": now_iso(),
        }
        history["system_prompt"] = request.system_prompt or history.get("system_prompt", "")

        maybe_summarize_history = self._dep(deps, "maybe_summarize_history")
        if request.summarize_old_history and request.max_turns is not None:
            try:
                history = maybe_summarize_history(
                    model=llm,
                    history=history,
                    max_turns=int(request.max_turns),
                    summarize_old_history=bool(request.summarize_old_history),
                    summary_chunk_turns=int(request.summary_chunk_turns),
                    temperature=0.2,
                    max_tokens_summary=int(request.max_tokens_summary),
                    summary_max_chars=int(request.summary_max_chars),
                    suppress_logs=(request.log_level != "debug"),
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    text_chat_builder_overrides=request.text_chat_builder_overrides,
                    advanced_generation_kwargs=request.advanced_summary_generation_kwargs,
                )
            except Exception as e:
                # P1: Log summarization failure for debugging
                log_error_safely("TurnExecutionService", e, "Failed to summarize history", LOG_LEVEL_MINIMAL)

        atomic_write_json = self._dep(deps, "atomic_write_json")
        persistence_succeeded = True
        persistence_error: Optional[Exception] = None
        try:
            atomic_write_json(hist_path, history)
        except Exception as e:
            # P0: Log history file save failure - this is critical as it can cause data loss
            log_error_safely("TurnExecutionService", e, f"Failed to save history file: {hist_path}", LOG_LEVEL_MINIMAL)
            persistence_succeeded = False
            persistence_error = e

        return HistoryPersistenceResult(
            history=history,
            persistence_succeeded=persistence_succeeded,
            persistence_error=persistence_error,
        )
