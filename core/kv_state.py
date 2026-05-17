# KV state signature/restore/save helpers shared by chat and cycle flows.
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional

from .logging_utils import log_error_safely, LOG_LEVEL_MINIMAL


def build_kv_state_signature(
    *,
    history: Dict[str, Any],
    max_turns: Optional[int],
    summarize_old_history: bool,
    system_prompt: str,
    model_path: str,
    mmproj_path: Optional[str],
    n_ctx: int,
    n_gpu_layers: int,
    tensor_split: Optional[list[float]] = None,
    get_context_turns: Callable[..., Any],
) -> str:
    turns_ctx = get_context_turns(history, max_turns=max_turns)
    summary_txt = ""
    if bool(summarize_old_history) and history.get("summary", {}).get("enabled", False):
        summary_txt = history.get("summary", {}).get("text", "") or ""
    effective_system = (system_prompt or history.get("system_prompt", "") or "").strip()
    kv_prefix_material = json.dumps(
        {
            "model_path": os.path.abspath(model_path) if model_path else "",
            "mmproj_path": os.path.abspath(mmproj_path) if mmproj_path else "",
            "n_ctx": int(n_ctx) if n_ctx is not None else None,
            "n_gpu_layers": int(n_gpu_layers) if n_gpu_layers is not None else None,
            "tensor_split": [float(x) for x in tensor_split] if tensor_split is not None else None,
            "system": effective_system,
            "summary": summary_txt,
            "turns": turns_ctx,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(kv_prefix_material.encode("utf-8")).hexdigest()


def try_restore_kv_state(
    *,
    session_id: str,
    signature: str,
    llm: Any,
    mem_kv_state: Dict[str, Any],
    log_prefix: str,
    log_level: str,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    clear_kv_state_for_session: Callable[[str], None],
    is_state_data_mismatch_error: Callable[[Exception], bool],
    invalidate_cache: Callable[[Any, bool], None],
    saved_llama_state_size: Callable[[Any], Optional[int]],
    current_llama_state_size: Callable[[Any], Optional[int]],
    kv_state_debug_info: Callable[[Any], str],
    include_error_in_invalidate_message: bool,
) -> None:
    entry = mem_kv_state.get(session_id)
    if entry and entry.get("signature") == signature and hasattr(llm, "load_state"):
        try:
            state_payload = entry.get("state")
            saved_size = saved_llama_state_size(state_payload)
            current_size = current_llama_state_size(llm)
            skip_load = (
                saved_size is not None
                and current_size is not None
                and saved_size != current_size
            )
            if skip_load:
                clear_kv_state_for_session(session_id)
                if log_level != "minimal":
                    print(f"{log_prefix} KV state: INVALIDATED (incompatible state size)")
                if log_level == "debug":
                    print(
                        f"{log_prefix} KV precheck mismatch: "
                        f"session_id={session_id}, sig={signature[:12]}, "
                        f"saved_size={saved_size}, current_size={current_size}, "
                        f"model={os.path.basename(model_path)}"
                    )
            else:
                if log_level == "debug":
                    print(
                        f"{log_prefix} KV load attempt: "
                        f"session_id={session_id}, sig={signature[:12]}, "
                        f"{kv_state_debug_info(state_payload)}, "
                        f"model={os.path.basename(model_path)}, n_ctx={int(n_ctx)}, n_gpu_layers={int(n_gpu_layers)}"
                    )
                llm.load_state(state_payload)
                if log_level != "minimal":
                    print(f"{log_prefix} KV state: HIT (memory)")
        except Exception as e:
            clear_kv_state_for_session(session_id)
            if is_state_data_mismatch_error(e):
                try:
                    invalidate_cache(llm, True)
                except Exception as invalidate_err:
                    # P1: Log cache invalidation failure on state mismatch
                    log_error_safely("KVState", invalidate_err, "Failed to invalidate cache on state mismatch", LOG_LEVEL_MINIMAL)
            if log_level != "minimal":
                if include_error_in_invalidate_message:
                    print(f"{log_prefix} KV state: INVALIDATED (load failed: {e})")
                else:
                    print(f"{log_prefix} KV state: INVALIDATED (load failed)")
            if log_level == "debug":
                print(
                    f"{log_prefix} KV load context: "
                    f"session_id={session_id}, sig={signature[:12]}, "
                    f"model={os.path.basename(model_path)}, n_ctx={int(n_ctx)}, n_gpu_layers={int(n_gpu_layers)}"
                )
                if not include_error_in_invalidate_message:
                    print(f"{log_prefix} KV state load failed: {e}")
    else:
        if log_level != "minimal":
            reason = "no state" if not entry else "prefix changed"
            print(f"{log_prefix} KV state: MISS ({reason})")
        if log_level == "debug" and entry:
            print(
                f"{log_prefix} KV miss details: "
                f"session_id={session_id}, stored_sig={str(entry.get('signature', ''))[:12]}, "
                f"current_sig={signature[:12]}, {kv_state_debug_info(entry.get('state'))}"
            )


def try_save_kv_state(
    *,
    session_id: str,
    signature: str,
    llm: Any,
    mem_kv_state: Dict[str, Any],
    log_prefix: str,
    log_level: str,
    kv_state_debug_info: Callable[[Any], str],
    log_saved_when_not_minimal: bool,
    log_unsupported_when_not_minimal: bool,
) -> None:
    if not hasattr(llm, "save_state"):
        if log_unsupported_when_not_minimal and log_level != "minimal":
            print(f"{log_prefix} KV state: UNSUPPORTED (no save_state)")
        return

    saved_state = llm.save_state()
    mem_kv_state[session_id] = {"signature": signature, "state": saved_state}

    should_log_saved = False
    if log_saved_when_not_minimal and log_level != "minimal":
        should_log_saved = True
    if log_level == "debug":
        should_log_saved = True
    if should_log_saved:
        print(
            f"{log_prefix} KV state: SAVED (memory) "
            f"session_id={session_id}, sig={signature[:12]}, "
            f"{kv_state_debug_info(saved_state)}"
        )
