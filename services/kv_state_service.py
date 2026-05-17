# Service for KV cache state restore/save and mismatch recovery orchestration.
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

try:
    from ..core.logging_utils import log_error_safely, LOG_LEVEL_MINIMAL
except Exception:
    from core.logging_utils import log_error_safely, LOG_LEVEL_MINIMAL


class KvStateService:
    def _dep(self, deps: Mapping[str, Any], key: str) -> Any:
        if key not in deps:
            raise KeyError(f"Missing dependency: {key}")
        return deps[key]

    def restore_state(
        self,
        *,
        request: Any,
        deps: Mapping[str, Any],
        mgr: Any,
        llm: Any,
        history: Dict[str, Any],
        model_path: str,
        mmproj_path: Optional[str],
        clear_kv_state_for_session: Callable[[str], None],
    ) -> tuple[Any, Any, Any]:
        build_kv_state_signature = self._dep(deps, "build_kv_state_signature")
        try_restore_kv_state = self._dep(deps, "try_restore_kv_state")
        is_state_data_mismatch_error = self._dep(deps, "is_state_data_mismatch_error")
        saved_llama_state_size = self._dep(deps, "saved_llama_state_size")
        current_llama_state_size = self._dep(deps, "current_llama_state_size")
        kv_state_debug_info = self._dep(deps, "kv_state_debug_info")
        get_context_turns = self._dep(deps, "get_context_turns")

        if (request.runtime_cache or "off") == "KV_cache" and request.image is None:
            try:
                kv_sig = build_kv_state_signature(
                    history=history,
                    max_turns=request.max_turns,
                    summarize_old_history=bool(request.summarize_old_history),
                    system_prompt=(request.system_prompt or ""),
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    n_ctx=int(request.n_ctx),
                    n_gpu_layers=int(request.n_gpu_layers),
                    tensor_split=getattr(request, "tensor_split", None),
                    get_context_turns=get_context_turns,
                )
                try_restore_kv_state(
                    session_id=request.session_id,
                    signature=kv_sig,
                    llm=llm,
                    mem_kv_state=self._dep(deps, "mem_kv_state"),
                    log_prefix=request.log_prefix,
                    log_level=request.log_level,
                    model_path=model_path,
                    n_ctx=int(request.n_ctx),
                    n_gpu_layers=int(request.n_gpu_layers),
                    clear_kv_state_for_session=clear_kv_state_for_session,
                    is_state_data_mismatch_error=is_state_data_mismatch_error,
                    invalidate_cache=lambda _llm, remove_disk_data: mgr.invalidate_cache(
                        _llm, remove_disk_data=remove_disk_data
                    ),
                    saved_llama_state_size=saved_llama_state_size,
                    current_llama_state_size=current_llama_state_size,
                    kv_state_debug_info=kv_state_debug_info,
                    include_error_in_invalidate_message=bool(request.include_error_in_invalidate_message),
                )
            except Exception as e:
                # P1: Log KV state restore failure for debugging
                log_error_safely("TurnExecutionService", e, "Failed to restore KV state", LOG_LEVEL_MINIMAL)

        return is_state_data_mismatch_error, kv_state_debug_info, get_context_turns

    def save_state(
        self,
        *,
        request: Any,
        deps: Mapping[str, Any],
        llm: Any,
        history: Dict[str, Any],
        model_path: str,
        mmproj_path: Optional[str],
        kv_state_debug_info: Any,
        get_context_turns: Any,
    ) -> None:
        if (request.runtime_cache or "off") != "KV_cache" or request.image is not None:
            return

        try:
            build_kv_state_signature = self._dep(deps, "build_kv_state_signature")
            kv_sig2 = build_kv_state_signature(
                history=history,
                max_turns=request.max_turns,
                summarize_old_history=bool(request.summarize_old_history),
                system_prompt=(request.system_prompt or ""),
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_ctx=int(request.n_ctx),
                n_gpu_layers=int(request.n_gpu_layers),
                tensor_split=getattr(request, "tensor_split", None),
                get_context_turns=get_context_turns,
            )
            try_save_kv_state = self._dep(deps, "try_save_kv_state")
            try_save_kv_state(
                session_id=request.session_id,
                signature=kv_sig2,
                llm=llm,
                mem_kv_state=self._dep(deps, "mem_kv_state"),
                log_prefix=request.log_prefix,
                log_level=request.log_level,
                kv_state_debug_info=kv_state_debug_info,
                log_saved_when_not_minimal=bool(request.kv_log_saved_when_not_minimal),
                log_unsupported_when_not_minimal=bool(request.kv_log_unsupported_when_not_minimal),
            )
        except Exception as e:
            # P1: Log KV state save failure for debugging
            log_error_safely("TurnExecutionService", e, "Failed to save KV state", LOG_LEVEL_MINIMAL)

    def on_state_cache_mismatch(
        self,
        *,
        request: Any,
        deps: Mapping[str, Any],
        clear_kv_state_for_session: Callable[[str], None],
        mgr: Any,
        llm: Any,
        err: Exception,
    ) -> None:
        clear_kv_state_for_session(request.session_id)
        if request.log_level != "minimal":
            cache_debug_label = self._dep(deps, "cache_debug_label")
            print(
                f"{request.log_prefix} Cache mismatch details: "
                f"session_id={request.session_id}, cache={cache_debug_label(mgr)}, error={err}"
            )
        try:
            mgr.invalidate_cache(llm, remove_disk_data=True)
        except Exception as e:
            # P1: Log cache invalidation failure on cache mismatch
            log_error_safely("TurnExecutionService", e, "Failed to invalidate cache on mismatch", LOG_LEVEL_MINIMAL)
        if request.log_level != "minimal":
            print(f"{request.log_prefix} Detected incompatible cache state; cache invalidated and retrying once.")
