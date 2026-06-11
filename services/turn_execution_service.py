# Service for shared single-turn runtime execution and node-execution orchestration.
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict

try:
    from ..core.logging_utils import (
        log_error_safely,
        get_module_logger,
        LOG_LEVEL_MINIMAL,
        LOG_LEVEL_TIMING,
    )
    from .generation_execution_service import GenerationExecutionService
    from .kv_state_service import KvStateService
    from .history_persistence_service import HistoryPersistenceResult, HistoryPersistenceService
except Exception:
    from core.logging_utils import (
        log_error_safely,
        get_module_logger,
        LOG_LEVEL_MINIMAL,
        LOG_LEVEL_TIMING,
    )
    from services.generation_execution_service import GenerationExecutionService
    from services.kv_state_service import KvStateService
    from services.history_persistence_service import HistoryPersistenceResult, HistoryPersistenceService

class TurnExecutionDependencies(TypedDict):
    llama_cpp_available: bool
    llama_cpp_import_error: Any
    is_no_models_placeholder: Callable[[Optional[str]], bool]
    get_llm_model_roots: Callable[[], List[str]]
    resolve_model_and_mmproj: Callable[[List[str], str, str], tuple[str, Optional[str]]]
    mmproj_auto: str
    mmproj_not_required: str
    load_history: Callable[..., tuple[Dict[str, Any], str]]
    clear_kv_state_for_session: Callable[[str], None]
    rewrite_continue_prompt: Callable[..., Any]
    detect_history_language: Callable[[Dict[str, Any]], str]
    session_cache_root: Callable[[str, Optional[str]], str]
    build_chat_messages: Callable[..., List[Dict[str, Any]]]
    build_text_chat_request: Callable[..., Optional[Dict[str, Any]]]
    build_kv_state_signature: Callable[..., str]
    try_restore_kv_state: Callable[..., None]
    is_state_data_mismatch_error: Callable[[Exception], bool]
    saved_llama_state_size: Callable[[Any], Optional[int]]
    current_llama_state_size: Callable[[Any], Optional[int]]
    kv_state_debug_info: Callable[[Any], str]
    get_context_turns: Callable[..., List[Dict[str, Any]]]
    mem_kv_state: Dict[str, Any]
    maybe_compact_summary: Callable[..., Dict[str, Any]]
    cache_debug_label: Callable[[Any], str]
    run_generation_with_adaptive_retry: Callable[..., Any]
    make_suppress_backend_logs: Callable[[bool], Any]
    processing_interrupted: Callable[[], bool]
    throw_if_processing_interrupted: Callable[[], None]
    is_interrupt_error: Callable[[Exception], bool]
    iter_chat_completion_robust: Callable[..., Any]
    create_chat_completion_robust: Callable[..., Dict[str, Any]]
    extract_stream_content: Callable[[Any], str]
    retry_kwargs_with_repeat_last_n_fallback: Callable[[Dict[str, Any], int], Dict[str, Any]]
    strip_reasoning_output: Callable[[str], str]
    next_turn_id: Callable[[Dict[str, Any]], int]
    now_iso: Callable[[], str]
    maybe_summarize_history: Callable[..., Dict[str, Any]]
    atomic_write_json: Callable[[str, Dict[str, Any]], None]
    try_save_kv_state: Callable[..., None]

@dataclass(frozen=True)
class TurnExecutionRequest:
    user_text: str
    session_id: str
    model: str
    mmproj: str
    system_prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    n_gpu_layers: int
    n_ctx: int
    tensor_split: Optional[List[float]] = None
    image: Any = None
    max_turns: Optional[int] = 12
    summarize_old_history: bool = True
    summary_chunk_turns: int = 3
    max_tokens_summary: int = 128
    summary_max_chars: int = 1500
    dynamic_max_tokens: bool = True
    min_generation_tokens: int = 96
    safety_margin_tokens: int = 64
    persistent_cache: str = "off"
    repeat_penalty: float = 1.12
    repeat_last_n: int = 256
    rewrite_continue: bool = True
    runtime_cache: str = "LlamaTrieCache"
    log_level: str = "timing"
    suppress_backend_logs: bool = True
    history_dir: str = ""
    reset_session: bool = False
    stream_to_console: bool = False
    model_manager: Optional[Any] = None
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    advanced_generation_kwargs: Optional[Dict[str, Any]] = None
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None

    # Behavior switches to preserve subtle differences between legacy call paths.
    strip_assistant_before_reasoning_filter: bool = False
    include_image_and_stream_in_turn_params: bool = True
    kv_log_saved_when_not_minimal: bool = False
    kv_log_unsupported_when_not_minimal: bool = False
    include_error_in_invalidate_message: bool = False
    enable_attempt_logging: bool = False

    # Logging label differs by caller ([LLM Session Chat] / [LLM Dialogue Cycle]).
    log_prefix: str = "[LLM Session Chat]"

    # Injected callables and constants from node layer.
    dependencies: TurnExecutionDependencies = field(default_factory=dict)

    @classmethod
    def from_node_inputs(
        cls,
        *,
        user_text: str,
        session_id: str,
        model: str,
        mmproj: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        n_gpu_layers: int,
        n_ctx: int,
        tensor_split: Optional[List[float]] = None,
        image: Any,
        max_turns: Optional[int],
        summarize_old_history: bool,
        summary_chunk_turns: int,
        max_tokens_summary: int,
        summary_max_chars: int,
        dynamic_max_tokens: bool,
        min_generation_tokens: int,
        safety_margin_tokens: int,
        persistent_cache: str,
        repeat_penalty: float,
        repeat_last_n: int,
        rewrite_continue: bool,
        runtime_cache: str,
        log_level: str,
        suppress_backend_logs: bool,
        history_dir: str,
        reset_session: bool,
        stream_to_console: bool,
        model_manager: Optional[Any],
        chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]],
        text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]],
        advanced_generation_kwargs: Optional[Dict[str, Any]] = None,
        advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None,
        strip_assistant_before_reasoning_filter: bool,
        include_image_and_stream_in_turn_params: bool,
        kv_log_saved_when_not_minimal: bool,
        kv_log_unsupported_when_not_minimal: bool,
        include_error_in_invalidate_message: bool,
        enable_attempt_logging: bool,
        log_prefix: str,
        dependencies: TurnExecutionDependencies,
    ) -> "TurnExecutionRequest":
        return cls(
            user_text=user_text,
            session_id=session_id,
            model=model,
            mmproj=mmproj,
            system_prompt=system_prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            n_gpu_layers=int(n_gpu_layers),
            n_ctx=int(n_ctx),
            tensor_split=(list(tensor_split) if tensor_split is not None else None),
            image=image,
            max_turns=(int(max_turns) if max_turns is not None else None),
            summarize_old_history=bool(summarize_old_history),
            summary_chunk_turns=int(summary_chunk_turns),
            max_tokens_summary=int(max_tokens_summary),
            summary_max_chars=int(summary_max_chars),
            dynamic_max_tokens=bool(dynamic_max_tokens),
            min_generation_tokens=int(min_generation_tokens),
            safety_margin_tokens=int(safety_margin_tokens),
            persistent_cache=str(persistent_cache),
            repeat_penalty=float(repeat_penalty),
            repeat_last_n=int(repeat_last_n),
            rewrite_continue=bool(rewrite_continue),
            runtime_cache=str(runtime_cache),
            log_level=str(log_level),
            suppress_backend_logs=bool(suppress_backend_logs),
            history_dir=history_dir or "",
            reset_session=bool(reset_session),
            stream_to_console=bool(stream_to_console),
            model_manager=model_manager,
            chat_handler_overrides=chat_handler_overrides,
            text_chat_builder_overrides=text_chat_builder_overrides,
            advanced_generation_kwargs=(
                dict(advanced_generation_kwargs) if isinstance(advanced_generation_kwargs, dict) else None
            ),
            advanced_summary_generation_kwargs=(
                dict(advanced_summary_generation_kwargs)
                if isinstance(advanced_summary_generation_kwargs, dict)
                else None
            ),
            strip_assistant_before_reasoning_filter=bool(strip_assistant_before_reasoning_filter),
            include_image_and_stream_in_turn_params=bool(include_image_and_stream_in_turn_params),
            kv_log_saved_when_not_minimal=bool(kv_log_saved_when_not_minimal),
            kv_log_unsupported_when_not_minimal=bool(kv_log_unsupported_when_not_minimal),
            include_error_in_invalidate_message=bool(include_error_in_invalidate_message),
            enable_attempt_logging=bool(enable_attempt_logging),
            log_prefix=log_prefix,
            dependencies=dependencies,
        )


@dataclass(frozen=True)
class TurnExecutionResult:
    assistant_text: str
    history_path: Optional[str] = None
    generation_succeeded: bool = False
    error: Optional[Exception] = None
    persistence_succeeded: bool = True
    persistence_error: Optional[Exception] = None


@dataclass(frozen=True)
class SessionChatNodeExecutionRequest:
    model: str
    turn_kwargs: Dict[str, Any]


@dataclass(frozen=True)
class SessionChatNodeExecutionDependencies:
    require_llama_cpp_available: Callable[[], None]
    resolve_valid_model_path: Callable[[str, float], Optional[str]]
    get_or_create_model_manager: Callable[[], Any]
    execute_session_chat_turn: Callable[..., TurnExecutionResult]
    session_chat_error_return: Callable[[float, Optional[str]], tuple]
    log_session_chat_total: Callable[[float, str], None]


class SessionChatNodeExecutionService:
    def run(
        self,
        *,
        request: SessionChatNodeExecutionRequest,
        dependencies: SessionChatNodeExecutionDependencies,
    ) -> tuple:
        start_time = time.perf_counter()
        dependencies.require_llama_cpp_available()
        if dependencies.resolve_valid_model_path(request.model, start_time) is None:
            return ("",)

        turn_kwargs = dict(request.turn_kwargs)
        if turn_kwargs.get("model_manager") is None:
            turn_kwargs["model_manager"] = dependencies.get_or_create_model_manager()
        result = dependencies.execute_session_chat_turn(**turn_kwargs)

        if not result.generation_succeeded:
            if result.error is not None:
                return dependencies.session_chat_error_return(
                    start_time,
                    f"[LLM Session Chat] Error during generation: {result.error}",
                )
            return dependencies.session_chat_error_return(start_time, None)

        if not result.persistence_succeeded:
            module_logger = get_module_logger("SessionChatNodeExecutionService")
            detail = f": {result.persistence_error}" if result.persistence_error is not None else ""
            module_logger(
                f"[LLM Session Chat] Warning: response generated but history was not saved{detail}",
                LOG_LEVEL_MINIMAL,
            )

        dependencies.log_session_chat_total(start_time, "Finished")
        return (result.assistant_text,)

class TurnExecutionService:
    def __init__(
        self,
        generation_execution_service: Optional[GenerationExecutionService] = None,
        kv_state_service: Optional[KvStateService] = None,
        history_persistence_service: Optional[HistoryPersistenceService] = None,
    ) -> None:
        self._generation_execution_service = generation_execution_service or GenerationExecutionService()
        self._kv_state_service = kv_state_service or KvStateService()
        self._history_persistence_service = history_persistence_service or HistoryPersistenceService()

    def _dep(self, deps: TurnExecutionDependencies, key: str) -> Any:
        if key not in deps:
            raise KeyError(f"Missing dependency: {key}")
        return deps[key]

    def execute_from_node_inputs(self, **kwargs: Any) -> TurnExecutionResult:
        request = TurnExecutionRequest.from_node_inputs(**kwargs)
        return self.execute_turn(request)


    def execute_session_chat_turn(self, **kwargs: Any) -> TurnExecutionResult:
        return self.execute_from_node_inputs(
            strip_assistant_before_reasoning_filter=False,
            include_image_and_stream_in_turn_params=True,
            kv_log_saved_when_not_minimal=False,
            kv_log_unsupported_when_not_minimal=False,
            include_error_in_invalidate_message=False,
            enable_attempt_logging=True,
            log_prefix="[LLM Session Chat]",
            **kwargs,
        )

    def execute_dialogue_cycle_turn(self, **kwargs: Any) -> TurnExecutionResult:
        return self.execute_from_node_inputs(
            strip_assistant_before_reasoning_filter=True,
            include_image_and_stream_in_turn_params=False,
            kv_log_saved_when_not_minimal=True,
            kv_log_unsupported_when_not_minimal=True,
            include_error_in_invalidate_message=True,
            enable_attempt_logging=False,
            log_prefix="[LLM Dialogue Cycle]",
            **kwargs,
        )

    def _preflight(
        self, request: TurnExecutionRequest, deps: TurnExecutionDependencies
    ) -> tuple[Optional[Any], Optional[TurnExecutionResult]]:
        llama_cpp_available = bool(deps.get("llama_cpp_available", True))
        if not llama_cpp_available:
            msg = "llama_cpp is not available"
            import_err = deps.get("llama_cpp_import_error")
            if import_err:
                msg += f" ({import_err})"
            return None, TurnExecutionResult(assistant_text="", generation_succeeded=False, error=RuntimeError(msg))

        mgr = request.model_manager
        if mgr is None:
            return None, TurnExecutionResult(
                assistant_text="",
                generation_succeeded=False,
                error=RuntimeError("model_manager is required"),
            )

        is_no_models_placeholder = self._dep(deps, "is_no_models_placeholder")
        if is_no_models_placeholder(request.model):
            return None, TurnExecutionResult(assistant_text="", generation_succeeded=False)

        return mgr, None

    def _resolve_model_paths(
        self, request: TurnExecutionRequest, deps: TurnExecutionDependencies
    ) -> tuple[Optional[str], Optional[str], Optional[TurnExecutionResult]]:
        get_llm_model_roots = self._dep(deps, "get_llm_model_roots")
        resolve_model_and_mmproj = self._dep(deps, "resolve_model_and_mmproj")
        roots = get_llm_model_roots()

        try:
            model_path, mmproj_path = resolve_model_and_mmproj(roots, request.model, request.mmproj)
        except Exception as err:
            return None, None, TurnExecutionResult(assistant_text="", generation_succeeded=False, error=err)

        return model_path, mmproj_path, None

    def _build_model_sig(
        self,
        *,
        deps: TurnExecutionDependencies,
        model_path: str,
        mmproj_path: Optional[str],
        n_ctx: int,
        n_gpu_layers: int,
        tensor_split: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        model_sig = {
            "model_file": os.path.basename(model_path),
            "mmproj_file": (
                os.path.basename(mmproj_path)
                if (mmproj_path and mmproj_path != deps.get("mmproj_not_required"))
                else ""
            ),
            "n_ctx": int(n_ctx),
            "n_gpu_layers": int(n_gpu_layers),
        }
        if tensor_split is not None:
            model_sig["tensor_split"] = [float(x) for x in tensor_split]
        return model_sig

    def _load_history(
        self,
        *,
        request: TurnExecutionRequest,
        deps: TurnExecutionDependencies,
        model_sig: Dict[str, Any],
    ) -> tuple[Dict[str, Any], str]:
        load_history = self._dep(deps, "load_history")
        return load_history(
            session_id=request.session_id,
            history_dir=(request.history_dir or None),
            system_prompt=request.system_prompt,
            model_sig=model_sig,
            log_level=request.log_level,
            reset_session=bool(request.reset_session),
        )

    def _reset_state_if_needed(
        self, *, request: TurnExecutionRequest, deps: TurnExecutionDependencies, mgr: Any
    ) -> Any:
        clear_kv_state_for_session = self._dep(deps, "clear_kv_state_for_session")
        if bool(request.reset_session):
            clear_kv_state_for_session(request.session_id)
            try:
                if getattr(mgr, "model", None) is not None:
                    mgr.invalidate_cache(mgr.model, remove_disk_data=False)
            except Exception as e:
                # P0: Log cache invalidation failure during reset to detect state inconsistency
                log_error_safely("TurnExecutionService", e, "Failed to invalidate cache during session reset", LOG_LEVEL_MINIMAL)
        return clear_kv_state_for_session

    def _rewrite_user_text(
        self,
        *,
        request: TurnExecutionRequest,
        deps: TurnExecutionDependencies,
        history: Dict[str, Any],
    ) -> str:
        rewrite_continue_prompt = self._dep(deps, "rewrite_continue_prompt")
        detect_history_language = self._dep(deps, "detect_history_language")
        continue_result = rewrite_continue_prompt(
            user_text=(request.user_text or ""),
            history=history,
            rewrite_continue=bool(request.rewrite_continue),
            detect_history_language=detect_history_language,
        )
        if continue_result.rewritten and request.log_level == "debug":
            print(f"{request.log_prefix} Continue detected, language: {continue_result.detected_language}")
        return continue_result.user_text_for_model

    def _load_model_with_cache(
        self,
        *,
        request: TurnExecutionRequest,
        deps: TurnExecutionDependencies,
        mgr: Any,
        model_path: str,
        mmproj_path: Optional[str],
        hist_path: str,
    ) -> tuple[Optional[Any], Optional[TurnExecutionResult]]:
        session_cache_root = self._dep(deps, "session_cache_root")
        try:
            mgr.cache_dir_override = session_cache_root(request.session_id, request.history_dir or None)
            os.makedirs(mgr.cache_dir_override, exist_ok=True)
        except Exception as e:
            # P0: Log cache directory creation failure and continue without cache
            log_error_safely("TurnExecutionService", e, "Failed to create cache directory, continuing without cache", LOG_LEVEL_MINIMAL)
            mgr.cache_dir_override = None

        mmproj_choice = str(request.mmproj or "").strip()
        mmproj_choice_normalized = "".join(c for c in mmproj_choice.lower() if c.isalnum())
        mmproj_auto = str(deps.get("mmproj_auto", "(Auto-detect)") or "").strip()
        mmproj_auto_normalized = "".join(c for c in mmproj_auto.lower() if c.isalnum())
        mmproj_not_required = str(deps.get("mmproj_not_required", "") or "").strip()
        explicit_mmproj_selected = bool(mmproj_choice) and (
            mmproj_choice != mmproj_not_required
            and mmproj_choice_normalized not in {"autodetect", "auto", mmproj_auto_normalized}
        )
        vision_required = request.image is not None or explicit_mmproj_selected

        try:
            llm = mgr.load_model(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_ctx=int(request.n_ctx),
                n_gpu_layers=int(request.n_gpu_layers),
                tensor_split=request.tensor_split,
                chat_handler_overrides=request.chat_handler_overrides,
                vision_required=vision_required,
                verbose=False,
            )
        except Exception as err:
            return (
                None,
                TurnExecutionResult(
                    assistant_text="",
                    history_path=hist_path,
                    generation_succeeded=False,
                    error=err,
                ),
            )

        try:
            mgr.configure_cache(
                llm,
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_ctx=int(request.n_ctx),
                n_gpu_layers=int(request.n_gpu_layers),
                tensor_split=request.tensor_split,
                persistent_cache=request.persistent_cache,
                runtime_cache=request.runtime_cache,
            )
        except Exception:
            pass

        return llm, None

    def _build_generation_inputs(
        self,
        *,
        request: TurnExecutionRequest,
        deps: TurnExecutionDependencies,
        history: Dict[str, Any],
        user_text_for_model: str,
        model_path: str,
        mmproj_path: Optional[str],
    ) -> tuple[Any, Any]:
        build_chat_messages = self._dep(deps, "build_chat_messages")
        build_text_chat_request = self._dep(deps, "build_text_chat_request")
        messages = build_chat_messages(
            history=history,
            user_text=user_text_for_model or "",
            image_tensor=request.image,
            max_turns=int(request.max_turns) if request.max_turns is not None else None,
            summarize_old_history=bool(request.summarize_old_history),
            system_prompt=request.system_prompt or "",
        )
        text_chat_request = build_text_chat_request(
            model_path=model_path,
            mmproj_path=mmproj_path,
            messages=messages,
            text_chat_builder_overrides=request.text_chat_builder_overrides,
        )
        return messages, text_chat_request

    def _maybe_restore_kv_state(
        self,
        *,
        request: TurnExecutionRequest,
        deps: TurnExecutionDependencies,
        mgr: Any,
        llm: Any,
        history: Dict[str, Any],
        model_path: str,
        mmproj_path: Optional[str],
        clear_kv_state_for_session: Any,
    ) -> tuple[Any, Any, Any]:
        return self._kv_state_service.restore_state(
            request=request,
            deps=deps,
            mgr=mgr,
            llm=llm,
            history=history,
            model_path=model_path,
            mmproj_path=mmproj_path,
            clear_kv_state_for_session=clear_kv_state_for_session,
        )

    def _persist_history_and_summary(
        self,
        *,
        request: TurnExecutionRequest,
        deps: TurnExecutionDependencies,
        history: Dict[str, Any],
        assistant_text: str,
        generation_result: Any,
        llm: Any,
        hist_path: str,
        model_path: str,
        mmproj_path: Optional[str],
    ) -> HistoryPersistenceResult:
        return self._history_persistence_service.persist_history_and_summary(
            request=request,
            deps=deps,
            history=history,
            assistant_text=assistant_text,
            generation_result=generation_result,
            llm=llm,
            hist_path=hist_path,
            model_path=model_path,
            mmproj_path=mmproj_path,
        )

    def _maybe_save_kv_state(
        self,
        *,
        request: TurnExecutionRequest,
        deps: TurnExecutionDependencies,
        llm: Any,
        history: Dict[str, Any],
        model_path: str,
        mmproj_path: Optional[str],
        kv_state_debug_info: Any,
        get_context_turns: Any,
    ) -> None:
        self._kv_state_service.save_state(
            request=request,
            deps=deps,
            llm=llm,
            history=history,
            model_path=model_path,
            mmproj_path=mmproj_path,
            kv_state_debug_info=kv_state_debug_info,
            get_context_turns=get_context_turns,
        )

    def execute_turn(self, request: TurnExecutionRequest) -> TurnExecutionResult:
        deps = request.dependencies
        module_logger = get_module_logger("TurnExecutionService")

        mgr, early_result = self._preflight(request, deps)
        if early_result is not None:
            return early_result

        model_path, mmproj_path, early_result = self._resolve_model_paths(request, deps)
        if early_result is not None:
            return early_result
        assert mgr is not None and model_path is not None

        model_sig = self._build_model_sig(
            deps=deps,
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=int(request.n_ctx),
            n_gpu_layers=int(request.n_gpu_layers),
            tensor_split=request.tensor_split,
        )
        history, hist_path = self._load_history(request=request, deps=deps, model_sig=model_sig)
        clear_kv_state_for_session = self._reset_state_if_needed(request=request, deps=deps, mgr=mgr)
        user_text_for_model = self._rewrite_user_text(request=request, deps=deps, history=history)

        llm, early_result = self._load_model_with_cache(
            request=request,
            deps=deps,
            mgr=mgr,
            model_path=model_path,
            mmproj_path=mmproj_path,
            hist_path=hist_path,
        )
        if early_result is not None:
            return early_result
        assert llm is not None

        build_chat_messages = self._dep(deps, "build_chat_messages")
        build_text_chat_request = self._dep(deps, "build_text_chat_request")
        messages, text_chat_request = self._build_generation_inputs(
            request=request,
            deps=deps,
            history=history,
            user_text_for_model=user_text_for_model,
            model_path=model_path,
            mmproj_path=mmproj_path,
        )

        def _message_chars(msgs: Any) -> int:
            total = 0
            if not isinstance(msgs, list):
                return total
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                c = m.get("content", "")
                if isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            total += len(part["text"])
                elif isinstance(c, str):
                    total += len(c)
            return total

        def _prompt_chars(req_obj: Any) -> int:
            if isinstance(req_obj, dict) and isinstance(req_obj.get("prompt"), str):
                return len(req_obj["prompt"])
            return 0

        def _log_prompt_metrics(stage: str, turns_limit: Optional[int], msgs: Any, req_obj: Any) -> None:
            if request.log_level not in {"debug", "timing"}:
                return
            message_count = len(msgs) if isinstance(msgs, list) else 0
            msg_chars = _message_chars(msgs)
            prompt_chars = _prompt_chars(req_obj)
            module_logger(
                f"{request.log_prefix} {stage}: turns_limit={turns_limit}, "
                f"messages={message_count}, message_chars={msg_chars}, prompt_chars={prompt_chars}",
                LOG_LEVEL_TIMING,
            )

        _log_prompt_metrics("Prompt metrics (initial)", request.max_turns, messages, text_chat_request)

        is_state_data_mismatch_error, kv_state_debug_info, get_context_turns = self._maybe_restore_kv_state(
            request=request,
            deps=deps,
            mgr=mgr,
            llm=llm,
            history=history,
            model_path=model_path,
            mmproj_path=mmproj_path,
            clear_kv_state_for_session=clear_kv_state_for_session,
        )

        maybe_compact_summary = self._dep(deps, "maybe_compact_summary")

        def _on_state_cache_mismatch(err: Exception) -> None:
            self._kv_state_service.on_state_cache_mismatch(
                request=request,
                deps=deps,
                clear_kv_state_for_session=clear_kv_state_for_session,
                mgr=mgr,
                llm=llm,
                err=err,
            )

        def _on_compact_summary() -> None:
            nonlocal history
            if request.summarize_old_history:
                try:
                    history = maybe_compact_summary(
                        model=llm,
                        history=history,
                        summary_max_chars=int(request.summary_max_chars),
                        temperature=0.2,
                        max_tokens_summary=int(request.max_tokens_summary),
                        suppress_logs=(request.log_level != "debug"),
                        model_path=model_path,
                        mmproj_path=mmproj_path,
                        text_chat_builder_overrides=request.text_chat_builder_overrides,
                        advanced_generation_kwargs=request.advanced_summary_generation_kwargs,
                    )
                except Exception as e:
                    # P1: Log summary compaction failure for debugging
                    log_error_safely("TurnExecutionService", e, "Failed to compact summary", LOG_LEVEL_MINIMAL)

        def _rebuild_messages_for_turns_limit(new_turns_limit: Optional[int]):
            rebuilt_messages = build_chat_messages(
                history=history,
                user_text=user_text_for_model or "",
                image_tensor=request.image,
                max_turns=new_turns_limit,
                summarize_old_history=bool(request.summarize_old_history),
                system_prompt=request.system_prompt or "",
            )
            rebuilt_text_chat_request = build_text_chat_request(
                model_path=model_path,
                mmproj_path=mmproj_path,
                messages=rebuilt_messages,
                text_chat_builder_overrides=request.text_chat_builder_overrides,
            )
            _log_prompt_metrics("Prompt metrics (retry rebuild)", new_turns_limit, rebuilt_messages, rebuilt_text_chat_request)
            return rebuilt_messages, rebuilt_text_chat_request

        generation_outcome = self._generation_execution_service.execute(
            request=request,
            deps=deps,
            llm=llm,
            messages=messages,
            text_chat_request=text_chat_request,
            is_state_data_mismatch_error=is_state_data_mismatch_error,
            on_state_cache_mismatch=_on_state_cache_mismatch,
            on_compact_summary=_on_compact_summary,
            rebuild_messages_for_turns_limit=_rebuild_messages_for_turns_limit,
        )

        if generation_outcome.failed:
            return TurnExecutionResult(
                assistant_text="",
                history_path=hist_path,
                generation_succeeded=False,
                error=generation_outcome.error,
            )

        assistant_text = generation_outcome.assistant_text

        persistence_result = self._persist_history_and_summary(
            request=request,
            deps=deps,
            history=history,
            assistant_text=assistant_text,
            generation_result=generation_outcome.generation_result,
            llm=llm,
            hist_path=hist_path,
            model_path=model_path,
            mmproj_path=mmproj_path,
        )
        history = persistence_result.history
        self._maybe_save_kv_state(
            request=request,
            deps=deps,
            llm=llm,
            history=history,
            model_path=model_path,
            mmproj_path=mmproj_path,
            kv_state_debug_info=kv_state_debug_info,
            get_context_turns=get_context_turns,
        )

        return TurnExecutionResult(
            assistant_text=assistant_text,
            history_path=hist_path,
            generation_succeeded=True,
            error=None,
            persistence_succeeded=persistence_result.persistence_succeeded,
            persistence_error=persistence_result.persistence_error,
        )
