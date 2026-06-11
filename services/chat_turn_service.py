# Service for dialogue-cycle runtime orchestration and node-execution orchestration.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
@dataclass(frozen=True)
class DialogueCycleRequest:
    initial_user_text: str
    session_id: str
    cycles: int
    system_prompt: str
    system_prompt_A: str
    system_prompt_B: str
    runtime_cache: str
    stream_to_console: bool
    reset_session: bool
    history_dir: str
    turn_kwargs_A: Dict[str, Any]
    turn_kwargs_B: Dict[str, Any]


@dataclass(frozen=True)
class DialogueCycleDependencies:
    now_iso: Callable[[], str]
    transcript_path: Callable[[str, Optional[str]], str]
    append_transcript_lines: Callable[[str, List[str]], None]
    clear_kv_state_for_session: Callable[[str], None]
    get_or_create_model_manager: Callable[[str], Any]
    unload_model: Callable[[Any], None]
    chat_one_turn: Callable[..., str]


@dataclass(frozen=True)
class DialogueCycleNodeExecutionRequest:
    initial_user_text: str
    session_id: str
    cycles: int
    modelA: str
    mmprojA: str
    modelB: str
    mmprojB: str
    system_prompt: str
    system_prompt_A: str
    system_prompt_B: str
    max_tokens: int
    temperature: float
    top_p: float
    n_gpu_layers: int
    tensor_split: Optional[List[float]]
    n_ctx: int
    max_turns: int
    summarize_old_history: bool
    summary_chunk_turns: int
    max_tokens_summary: int
    summary_max_chars: int
    dynamic_max_tokens: bool
    min_generation_tokens: int
    safety_margin_tokens: int
    persistent_cache: str
    runtime_cache: str
    repeat_penalty: float
    repeat_last_n: int
    rewrite_continue: bool
    log_level: str
    suppress_backend_logs: bool
    history_dir: str
    reset_session: bool
    stream_to_console: bool
    chat_handler_overrides: Optional[Dict[str, Dict[str, Any]]]
    text_chat_builder_overrides: Optional[Dict[str, Dict[str, Any]]]
    advanced_generation_kwargs: Optional[Dict[str, Any]] = None
    advanced_summary_generation_kwargs: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DialogueCycleNodeExecutionDependencies:
    build_common_turn_kwargs: Callable[..., Dict[str, Any]]
    build_dialogue_cycle_request: Callable[..., DialogueCycleRequest]
    build_dialogue_cycle_dependencies: Callable[[], DialogueCycleDependencies]
    run_dialogue_cycle_with_dependencies: Callable[..., str]


class DialogueCycleNodeExecutionService:
    def run(
        self,
        *,
        request: DialogueCycleNodeExecutionRequest,
        dependencies: DialogueCycleNodeExecutionDependencies,
    ) -> str:
        common_turn_kwargs = dependencies.build_common_turn_kwargs(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            n_gpu_layers=request.n_gpu_layers,
            tensor_split=request.tensor_split,
            n_ctx=request.n_ctx,
            max_turns=request.max_turns,
            summarize_old_history=request.summarize_old_history,
            summary_chunk_turns=request.summary_chunk_turns,
            max_tokens_summary=request.max_tokens_summary,
            summary_max_chars=request.summary_max_chars,
            dynamic_max_tokens=request.dynamic_max_tokens,
            min_generation_tokens=request.min_generation_tokens,
            safety_margin_tokens=request.safety_margin_tokens,
            persistent_cache=request.persistent_cache,
            repeat_penalty=request.repeat_penalty,
            repeat_last_n=request.repeat_last_n,
            rewrite_continue=request.rewrite_continue,
            runtime_cache=request.runtime_cache,
            log_level=request.log_level,
            suppress_backend_logs=request.suppress_backend_logs,
            chat_handler_overrides=request.chat_handler_overrides,
            text_chat_builder_overrides=request.text_chat_builder_overrides,
            advanced_generation_kwargs=request.advanced_generation_kwargs,
            advanced_summary_generation_kwargs=request.advanced_summary_generation_kwargs,
        )
        dialogue_request = dependencies.build_dialogue_cycle_request(
            initial_user_text=request.initial_user_text,
            session_id=request.session_id,
            cycles=request.cycles,
            system_prompt=request.system_prompt,
            system_prompt_A=request.system_prompt_A,
            system_prompt_B=request.system_prompt_B,
            runtime_cache=request.runtime_cache,
            stream_to_console=request.stream_to_console,
            reset_session=request.reset_session,
            history_dir=request.history_dir,
            common_turn_kwargs=common_turn_kwargs,
            model_a=request.modelA,
            mmproj_a=request.mmprojA,
            model_b=request.modelB,
            mmproj_b=request.mmprojB,
        )
        dialogue_dependencies = dependencies.build_dialogue_cycle_dependencies()
        return dependencies.run_dialogue_cycle_with_dependencies(
            request=dialogue_request,
            dependencies=dialogue_dependencies,
        )


class ChatTurnService:
    def run_dialogue_cycle_with_dependencies(
        self,
        *,
        request: DialogueCycleRequest,
        dependencies: DialogueCycleDependencies,
    ) -> str:
        return self.run_dialogue_cycle(
            initial_user_text=request.initial_user_text,
            session_id=request.session_id,
            cycles=request.cycles,
            system_prompt=request.system_prompt,
            system_prompt_A=request.system_prompt_A,
            system_prompt_B=request.system_prompt_B,
            runtime_cache=request.runtime_cache,
            stream_to_console=request.stream_to_console,
            reset_session=request.reset_session,
            history_dir=request.history_dir,
            now_iso=dependencies.now_iso,
            transcript_path=dependencies.transcript_path,
            append_transcript_lines=dependencies.append_transcript_lines,
            clear_kv_state_for_session=dependencies.clear_kv_state_for_session,
            get_or_create_model_manager=dependencies.get_or_create_model_manager,
            unload_model=dependencies.unload_model,
            chat_one_turn=dependencies.chat_one_turn,
            turn_kwargs_A=request.turn_kwargs_A,
            turn_kwargs_B=request.turn_kwargs_B,
        )

    def run_dialogue_cycle(
        self,
        *,
        initial_user_text: str,
        session_id: str,
        cycles: int,
        system_prompt: str,
        system_prompt_A: str,
        system_prompt_B: str,
        runtime_cache: str,
        stream_to_console: bool,
        reset_session: bool,
        history_dir: str,
        now_iso: Callable[[], str],
        transcript_path: Callable[[str, Optional[str]], str],
        append_transcript_lines: Callable[[str, List[str]], None],
        clear_kv_state_for_session: Callable[[str], None],
        get_or_create_model_manager: Callable[[str], Any],
        unload_model: Callable[[Any], None],
        chat_one_turn: Callable[..., str],
        turn_kwargs_A: Dict[str, Any],
        turn_kwargs_B: Dict[str, Any],
    ) -> str:
        base_id = (session_id or "default").strip() or "default"
        sidA = f"{base_id}_A"
        sidB = f"{base_id}_B"
        tpath = transcript_path(base_id, history_dir or None)

        reset = bool(reset_session)
        if reset:
            clear_kv_state_for_session(sidA)
            clear_kv_state_for_session(sidB)

        transcript_lines: List[str] = []
        managerA = get_or_create_model_manager("A")
        managerB = get_or_create_model_manager("B")
        keep_loaded_mode = (runtime_cache or "off") in {"KV_cache", "LlamaTrieCache"}

        try:
            first = initial_user_text or ""
            line0 = f"[{now_iso()}] USER → A: {first}"
            append_transcript_lines(tpath, [line0])
            transcript_lines.append(line0)

            msg = first
            for _ in range(int(max(1, cycles))):
                sysA = (system_prompt_A or "").strip() or (system_prompt or "")
                lastA = chat_one_turn(
                    user_text=msg,
                    session_id=sidA,
                    system_prompt=sysA,
                    history_dir=history_dir,
                    reset_session=reset,
                    stream_to_console=bool(stream_to_console),
                    model_manager=managerA,
                    **turn_kwargs_A,
                )
                lineA = f"[{now_iso()}] A: {lastA}"
                append_transcript_lines(tpath, [lineA])
                transcript_lines.append(lineA)
                msg = lastA or ""
                if not keep_loaded_mode:
                    try:
                        unload_model(managerA)
                    except Exception:
                        pass
                reset_for_b = reset
                reset = False

                sysB = (system_prompt_B or "").strip() or (system_prompt or "")
                lastB = chat_one_turn(
                    user_text=msg,
                    session_id=sidB,
                    system_prompt=sysB,
                    history_dir=history_dir,
                    reset_session=reset_for_b,
                    stream_to_console=bool(stream_to_console),
                    model_manager=managerB,
                    **turn_kwargs_B,
                )
                lineB = f"[{now_iso()}] B: {lastB}"
                append_transcript_lines(tpath, [lineB])
                transcript_lines.append(lineB)
                msg = lastB or ""
                if not keep_loaded_mode:
                    try:
                        unload_model(managerB)
                    except Exception:
                        pass
        finally:
            if not keep_loaded_mode:
                try:
                    unload_model(managerA)
                except Exception:
                    pass
                try:
                    unload_model(managerB)
                except Exception:
                    pass

        return "\n".join(transcript_lines)





