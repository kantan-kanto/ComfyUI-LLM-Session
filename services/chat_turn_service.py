# Service that orchestrates A<->B dialogue cycles and transcript updates.
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
    model_manager_factory: Callable[[], Any]
    unload_model: Callable[[Any], None]
    chat_one_turn: Callable[..., str]


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
            model_manager_factory=dependencies.model_manager_factory,
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
        model_manager_factory: Callable[[], Any],
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
        managerA = model_manager_factory()
        managerB = model_manager_factory()
        kv_memory_enabled = (runtime_cache or "off") == "KV_cache"

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
                if not kv_memory_enabled:
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
                if not kv_memory_enabled:
                    try:
                        unload_model(managerB)
                    except Exception:
                        pass
        finally:
            try:
                unload_model(managerA)
            except Exception:
                pass
            try:
                unload_model(managerB)
            except Exception:
                pass

        return "\n".join(transcript_lines)
