from __future__ import annotations

from services.chat_turn_service import (
    ChatTurnService,
    DialogueCycleDependencies,
    DialogueCycleNodeExecutionDependencies,
    DialogueCycleNodeExecutionRequest,
    DialogueCycleNodeExecutionService,
    DialogueCycleRequest,
)


class DummyManager:
    def __init__(self, name: str):
        self.name = name


def test_run_dialogue_cycle_writes_transcript_and_unloads_non_kv_cache() -> None:
    service = ChatTurnService()
    appended: list[tuple[str, list[str]]] = []
    unload_calls: list[str] = []

    def _append(path: str, lines: list[str]) -> None:
        appended.append((path, list(lines)))

    managers = [DummyManager("A"), DummyManager("B")]

    def _factory():
        return managers.pop(0)

    def _chat_one_turn(**kwargs):
        sid = kwargs["session_id"]
        msg = kwargs["user_text"]
        return f"{sid}:{msg}"

    transcript = service.run_dialogue_cycle(
        initial_user_text="hello",
        session_id="sid",
        cycles=1,
        system_prompt="sys",
        system_prompt_A="",
        system_prompt_B="",
        runtime_cache="off",
        stream_to_console=False,
        reset_session=False,
        history_dir="",
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
        transcript_path=lambda _sid, _history_dir: "transcript.txt",
        append_transcript_lines=_append,
        clear_kv_state_for_session=lambda _sid: None,
        get_or_create_model_manager=lambda role: _factory(),
        unload_model=lambda manager: unload_calls.append(manager.name),
        chat_one_turn=_chat_one_turn,
        turn_kwargs_A={"model": "ma", "mmproj": "(Auto detect)"},
        turn_kwargs_B={"model": "mb", "mmproj": "(Auto detect)"},
    )

    lines = [entry[1][0] for entry in appended]
    assert lines[0].endswith("USER → A: hello")
    assert lines[1].endswith("A: sid_A:hello")
    assert lines[2].endswith("B: sid_B:sid_A:hello")
    assert "A: sid_A:hello" in transcript
    assert "B: sid_B:sid_A:hello" in transcript
    # non-KV cache unloads once per turn and again in finally
    assert unload_calls.count("A") == 2
    assert unload_calls.count("B") == 2


def test_run_dialogue_cycle_reset_session_applies_only_first_round() -> None:
    service = ChatTurnService()
    reset_flags: list[bool] = []

    managers = [DummyManager("A"), DummyManager("B")]

    def _factory():
        return managers.pop(0)

    def _chat_one_turn(**kwargs):
        reset_flags.append(bool(kwargs["reset_session"]))
        return "ok"

    service.run_dialogue_cycle(
        initial_user_text="hello",
        session_id="sid",
        cycles=2,
        system_prompt="sys",
        system_prompt_A="",
        system_prompt_B="",
        runtime_cache="off",
        stream_to_console=False,
        reset_session=True,
        history_dir="",
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
        transcript_path=lambda _sid, _history_dir: "transcript.txt",
        append_transcript_lines=lambda _path, _lines: None,
        clear_kv_state_for_session=lambda _sid: None,
        get_or_create_model_manager=lambda role: _factory(),
        unload_model=lambda _manager: None,
        chat_one_turn=_chat_one_turn,
        turn_kwargs_A={"model": "ma", "mmproj": "(Auto detect)"},
        turn_kwargs_B={"model": "mb", "mmproj": "(Auto detect)"},
    )

    assert reset_flags == [True, True, False, False]


def test_run_dialogue_cycle_kv_cache_skips_in_loop_unload() -> None:
    service = ChatTurnService()
    unload_calls: list[str] = []

    managers = [DummyManager("A"), DummyManager("B")]

    def _factory():
        return managers.pop(0)

    service.run_dialogue_cycle(
        initial_user_text="hello",
        session_id="sid",
        cycles=1,
        system_prompt="sys",
        system_prompt_A="",
        system_prompt_B="",
        runtime_cache="KV_cache",
        stream_to_console=False,
        reset_session=False,
        history_dir="",
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
        transcript_path=lambda _sid, _history_dir: "transcript.txt",
        append_transcript_lines=lambda _path, _lines: None,
        clear_kv_state_for_session=lambda _sid: None,
        get_or_create_model_manager=lambda role: _factory(),
        unload_model=lambda manager: unload_calls.append(manager.name),
        chat_one_turn=lambda **_kwargs: "ok",
        turn_kwargs_A={"model": "ma", "mmproj": "(Auto detect)"},
        turn_kwargs_B={"model": "mb", "mmproj": "(Auto detect)"},
    )

    # KV cache keeps managers loaded across the cycle and skips final unload.
    assert unload_calls.count("A") == 0
    assert unload_calls.count("B") == 0


def test_run_dialogue_cycle_with_dependencies_uses_request_and_deps() -> None:
    service = ChatTurnService()
    appended: list[tuple[str, list[str]]] = []
    unload_calls: list[str] = []

    def _append(path: str, lines: list[str]) -> None:
        appended.append((path, list(lines)))

    managers = [DummyManager("A"), DummyManager("B")]

    def _factory():
        return managers.pop(0)

    def _chat_one_turn(**kwargs):
        sid = kwargs["session_id"]
        msg = kwargs["user_text"]
        return f"{sid}:{msg}"

    request = DialogueCycleRequest(
        initial_user_text="hello",
        session_id="sid",
        cycles=1,
        system_prompt="sys",
        system_prompt_A="",
        system_prompt_B="",
        runtime_cache="off",
        stream_to_console=False,
        reset_session=False,
        history_dir="",
        turn_kwargs_A={"model": "ma", "mmproj": "(Auto detect)"},
        turn_kwargs_B={"model": "mb", "mmproj": "(Auto detect)"},
    )
    dependencies = DialogueCycleDependencies(
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
        transcript_path=lambda _sid, _history_dir: "transcript.txt",
        append_transcript_lines=_append,
        clear_kv_state_for_session=lambda _sid: None,
        get_or_create_model_manager=lambda role: _factory(),
        unload_model=lambda manager: unload_calls.append(manager.name),
        chat_one_turn=_chat_one_turn,
    )

    transcript = service.run_dialogue_cycle_with_dependencies(
        request=request,
        dependencies=dependencies,
    )

    lines = [entry[1][0] for entry in appended]
    assert lines[0].endswith("USER → A: hello")
    assert lines[1].endswith("A: sid_A:hello")
    assert lines[2].endswith("B: sid_B:sid_A:hello")
    assert "A: sid_A:hello" in transcript
    assert "B: sid_B:sid_A:hello" in transcript
    assert unload_calls.count("A") == 2
    assert unload_calls.count("B") == 2



def test_dialogue_cycle_node_execution_service_builds_and_runs() -> None:
    service = DialogueCycleNodeExecutionService()
    calls: dict[str, object] = {}

    request = DialogueCycleNodeExecutionRequest(
        initial_user_text="hello",
        session_id="sid",
        cycles=1,
        modelA="ma",
        mmprojA="(Auto detect)",
        modelB="mb",
        mmprojB="(Auto detect)",
        system_prompt="sys",
        system_prompt_A="",
        system_prompt_B="",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        n_gpu_layers=0,
        n_ctx=1024,
        max_turns=12,
        summarize_old_history=True,
        summary_chunk_turns=3,
        max_tokens_summary=128,
        summary_max_chars=1500,
        dynamic_max_tokens=True,
        min_generation_tokens=96,
        safety_margin_tokens=64,
        persistent_cache="off",
        runtime_cache="off",
        repeat_penalty=1.12,
        repeat_last_n=256,
        rewrite_continue=True,
        log_level="timing",
        suppress_backend_logs=True,
        history_dir="",
        reset_session=False,
        stream_to_console=False,
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
    )

    def _build_common_turn_kwargs(**kwargs):
        calls["common"] = kwargs
        return {"k": "v"}

    def _build_dialogue_cycle_request(**kwargs):
        calls["request_kwargs"] = kwargs
        return DialogueCycleRequest(
            initial_user_text="hello",
            session_id="sid",
            cycles=1,
            system_prompt="sys",
            system_prompt_A="",
            system_prompt_B="",
            runtime_cache="off",
            stream_to_console=False,
            reset_session=False,
            history_dir="",
            turn_kwargs_A={"model": "ma"},
            turn_kwargs_B={"model": "mb"},
        )

    def _build_dialogue_cycle_dependencies():
        calls["deps_built"] = True
        return DialogueCycleDependencies(
            now_iso=lambda: "now",
            transcript_path=lambda _sid, _hist: "t.txt",
            append_transcript_lines=lambda _p, _l: None,
            clear_kv_state_for_session=lambda _sid: None,
            get_or_create_model_manager=lambda _role: object(),
            unload_model=lambda _m: None,
            chat_one_turn=lambda **_kwargs: "ok",
        )

    def _run_dialogue_cycle_with_dependencies(*, request, dependencies):
        calls["runner"] = (request, dependencies)
        return "transcript"

    deps = DialogueCycleNodeExecutionDependencies(
        build_common_turn_kwargs=_build_common_turn_kwargs,
        build_dialogue_cycle_request=_build_dialogue_cycle_request,
        build_dialogue_cycle_dependencies=_build_dialogue_cycle_dependencies,
        run_dialogue_cycle_with_dependencies=_run_dialogue_cycle_with_dependencies,
    )

    transcript = service.run(request=request, dependencies=deps)

    assert transcript == "transcript"
    assert "common" in calls
    assert "request_kwargs" in calls
    assert calls.get("deps_built") is True
    assert "runner" in calls


def test_dialogue_cycle_node_execution_service_passes_model_and_mmproj() -> None:
    service = DialogueCycleNodeExecutionService()
    observed: dict[str, object] = {}

    request = DialogueCycleNodeExecutionRequest(
        initial_user_text="hello",
        session_id="sid",
        cycles=1,
        modelA="ma",
        mmprojA="a.mmproj",
        modelB="mb",
        mmprojB="b.mmproj",
        system_prompt="sys",
        system_prompt_A="",
        system_prompt_B="",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        n_gpu_layers=0,
        n_ctx=1024,
        max_turns=12,
        summarize_old_history=True,
        summary_chunk_turns=3,
        max_tokens_summary=128,
        summary_max_chars=1500,
        dynamic_max_tokens=True,
        min_generation_tokens=96,
        safety_margin_tokens=64,
        persistent_cache="off",
        runtime_cache="off",
        repeat_penalty=1.12,
        repeat_last_n=256,
        rewrite_continue=True,
        log_level="timing",
        suppress_backend_logs=True,
        history_dir="",
        reset_session=False,
        stream_to_console=False,
        chat_handler_overrides=None,
        text_chat_builder_overrides=None,
    )

    deps = DialogueCycleNodeExecutionDependencies(
        build_common_turn_kwargs=lambda **_kwargs: {},
        build_dialogue_cycle_request=lambda **kwargs: (
            observed.update(kwargs)
            or DialogueCycleRequest(
                initial_user_text="hello",
                session_id="sid",
                cycles=1,
                system_prompt="sys",
                system_prompt_A="",
                system_prompt_B="",
                runtime_cache="off",
                stream_to_console=False,
                reset_session=False,
                history_dir="",
                turn_kwargs_A={},
                turn_kwargs_B={},
            )
        ),
        build_dialogue_cycle_dependencies=lambda: DialogueCycleDependencies(
            now_iso=lambda: "now",
            transcript_path=lambda _sid, _hist: "t.txt",
            append_transcript_lines=lambda _p, _l: None,
            clear_kv_state_for_session=lambda _sid: None,
            get_or_create_model_manager=lambda _role: object(),
            unload_model=lambda _m: None,
            chat_one_turn=lambda **_kwargs: "ok",
        ),
        run_dialogue_cycle_with_dependencies=lambda **_kwargs: "ok",
    )

    service.run(request=request, dependencies=deps)

    assert observed["model_a"] == "ma"
    assert observed["mmproj_a"] == "a.mmproj"
    assert observed["model_b"] == "mb"
    assert observed["mmproj_b"] == "b.mmproj"


def test_run_dialogue_cycle_llama_trie_cache_skips_all_unload() -> None:
    service = ChatTurnService()
    unload_calls: list[str] = []

    managers = [DummyManager("A"), DummyManager("B")]

    def _factory():
        return managers.pop(0)

    service.run_dialogue_cycle(
        initial_user_text="hello",
        session_id="sid",
        cycles=1,
        system_prompt="sys",
        system_prompt_A="",
        system_prompt_B="",
        runtime_cache="LlamaTrieCache",
        stream_to_console=False,
        reset_session=False,
        history_dir="",
        now_iso=lambda: "2026-03-21T12:00:00+09:00",
        transcript_path=lambda _sid, _history_dir: "transcript.txt",
        append_transcript_lines=lambda _path, _lines: None,
        clear_kv_state_for_session=lambda _sid: None,
        get_or_create_model_manager=lambda role: _factory(),
        unload_model=lambda manager: unload_calls.append(manager.name),
        chat_one_turn=lambda **_kwargs: "ok",
        turn_kwargs_A={"model": "ma", "mmproj": "(Auto detect)"},
        turn_kwargs_B={"model": "mb", "mmproj": "(Auto detect)"},
    )

    assert unload_calls.count("A") == 0
    assert unload_calls.count("B") == 0

