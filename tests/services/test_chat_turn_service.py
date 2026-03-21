from __future__ import annotations

from services.chat_turn_service import ChatTurnService, DialogueCycleDependencies, DialogueCycleRequest


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
        model_manager_factory=_factory,
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
        model_manager_factory=_factory,
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
        model_manager_factory=_factory,
        unload_model=lambda manager: unload_calls.append(manager.name),
        chat_one_turn=lambda **_kwargs: "ok",
        turn_kwargs_A={"model": "ma", "mmproj": "(Auto detect)"},
        turn_kwargs_B={"model": "mb", "mmproj": "(Auto detect)"},
    )

    # KV cache keeps managers loaded in-loop; only final cleanup unloads once each.
    assert unload_calls.count("A") == 1
    assert unload_calls.count("B") == 1


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
        model_manager_factory=_factory,
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
