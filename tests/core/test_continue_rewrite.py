from __future__ import annotations

from core.continue_rewrite import rewrite_continue_prompt


def _detector(lang: str):
    def _inner(_history):
        return lang

    return _inner


def test_rewrite_disabled_keeps_original_text() -> None:
    result = rewrite_continue_prompt(
        user_text="continue please",
        history={},
        rewrite_continue=False,
        detect_history_language=_detector("ja"),
    )
    assert result.user_text_for_model == "continue please"
    assert result.rewritten is False
    assert result.detected_language is None


def test_non_continue_text_keeps_original() -> None:
    result = rewrite_continue_prompt(
        user_text="hello",
        history={},
        rewrite_continue=True,
        detect_history_language=_detector("en"),
    )
    assert result.user_text_for_model == "hello"
    assert result.rewritten is False


def test_continue_rewrites_for_ja() -> None:
    result = rewrite_continue_prompt(
        user_text="  Continue",
        history={"turns": []},
        rewrite_continue=True,
        detect_history_language=_detector("ja"),
    )
    assert result.rewritten is True
    assert result.detected_language == "ja"
    assert "最後の完全な文から続けてください" in result.user_text_for_model


def test_continue_rewrites_for_unknown_language_uses_fallback() -> None:
    result = rewrite_continue_prompt(
        user_text="continue",
        history={},
        rewrite_continue=True,
        detect_history_language=_detector("fr"),
    )
    assert result.rewritten is True
    assert result.detected_language == "fr"
    assert result.user_text_for_model == "Continue from the last complete sentence. Do not repeat previous content."
