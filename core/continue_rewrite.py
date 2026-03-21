# Continue-input rewrite helper with language-aware prompt mapping.
from __future__ import annotations

import re
from typing import Any, Callable, Dict

try:
    from .turn_types import ContinueRewriteResult
except Exception:
    from core.turn_types import ContinueRewriteResult


_CONTINUE_RE = re.compile(r"^\s*continue\b", flags=re.IGNORECASE)

_CONTINUE_PROMPTS: Dict[str, str] = {
    "zh": (
        "从最后一个完整的句子继续。"
        "不要重复之前的短语、列表、标题或段落。"
        "如果你开始重复词语或地名，立即停止并写一个一句话摘要。"
    ),
    "ja": (
        "最後の完全な文から続けてください。"
        "以前のフレーズ、リスト、見出し、セクションを繰り返さないでください。"
        "単語や地名を繰り返し始めたら、すぐに停止して1文の要約を書いてください。"
    ),
    "en": (
        "Continue from the last complete sentence. "
        "Do not repeat previous phrases, lists, headings, or sections. "
        "If you start repeating words or place names, stop immediately and write a 1-sentence summary instead."
    ),
}

_FALLBACK_CONTINUE_PROMPT = "Continue from the last complete sentence. Do not repeat previous content."


def rewrite_continue_prompt(
    *,
    user_text: str,
    history: Dict[str, Any],
    rewrite_continue: bool,
    detect_history_language: Callable[[Dict[str, Any]], str],
) -> ContinueRewriteResult:
    raw_user_text = user_text or ""
    is_continue = bool(_CONTINUE_RE.match(raw_user_text))
    if not rewrite_continue or not is_continue:
        return ContinueRewriteResult(user_text_for_model=raw_user_text)

    detected_lang = detect_history_language(history)
    rewritten_text = _CONTINUE_PROMPTS.get(detected_lang, _FALLBACK_CONTINUE_PROMPT)
    return ContinueRewriteResult(
        user_text_for_model=rewritten_text,
        detected_language=detected_lang,
        rewritten=True,
    )

