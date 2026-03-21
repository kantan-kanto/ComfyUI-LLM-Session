# Shared generation execution + adaptive retry loop with fallback handling.
from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

try:
    from .turn_types import GenerationRunResult
except Exception:
    from core.turn_types import GenerationRunResult


T = TypeVar("T")


def run_with_typeerror_fallback(
    *,
    execute_with_kwargs: Callable[[Dict[str, Any]], T],
    completion_kwargs: Dict[str, Any],
    retry_kwargs_with_repeat_last_n_fallback: Callable[[Dict[str, Any], int], Dict[str, Any]],
    repeat_last_n: int,
) -> T:
    """Retry execution with fallback kwargs on TypeError (max 3 attempts total)."""
    active_kwargs = dict(completion_kwargs)
    try:
        return execute_with_kwargs(active_kwargs)
    except TypeError:
        active_kwargs = retry_kwargs_with_repeat_last_n_fallback(active_kwargs, repeat_last_n)
        try:
            return execute_with_kwargs(active_kwargs)
        except TypeError:
            active_kwargs = retry_kwargs_with_repeat_last_n_fallback(active_kwargs, repeat_last_n)
            return execute_with_kwargs(active_kwargs)


def _run_generation_once(
    *,
    llm: Any,
    messages: List[Dict[str, Any]],
    text_chat_request: Optional[Dict[str, Any]],
    completion_kwargs: Dict[str, Any],
    stream_to_console: bool,
    suppress_backend_logs_ctx,
    iter_chat_completion_robust: Callable[..., Any],
    create_chat_completion_robust: Callable[..., Dict[str, Any]],
    extract_stream_content: Callable[[Any], str],
    retry_kwargs_with_repeat_last_n_fallback: Callable[[Dict[str, Any], int], Dict[str, Any]],
    repeat_last_n: int,
) -> str:
    def _stream_execute(active_kwargs: Dict[str, Any]):
        if text_chat_request is not None:
            return llm.create_completion(
                prompt=text_chat_request["prompt"],
                stop=text_chat_request["stop"],
                stream=True,
                **active_kwargs,
            )
        return iter_chat_completion_robust(llm, messages, **active_kwargs)

    def _non_stream_execute(active_kwargs: Dict[str, Any]):
        if text_chat_request is not None:
            return llm.create_completion(
                prompt=text_chat_request["prompt"],
                stop=text_chat_request["stop"],
                **active_kwargs,
            )
        return create_chat_completion_robust(llm, messages, **active_kwargs)

    with suppress_backend_logs_ctx:
        if stream_to_console:
            pieces: List[str] = []
            out = sys.__stdout__
            stream_iter = run_with_typeerror_fallback(
                execute_with_kwargs=_stream_execute,
                completion_kwargs=completion_kwargs,
                retry_kwargs_with_repeat_last_n_fallback=retry_kwargs_with_repeat_last_n_fallback,
                repeat_last_n=int(repeat_last_n),
            )

            for chunk in stream_iter:
                token = extract_stream_content(chunk)
                if not token:
                    continue
                pieces.append(token)
                try:
                    out.write(token)
                    out.flush()
                except Exception:
                    pass
            return "".join(pieces)

        resp = run_with_typeerror_fallback(
            execute_with_kwargs=_non_stream_execute,
            completion_kwargs=completion_kwargs,
            retry_kwargs_with_repeat_last_n_fallback=retry_kwargs_with_repeat_last_n_fallback,
            repeat_last_n=int(repeat_last_n),
        )

        if text_chat_request is not None:
            return resp["choices"][0]["text"] or ""
        return resp["choices"][0]["message"]["content"] or ""


def run_generation_with_adaptive_retry(
    *,
    llm: Any,
    messages: List[Dict[str, Any]],
    text_chat_request: Optional[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    repeat_last_n: int,
    dynamic_max_tokens: bool,
    min_generation_tokens: int,
    safety_margin_tokens: int,
    initial_turns_limit: Optional[int],
    stream_to_console: bool,
    max_attempts: int,
    is_ctx_error: Callable[[Exception], bool],
    is_state_data_mismatch_error: Callable[[Exception], bool],
    on_state_cache_mismatch: Callable[[Exception], None],
    on_compact_summary: Callable[[], None],
    rebuild_messages_for_turns_limit: Callable[[Optional[int]], Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]],
    attempt_logger: Optional[Callable[[bool, int, float, int, Optional[int], Optional[Exception]], None]],
    debug_traceback: bool,
    traceback_print_exc: Callable[[], None],
    suppress_backend_logs_ctx_factory: Callable[[], Any],
    iter_chat_completion_robust: Callable[..., Any],
    create_chat_completion_robust: Callable[..., Dict[str, Any]],
    extract_stream_content: Callable[[Any], str],
    retry_kwargs_with_repeat_last_n_fallback: Callable[[Dict[str, Any], int], Dict[str, Any]],
) -> GenerationRunResult:
    assistant_text = ""
    gen_tokens = int(max_tokens)
    if bool(dynamic_max_tokens):
        gen_tokens = max(1, gen_tokens - max(0, int(safety_margin_tokens)))
    turns_limit = initial_turns_limit

    attempts = 0
    last_err: Optional[Exception] = None
    state_cache_recovered = False

    while attempts < int(max_attempts):
        attempt_no = attempts + 1
        t_start = time.perf_counter()
        try:
            completion_kwargs: Dict[str, Any] = {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(gen_tokens),
            }
            if repeat_last_n and int(repeat_last_n) > 0:
                completion_kwargs["penalty_last_n"] = int(repeat_last_n)
            if repeat_penalty and float(repeat_penalty) != 1.0:
                completion_kwargs["repeat_penalty"] = float(repeat_penalty)

            assistant_text = _run_generation_once(
                llm=llm,
                messages=messages,
                text_chat_request=text_chat_request,
                completion_kwargs=completion_kwargs,
                stream_to_console=bool(stream_to_console),
                suppress_backend_logs_ctx=suppress_backend_logs_ctx_factory(),
                iter_chat_completion_robust=iter_chat_completion_robust,
                create_chat_completion_robust=create_chat_completion_robust,
                extract_stream_content=extract_stream_content,
                retry_kwargs_with_repeat_last_n_fallback=retry_kwargs_with_repeat_last_n_fallback,
                repeat_last_n=int(repeat_last_n),
            )

            if attempt_logger is not None:
                attempt_logger(True, attempt_no, time.perf_counter() - t_start, int(gen_tokens), turns_limit, None)

            return GenerationRunResult(
                assistant_text=assistant_text,
                gen_tokens=int(gen_tokens),
                turns_limit=turns_limit,
                last_err=last_err,
                succeeded=True,
                non_ctx_error=False,
            )
        except Exception as err:
            last_err = err
            if attempt_logger is not None:
                attempt_logger(False, attempt_no, time.perf_counter() - t_start, int(gen_tokens), turns_limit, err)
            if debug_traceback:
                traceback_print_exc()

            if is_state_data_mismatch_error(err) and not state_cache_recovered:
                state_cache_recovered = True
                on_state_cache_mismatch(err)
                attempts += 1
                continue

            if dynamic_max_tokens and is_ctx_error(err):
                if gen_tokens > int(min_generation_tokens):
                    gen_tokens = max(int(min_generation_tokens), gen_tokens // 2)
                else:
                    if turns_limit is not None and turns_limit > 0:
                        turns_limit = max(0, turns_limit - 1)
                    else:
                        on_compact_summary()

                messages, text_chat_request = rebuild_messages_for_turns_limit(turns_limit)
                attempts += 1
                continue

            return GenerationRunResult(
                assistant_text="",
                gen_tokens=int(gen_tokens),
                turns_limit=turns_limit,
                last_err=last_err,
                succeeded=False,
                non_ctx_error=True,
            )

    return GenerationRunResult(
        assistant_text=assistant_text,
        gen_tokens=int(gen_tokens),
        turns_limit=turns_limit,
        last_err=last_err,
        succeeded=bool(assistant_text),
        non_ctx_error=False,
    )
