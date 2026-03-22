# Service for generation execution orchestration and output normalization.
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass(frozen=True)
class GenerationExecutionOutcome:
    generation_result: Any
    assistant_text: str
    failed: bool
    error: Optional[Exception]


class GenerationExecutionService:
    def _dep(self, deps: Mapping[str, Any], key: str) -> Any:
        if key not in deps:
            raise KeyError(f"Missing dependency: {key}")
        return deps[key]

    @staticmethod
    def _is_ctx_error(err: Exception) -> bool:
        s = str(err)
        return ("exceeds n_ctx" in s) or ("Prompt exceeds n_ctx" in s) or ("n_ctx" in s and "exceed" in s)

    def _build_attempt_logger(self, request: Any) -> Optional[Callable[..., None]]:
        if not request.enable_attempt_logging:
            return None

        def _attempt_logger(
            ok: bool,
            attempt_no: int,
            elapsed: float,
            gen_tok: int,
            turns: Optional[int],
            err: Optional[Exception],
        ) -> None:
            if ok:
                print(
                    f"{request.log_prefix} Generation attempt {attempt_no} succeeded in {elapsed:.2f} seconds "
                    f"(max_tokens={int(gen_tok)}, turns_limit={turns})"
                )
                return
            print(
                f"{request.log_prefix} Generation attempt {attempt_no} failed in {elapsed:.2f} seconds "
                f"(max_tokens={int(gen_tok)}, turns_limit={turns}): {err}"
            )

        return _attempt_logger

    def execute(
        self,
        *,
        request: Any,
        deps: Mapping[str, Any],
        llm: Any,
        messages: Any,
        text_chat_request: Any,
        is_state_data_mismatch_error: Callable[[Exception], bool],
        on_state_cache_mismatch: Callable[[Exception], None],
        on_compact_summary: Callable[[], None],
        rebuild_messages_for_turns_limit: Callable[[Optional[int]], tuple[Any, Any]],
    ) -> GenerationExecutionOutcome:
        run_generation_with_adaptive_retry = self._dep(deps, "run_generation_with_adaptive_retry")
        make_suppress_backend_logs = self._dep(deps, "make_suppress_backend_logs")
        attempt_logger = self._build_attempt_logger(request)

        generation_result = run_generation_with_adaptive_retry(
            llm=llm,
            messages=messages,
            text_chat_request=text_chat_request,
            max_tokens=int(request.max_tokens),
            temperature=float(request.temperature),
            top_p=float(request.top_p),
            repeat_penalty=float(request.repeat_penalty),
            repeat_last_n=int(request.repeat_last_n),
            dynamic_max_tokens=bool(request.dynamic_max_tokens),
            min_generation_tokens=int(request.min_generation_tokens),
            safety_margin_tokens=int(request.safety_margin_tokens),
            initial_turns_limit=(int(request.max_turns) if request.max_turns is not None else None),
            stream_to_console=bool(request.stream_to_console),
            max_attempts=6,
            is_ctx_error=self._is_ctx_error,
            is_state_data_mismatch_error=is_state_data_mismatch_error,
            on_state_cache_mismatch=on_state_cache_mismatch,
            on_compact_summary=on_compact_summary,
            rebuild_messages_for_turns_limit=rebuild_messages_for_turns_limit,
            attempt_logger=attempt_logger,
            debug_traceback=(request.log_level == "debug"),
            traceback_print_exc=traceback.print_exc,
            suppress_backend_logs_ctx_factory=lambda: make_suppress_backend_logs(
                bool(request.suppress_backend_logs) and (request.log_level != "debug")
            ),
            iter_chat_completion_robust=self._dep(deps, "iter_chat_completion_robust"),
            create_chat_completion_robust=self._dep(deps, "create_chat_completion_robust"),
            extract_stream_content=self._dep(deps, "extract_stream_content"),
            retry_kwargs_with_repeat_last_n_fallback=self._dep(deps, "retry_kwargs_with_repeat_last_n_fallback"),
        )

        assistant_text = generation_result.assistant_text
        if request.strip_assistant_before_reasoning_filter:
            assistant_text = assistant_text.strip()

        if generation_result.non_ctx_error:
            return GenerationExecutionOutcome(
                generation_result=generation_result,
                assistant_text="",
                failed=True,
                error=generation_result.last_err,
            )

        strip_reasoning_output = self._dep(deps, "strip_reasoning_output")
        assistant_text = strip_reasoning_output(assistant_text)
        if not assistant_text:
            return GenerationExecutionOutcome(
                generation_result=generation_result,
                assistant_text="",
                failed=True,
                error=generation_result.last_err,
            )

        return GenerationExecutionOutcome(
            generation_result=generation_result,
            assistant_text=assistant_text,
            failed=False,
            error=None,
        )