"""
validator_retry.py — retry wrapper for validator LLM calls that occasionally
return malformed JSON.

THE PROBLEM THIS FILE ADDRESSES
--------------------------------
Every validator step in the pipeline (PRE, goal-baseline, POST, and the
independent evidence-review pass) follows the same pattern: call the LLM
once, parse the raw response as JSON, then run it through
`normalize_validation_result`, which enforces a strict schema (must be a JSON
object, must have a "results" list of the expected length, condition text
must match verbatim, status/progress_trend must be one of the allowed
values, etc.). Any violation of that schema raises a ValueError, and there is
currently no retry: a single malformed response ends the whole validation
loop with "[ERROR][validation_loop] ...".

This was observed directly: gemini-robotics-er-1.6-preview returned a
response whose top-level JSON was not an object on a PRE validator call, and
the entire run stopped instead of simply re-asking the model.

THE FIX
-------
A generic retry wrapper: call the model again (up to `max_retries` extra
times) whenever the parse-or-normalize step raises, and only propagate the
error once every attempt has failed. Nothing about prompts, schemas, or
normalization changes -- this only adds persistence around a transient,
already-observed failure mode.
"""

from __future__ import annotations

from typing import Any, Callable

from src.utils import try_parse_json


def call_and_normalize_with_retry(
    *,
    call_llm_fn: Callable[[], dict[str, Any]],
    normalize_fn: Callable[[Any, dict[str, Any]], dict[str, Any]],
    label: str,
    max_retries: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Call `call_llm_fn()` (a zero-arg callable wrapping call_llm_completion),
    parse its raw_response as JSON, and run it through
    `normalize_fn(parsed_response, llm_result)`. If parsing or normalization
    raises, retry with a fresh LLM call up to `max_retries` additional times
    before re-raising the last error.

    Returns (normalized_response, llm_result) from the first successful
    attempt.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 2):
        result = call_llm_fn()
        parse_ok, parsed_response = try_parse_json(result["raw_response"])

        if not parse_ok:
            last_error = ValueError(
                f"[{label}] Model response could not be parsed as valid "
                f"JSON.\n\nRaw response:\n{result['raw_response']}"
            )
        else:
            try:
                normalized = normalize_fn(parsed_response, result)
            except ValueError as exc:
                last_error = exc
            else:
                if attempt > 1:
                    print(
                        f"[OK][{label}] Malformed response recovered on "
                        f"retry {attempt - 1}."
                    )
                return normalized, result

        if attempt <= max_retries:
            print(
                f"[WARN][{label}] Attempt {attempt} produced a malformed "
                f"validator response ({last_error}); retrying "
                f"({max_retries - attempt + 1} attempt(s) left)."
            )

    assert last_error is not None
    raise last_error
