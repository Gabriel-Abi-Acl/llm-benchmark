"""Repete chamada só em rate limit.

Erro de esquema, credencial ou resposta sem token não é transitório:
repetir não muda o resultado e gasta cota.
"""

from collections.abc import Callable
from dataclasses import dataclass

from llm_benchmark.types import LLMRequest, LLMResponse

_RATE_PHRASES = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "resource exhausted",
    "quota exceeded",
)


@dataclass(frozen=True)
class AttemptResult:
    """Resultado da última tentativa. `error` preenchido significa que não houve resposta."""

    response: LLMResponse | None
    latency_ms: float
    error: Exception | None


def is_rate_limit_error(exc: BaseException) -> bool:
    """429, nome da classe ou frase conhecida. Outro erro não espera."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status == 429:
        return True
    name = type(exc).__name__.lower().replace("_", "")
    if "ratelimit" in name:
        return True
    text = str(exc).lower()
    return any(phrase in text for phrase in _RATE_PHRASES)


def backoff_seconds(failed_attempt: int, base_delay_s: float, exc: BaseException) -> float:
    """Espera depois da falha `failed_attempt` (1 = primeira). Retry-After ganha se for maior."""
    if failed_attempt < 1:
        raise ValueError("failed_attempt must be >= 1")
    if base_delay_s < 0:
        raise ValueError("base_delay_s must be >= 0")
    delay = base_delay_s * (2 ** (failed_attempt - 1))
    retry_after = _retry_after_seconds(exc)
    if retry_after is None:
        return delay
    return max(delay, retry_after)


def call_with_retry(
    complete: Callable[[LLMRequest], LLMResponse],
    request: LLMRequest,
    *,
    clock: Callable[[], float],
    max_attempts: int,
    base_delay_s: float,
    sleep: Callable[[float], None],
) -> AttemptResult:
    """Tenta `complete` até `max_attempts`. A latência é só a da última tentativa."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if base_delay_s < 0:
        raise ValueError("base_delay_s must be >= 0")
    last_latency = 0.0
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        start = clock()
        try:
            response = complete(request)
        except Exception as exc:
            last_latency = (clock() - start) * 1000
            last_error = exc
            if not is_rate_limit_error(exc) or attempt == max_attempts:
                return AttemptResult(response=None, latency_ms=last_latency, error=exc)
            sleep(backoff_seconds(attempt, base_delay_s, exc))
            continue
        return AttemptResult(
            response=response,
            latency_ms=(clock() - start) * 1000,
            error=None,
        )
    return AttemptResult(response=None, latency_ms=last_latency, error=last_error)


def _retry_after_seconds(exc: BaseException) -> float | None:
    for attr in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, attr, None)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if value >= 0:
            return float(value)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = None
    getter = getattr(headers, "get", None)
    if callable(getter):
        raw = getter("retry-after")
        if raw is None:
            raw = getter("Retry-After")
    if isinstance(raw, bool) or not isinstance(raw, int | float | str):
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed
