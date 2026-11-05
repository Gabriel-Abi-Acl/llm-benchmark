"""Executa o conjunto v1: resposta, latência, token, custo e nota.

O runner do placeholder (`runner.run_eval`) continua sem nota. Este módulo
é o que grava o JSONL pontuado. Cache e retry ficam só neste caminho.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

from llm_benchmark.cache import CachedResponse, load_cached, request_key, store_cached
from llm_benchmark.errors import DatasetError
from llm_benchmark.eval_dataset import EvalTask
from llm_benchmark.metrics import SCORE_METHODS, LLMJudge, Score, SupportsJudge, score_answer
from llm_benchmark.pricing import ModelPrice, estimate_cost_usd, format_usd
from llm_benchmark.retry import AttemptResult, call_with_retry
from llm_benchmark.types import LLMRequest, LLMResponse, TokenUsage

RECORD_FIELDS = (
    "task_id",
    "type",
    "provider",
    "model",
    "answer",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "estimated_cost_usd",
    "error",
    "cached",
    "abstained",
    "correct",
    "score_method",
    "overconfident",
)


@dataclass(frozen=True)
class EvalRecord:
    """Uma observação pontuada. `correct` nulo é erro de transporte, fora da acurácia."""

    task_id: str
    task_type: str
    provider: str
    model: str
    answer: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: str | None
    error: str | None
    cached: bool
    abstained: bool
    correct: bool | None
    score_method: str | None
    overconfident: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "type": self.task_type,
            "provider": self.provider,
            "model": self.model,
            "answer": self.answer,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "error": self.error,
            "cached": self.cached,
            "abstained": self.abstained,
            "correct": self.correct,
            "score_method": self.score_method,
            "overconfident": self.overconfident,
        }


class SupportsComplete(Protocol):
    """Estrutural: o objeto precisa de nome, modelo e `complete`."""

    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def run_evaluation(
    provider: SupportsComplete,
    tasks: Sequence[EvalTask],
    *,
    temperature: float = 0.0,
    clock: Callable[[], float] | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    cache_dir: Path | None = None,
    max_attempts: int = 5,
    base_delay_s: float = 1.0,
    sleep: Callable[[float], None] | None = None,
    judge: SupportsJudge | None = None,
    judge_provider: SupportsComplete | None = None,
) -> list[EvalRecord]:
    """Roda cada tarefa em ordem. Falha de uma não cancela as outras.

    `judge_provider`, se vier, recebe o mesmo retry e o mesmo diretório de cache,
    num namespace separado. O custo gravado é só o da resposta candidata.
    """
    now = clock or time.perf_counter
    pause = sleep or time.sleep
    records: list[EvalRecord] = []
    active_judge = judge
    if active_judge is None and judge_provider is not None:
        active_judge = LLMJudge(
            _CachedComplete(
                judge_provider,
                namespace="judge",
                cache_dir=cache_dir,
                clock=now,
                max_attempts=max_attempts,
                base_delay_s=base_delay_s,
                sleep=pause,
            )
        )
    for task in tasks:
        request = LLMRequest(prompt=task.prompt, temperature=temperature)
        fetched, cached = _fetch(
            provider,
            request,
            namespace="task",
            cache_dir=cache_dir,
            clock=now,
            max_attempts=max_attempts,
            base_delay_s=base_delay_s,
            sleep=pause,
        )
        records.append(
            _record_from_fetch(
                task,
                provider,
                fetched,
                cached=cached,
                prices=prices,
                judge=active_judge,
            )
        )
    return records


def write_records(path: Path, records: Sequence[EvalRecord]) -> None:
    """JSONL UTF-8, uma observação por linha, terminando em newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_records(path: Path) -> list[EvalRecord]:
    """Lê o JSONL pontuado. Não reabre o gabarito e não chama provedor."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(f"Results not found: {path}") from exc
    except OSError as exc:
        raise DatasetError(f"Results not readable: {path}") from exc
    if raw_text.strip() == "":
        raise DatasetError(f"Results have no observations: {path}")
    records: list[EvalRecord] = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if line.strip() == "":
            raise DatasetError(f"Line {line_no}: blank line")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Line {line_no}: not valid JSON") from exc
        records.append(record_from_dict(raw, line_no=line_no))
    return records


def record_from_dict(raw: object, *, line_no: int) -> EvalRecord:
    """Valida uma linha já pontuada. Campo a mais é erro: a tabela não ignora typo."""
    if not isinstance(raw, dict):
        raise DatasetError(f"Line {line_no}: observation is not an object")
    extra = sorted(set(raw) - set(RECORD_FIELDS))
    if extra:
        raise DatasetError(f"Line {line_no}: unknown fields: {', '.join(extra)}")
    missing = [key for key in RECORD_FIELDS if key not in raw]
    if missing:
        raise DatasetError(f"Line {line_no}: missing {', '.join(missing)}")

    def fail(message: str) -> DatasetError:
        return DatasetError(f"Line {line_no}: {message}")

    parsed_method = _score_method(raw["score_method"], fail)
    cost = raw["estimated_cost_usd"]
    if cost is not None:
        cost = _cost_string(cost, fail)
    return EvalRecord(
        task_id=_required_text(raw["task_id"], "task_id", fail),
        task_type=_required_text(raw["type"], "type", fail),
        provider=_required_text(raw["provider"], "provider", fail),
        model=_required_text(raw["model"], "model", fail),
        answer=_optional_text(raw["answer"], "answer", fail),
        latency_ms=_float(raw["latency_ms"], "latency_ms", fail),
        input_tokens=_optional_int(raw["input_tokens"], "input_tokens", fail),
        output_tokens=_optional_int(raw["output_tokens"], "output_tokens", fail),
        estimated_cost_usd=cost,
        error=_optional_text(raw["error"], "error", fail),
        cached=_flag(raw["cached"], "cached", fail),
        abstained=_flag(raw["abstained"], "abstained", fail),
        correct=_optional_flag(raw["correct"], "correct", fail),
        score_method=parsed_method,
        overconfident=_flag(raw["overconfident"], "overconfident", fail),
    )


class _CachedComplete:
    """`complete` com retry e cache, para o juiz usar a mesma política da tarefa."""

    def __init__(
        self,
        provider: SupportsComplete,
        *,
        namespace: str,
        cache_dir: Path | None,
        clock: Callable[[], float],
        max_attempts: int,
        base_delay_s: float,
        sleep: Callable[[float], None],
    ) -> None:
        self._provider = provider
        self._namespace = namespace
        self._cache_dir = cache_dir
        self._clock = clock
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._sleep = sleep

    def complete(self, request: LLMRequest) -> LLMResponse:
        fetched, _ = _fetch(
            self._provider,
            request,
            namespace=self._namespace,
            cache_dir=self._cache_dir,
            clock=self._clock,
            max_attempts=self._max_attempts,
            base_delay_s=self._base_delay_s,
            sleep=self._sleep,
        )
        if fetched.error is not None or fetched.response is None:
            raise fetched.error or RuntimeError("judge call failed")
        return fetched.response


def _fetch(
    provider: SupportsComplete,
    request: LLMRequest,
    *,
    namespace: str,
    cache_dir: Path | None,
    clock: Callable[[], float],
    max_attempts: int,
    base_delay_s: float,
    sleep: Callable[[float], None],
) -> tuple[AttemptResult, bool]:
    key = request_key(
        provider=provider.name,
        model=provider.model,
        request=request,
        namespace=namespace,
    )
    if cache_dir is not None:
        cached = load_cached(cache_dir, key)
        if cached is not None:
            response = LLMResponse(
                text=cached.text,
                model=cached.model,
                provider=cached.provider,
                usage=TokenUsage(
                    input_tokens=cached.input_tokens,
                    output_tokens=cached.output_tokens,
                ),
            )
            return AttemptResult(response=response, latency_ms=cached.latency_ms, error=None), True
    result = call_with_retry(
        provider.complete,
        request,
        clock=clock,
        max_attempts=max_attempts,
        base_delay_s=base_delay_s,
        sleep=sleep,
    )
    if result.response is not None and cache_dir is not None:
        _store_quietly(cache_dir, key, result)
    return result, False


def _store_quietly(cache_dir: Path, key: dict[str, object], result: AttemptResult) -> None:
    # Falha de disco não apaga a resposta que o provedor já deu.
    response = result.response
    if response is None:
        return
    try:
        store_cached(
            cache_dir,
            key,
            CachedResponse(
                text=response.text,
                model=response.model,
                provider=response.provider,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=result.latency_ms,
            ),
        )
    except OSError:
        return


def _record_from_fetch(
    task: EvalTask,
    provider: SupportsComplete,
    fetched: AttemptResult,
    *,
    cached: bool,
    prices: Mapping[str, ModelPrice] | None,
    judge: SupportsJudge | None,
) -> EvalRecord:
    if fetched.error is not None or fetched.response is None:
        answer = None
        provider_name = provider.name
        model = provider.model
        input_tokens = None
        output_tokens = None
        cost = None
        error = _error_text(fetched.error)
    else:
        response = fetched.response
        answer = response.text
        provider_name = response.provider
        model = response.model
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        estimated = estimate_cost_usd(
            response.model,
            input_tokens,
            output_tokens,
            provider=response.provider,
            prices=prices,
        )
        cost = None if estimated is None else format_usd(estimated)
        error = None
    score = score_answer(task, answer, error=error, judge=judge)
    return _with_score(
        EvalRecord(
            task_id=task.id,
            task_type=task.type,
            provider=provider_name,
            model=model,
            answer=answer,
            latency_ms=fetched.latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            error=error,
            cached=cached,
            abstained=False,
            correct=None,
            score_method=None,
            overconfident=False,
        ),
        score,
    )


def _with_score(record: EvalRecord, score: Score) -> EvalRecord:
    return replace(
        record,
        abstained=score.abstained,
        correct=score.correct,
        score_method=score.score_method,
        overconfident=score.overconfident,
    )


def _error_text(exc: Exception | None) -> str:
    if exc is None:
        return "RuntimeError: missing response"
    return f"{type(exc).__name__}: {exc}"


def _required_text(value: object, label: str, fail: Callable[[str], DatasetError]) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise fail(f"invalid {label}")
    return value


def _optional_text(value: object, label: str, fail: Callable[[str], DatasetError]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise fail(f"invalid {label}")
    return value


def _optional_int(value: object, label: str, fail: Callable[[str], DatasetError]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise fail(f"invalid {label}")
    return value


def _float(value: object, label: str, fail: Callable[[str], DatasetError]) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise fail(f"invalid {label}")
    return float(value)


def _flag(value: object, label: str, fail: Callable[[str], DatasetError]) -> bool:
    if not isinstance(value, bool):
        raise fail(f"invalid {label}")
    return value


def _optional_flag(value: object, label: str, fail: Callable[[str], DatasetError]) -> bool | None:
    if value is None:
        return None
    return _flag(value, label, fail)


def _score_method(value: object, fail: Callable[[str], DatasetError]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in SCORE_METHODS:
        raise fail(f"invalid score_method: {value!r}")
    return value


def _cost_string(value: object, fail: Callable[[str], DatasetError]) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise fail("invalid estimated_cost_usd")
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise fail("invalid estimated_cost_usd") from exc
    if number < 0:
        raise fail("invalid estimated_cost_usd")
    return value.strip()
