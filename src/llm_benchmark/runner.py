"""Roda o conjunto e devolve um registro por tarefa.

Latência é o relógio do cliente em volta de `complete`. Falha de uma tarefa
não interrompe as outras: o erro entra no registro, sem nota de qualidade.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from llm_benchmark.dataset import Task
from llm_benchmark.pricing import estimate_cost_usd, format_usd
from llm_benchmark.types import LLMRequest, LLMResponse


class SupportsComplete(Protocol):
    """O mínimo que o runner exige. Adaptador real e falso cumprem isso."""

    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


@dataclass(frozen=True)
class RunRecord:
    """Uma observação. Não há campo de nota: pontuar é o PR3."""

    task_id: str
    category: str
    language: str
    provider: str
    model: str
    answer: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: str | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_eval(
    provider: SupportsComplete,
    tasks: Sequence[Task],
    *,
    temperature: float = 0.0,
    clock: Callable[[], float] | None = None,
) -> list[RunRecord]:
    """Executa cada tarefa em ordem. `clock` injetável deixa a latência determinística no teste."""
    now = clock or time.perf_counter
    records: list[RunRecord] = []
    for task in tasks:
        start = now()
        try:
            response = provider.complete(LLMRequest(prompt=task.prompt, temperature=temperature))
        except Exception as exc:
            records.append(
                _error_record(task, provider, latency_ms=(now() - start) * 1000, exc=exc)
            )
            continue
        latency_ms = (now() - start) * 1000
        records.append(_success_record(task, response, latency_ms))
    return records


def _success_record(task: Task, response: LLMResponse, latency_ms: float) -> RunRecord:
    cost = estimate_cost_usd(
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        provider=response.provider,
    )
    return RunRecord(
        task_id=task.id,
        category=task.category,
        language=task.language,
        provider=response.provider,
        model=response.model,
        answer=response.text,
        latency_ms=latency_ms,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        estimated_cost_usd=None if cost is None else format_usd(cost),
        error=None,
    )


def _error_record(
    task: Task, provider: SupportsComplete, *, latency_ms: float, exc: Exception
) -> RunRecord:
    return RunRecord(
        task_id=task.id,
        category=task.category,
        language=task.language,
        provider=provider.name,
        model=provider.model,
        answer=None,
        latency_ms=latency_ms,
        input_tokens=None,
        output_tokens=None,
        estimated_cost_usd=None,
        error=f"{type(exc).__name__}: {exc}",
    )
