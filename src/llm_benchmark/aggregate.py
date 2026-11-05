"""Soma observações já pontuadas em tabela por provedor e por tipo.

A linha do provedor junta todas as famílias. A linha de tipo é a leitura
por família. Denominador zero fica célula vazia, não zero.
"""

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from llm_benchmark.eval_dataset import NON_ANSWERABLE_TYPES
from llm_benchmark.pricing import format_usd

_TYPE_ORDER = (
    "factual",
    "reasoning",
    "portuguese",
    "numeric",
    "unanswerable",
    "ambiguous",
)
COLUMNS = (
    "group",
    "provider",
    "model",
    "type",
    "n",
    "scored",
    "correct",
    "accuracy",
    "non_answerable",
    "overconfident",
    "overconfidence_rate",
    "errors",
    "mean_latency_ms",
    "total_cost_usd",
    "unpriced",
)


@dataclass(frozen=True)
class SummaryRow:
    """Uma linha da tabela. Taxas vazias significam denominador zero."""

    group: str
    provider: str
    model: str
    task_type: str
    n: int
    scored: int
    correct: int
    accuracy: str
    non_answerable: int
    overconfident: int
    overconfidence_rate: str
    errors: int
    mean_latency_ms: str
    total_cost_usd: str
    unpriced: int

    def as_csv(self) -> dict[str, object]:
        return {
            "group": self.group,
            "provider": self.provider,
            "model": self.model,
            "type": self.task_type,
            "n": self.n,
            "scored": self.scored,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "non_answerable": self.non_answerable,
            "overconfident": self.overconfident,
            "overconfidence_rate": self.overconfidence_rate,
            "errors": self.errors,
            "mean_latency_ms": self.mean_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "unpriced": self.unpriced,
        }


def summarize(records: Sequence[object]) -> list[SummaryRow]:
    """Agrupa por (provedor, modelo) e por (provedor, modelo, tipo)."""
    provider_rows = [
        _row("provider", key, items) for key, items in _buckets(records, by_type=False)
    ]
    type_rows = [_row("task_type", key, items) for key, items in _buckets(records, by_type=True)]
    return provider_rows + type_rows


def render_markdown(rows: Sequence[SummaryRow], *, sources: Sequence[str]) -> str:
    """Markdown com as duas leituras. O texto deixa claro que isto não é ranking."""
    listed = ", ".join(f"`{source}`" for source in sources) or "(nenhum arquivo)"
    provider_rows = [row for row in rows if row.group == "provider"]
    type_rows = [row for row in rows if row.group == "task_type"]
    parts = [
        "# Resumo da avaliação",
        "",
        "Cada linha do JSONL é uma observação. A tabela soma esse arquivo.",
        "A linha `provider` junta todos os tipos, inclusive unanswerable e ambiguous.",
        "A leitura por família está nas linhas `task_type`.",
        "Célula vazia em taxa significa denominador zero, não taxa zero.",
        "Isto não é ranking e não é resultado publicado.",
        "",
        f"Origem: {listed}",
        "",
        "## Por provedor",
        "",
        _markdown_table(provider_rows),
        "",
        "## Por tipo de tarefa",
        "",
        _markdown_table(type_rows),
        "",
    ]
    return "\n".join(parts)


def render_csv(rows: Sequence[SummaryRow]) -> str:
    """CSV com o mesmo contrato das colunas do markdown."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_csv())
    return buffer.getvalue()


def _buckets(
    records: Sequence[object], *, by_type: bool
) -> list[tuple[tuple[str, str, str], list[object]]]:
    buckets: dict[tuple[str, str, str], list[object]] = {}
    for record in records:
        task_type = _text(record, "task_type") if by_type else ""
        key = (_text(record, "provider"), _text(record, "model"), task_type)
        buckets.setdefault(key, []).append(record)

    def sort_key(item: tuple[tuple[str, str, str], list[object]]) -> tuple[str, str, int, str]:
        provider, model, task_type = item[0]
        order = _TYPE_ORDER.index(task_type) if task_type in _TYPE_ORDER else len(_TYPE_ORDER)
        return (provider, model, order, task_type)

    return sorted(buckets.items(), key=sort_key)


def _row(group: str, key: tuple[str, str, str], items: Sequence[object]) -> SummaryRow:
    provider, model, task_type = key
    scored_flags = [_correct(item) for item in items]
    scored_items = [
        item for item, flag in zip(items, scored_flags, strict=True) if flag is not None
    ]
    correct = sum(1 for item in scored_items if _correct(item) is True)
    errors = sum(1 for item in items if _optional_text(item, "error") is not None)
    non_answerable_items = [
        item
        for item in items
        if _text(item, "task_type") in NON_ANSWERABLE_TYPES and _correct(item) is not None
    ]
    overconfident = sum(1 for item in non_answerable_items if bool(_field(item, "overconfident")))
    total = Decimal(0)
    saw_cost = False
    unpriced = 0
    for item in items:
        cost = _optional_text(item, "estimated_cost_usd")
        if _optional_text(item, "error") is None and cost is None:
            unpriced += 1
        if cost is not None:
            total += Decimal(cost)
            saw_cost = True
    latencies = [float(_field(item, "latency_ms")) for item in items]
    mean = sum(latencies) / len(items) if items else 0.0
    return SummaryRow(
        group=group,
        provider=provider,
        model=model,
        task_type=task_type,
        n=len(items),
        scored=len(scored_items),
        correct=correct,
        accuracy=_ratio(correct, len(scored_items)),
        non_answerable=len(non_answerable_items),
        overconfident=overconfident,
        overconfidence_rate=_ratio(overconfident, len(non_answerable_items)),
        errors=errors,
        mean_latency_ms=f"{mean:.3f}",
        total_cost_usd=format_usd(total) if saw_cost else "",
        unpriced=unpriced,
    )


def _markdown_table(rows: Sequence[SummaryRow]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = [
        "| " + " | ".join(_cell(row.as_csv()[column]) for column in COLUMNS) + " |" for row in rows
    ]
    return "\n".join([header, rule, *body])


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{numerator / denominator:.4f}"


def _field(record: object, name: str) -> object:
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _text(record: object, name: str) -> str:
    value = _field(record, name)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _optional_text(record: object, name: str) -> str | None:
    value = _field(record, name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or null")
    return value


def _correct(record: object) -> bool | None:
    value = _field(record, "correct")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("correct must be a bool or null")
    return value
