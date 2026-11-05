"""Contrato da tabela: conta, taxa e custo por provedor e por tipo."""

from llm_benchmark.aggregate import render_csv, render_markdown, summarize
from llm_benchmark.eval_run import EvalRecord


def _record(**overrides: object) -> EvalRecord:
    payload: dict[str, object] = {
        "task_id": "t",
        "task_type": "factual",
        "provider": "fake",
        "model": "m",
        "answer": "x",
        "latency_ms": 0.0,
        "input_tokens": 1,
        "output_tokens": 1,
        "estimated_cost_usd": None,
        "error": None,
        "cached": False,
        "abstained": False,
        "correct": False,
        "score_method": "normalized",
        "overconfident": False,
    }
    payload.update(overrides)
    return EvalRecord(**payload)  # type: ignore[arg-type]


def test_summary_math_for_provider_and_task_type() -> None:
    records = [
        _record(task_id="f1", latency_ms=100.0, estimated_cost_usd="0.00125", correct=True),
        _record(task_id="f2", latency_ms=300.0, estimated_cost_usd="0.00125", correct=False),
        _record(
            task_id="u1",
            task_type="unanswerable",
            latency_ms=50.0,
            estimated_cost_usd="0.0001",
            answer="I don't know",
            correct=True,
            abstained=True,
            score_method="abstention",
            overconfident=False,
        ),
        _record(
            task_id="u2",
            task_type="unanswerable",
            latency_ms=150.0,
            estimated_cost_usd="0.0001",
            correct=False,
            score_method="abstention",
            overconfident=True,
        ),
        _record(
            task_id="u3",
            task_type="unanswerable",
            latency_ms=10.0,
            answer=None,
            input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            error="RuntimeError: nope",
            correct=None,
            score_method=None,
        ),
        _record(
            task_id="n1",
            task_type="numeric",
            provider="other",
            model="n",
            latency_ms=5.0,
            correct=True,
            estimated_cost_usd=None,
        ),
    ]
    rows = summarize(records)
    by_key = {(row.group, row.provider, row.model, row.task_type): row for row in rows}

    provider = by_key[("provider", "fake", "m", "")]
    assert provider.n == 5
    assert provider.scored == 4
    assert provider.correct == 2
    assert provider.accuracy == "0.5000"
    assert provider.non_answerable == 2
    assert provider.overconfident == 1
    assert provider.overconfidence_rate == "0.5000"
    assert provider.errors == 1
    assert provider.mean_latency_ms == "122.000"
    assert provider.total_cost_usd == "0.0027"
    assert provider.unpriced == 0

    factual = by_key[("task_type", "fake", "m", "factual")]
    assert factual.n == 2
    assert factual.correct == 1
    assert factual.accuracy == "0.5000"
    assert factual.non_answerable == 0
    assert factual.overconfidence_rate == ""
    assert factual.total_cost_usd == "0.0025"

    unanswerable = by_key[("task_type", "fake", "m", "unanswerable")]
    assert unanswerable.scored == 2
    assert unanswerable.correct == 1
    assert unanswerable.errors == 1
    assert unanswerable.overconfidence_rate == "0.5000"
    assert unanswerable.mean_latency_ms == "70.000"
    assert unanswerable.total_cost_usd == "0.0002"

    other = by_key[("provider", "other", "n", "")]
    assert other.accuracy == "1.0000"
    assert other.overconfidence_rate == ""
    assert other.total_cost_usd == ""
    assert other.unpriced == 1

    markdown = render_markdown(rows, sources=["runs/fake.jsonl"])
    assert "## Por provedor" in markdown
    assert "## Por tipo de tarefa" in markdown
    assert "0.5000" in markdown
    assert "não é ranking" in markdown
    assert "runs/fake.jsonl" in markdown

    csv_text = render_csv(rows)
    assert csv_text.splitlines()[0].startswith("group,provider,model,type,")
    assert "provider,fake,m," in csv_text
    assert "task_type,fake,m,unanswerable," in csv_text
    assert "provider,other,n," in csv_text


def test_two_correct_of_three_displays_four_decimals() -> None:
    records = [
        _record(task_id="a", correct=True),
        _record(task_id="b", correct=True),
        _record(task_id="c", correct=False),
    ]
    provider = summarize(records)[0]
    assert provider.scored == 3
    assert provider.correct == 2
    assert provider.accuracy == "0.6667"
