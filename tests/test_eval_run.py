"""Contrato do harness: nota, custo, cache, retry e o v1 com provedor fake."""

from decimal import Decimal
from pathlib import Path

import pytest

from llm_benchmark.errors import DatasetError, ProviderResponseError
from llm_benchmark.eval_dataset import (
    NON_ANSWERABLE_TYPES,
    EvalTask,
    load_eval_tasks,
)
from llm_benchmark.eval_run import read_records, run_evaluation, write_records
from llm_benchmark.pricing import ModelPrice
from llm_benchmark.providers.fake import FakeProvider
from llm_benchmark.retry import backoff_seconds, is_rate_limit_error
from llm_benchmark.types import LLMResponse, TokenUsage


class _Scripted:
    def __init__(
        self, answers: dict[str, str], *, failures: int = 0, error: Exception | None = None
    ) -> None:
        self.name = "fake"
        self.model = "gpt-5-mini"
        self._answers = answers
        self._failures = failures
        self._error = error
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._failures:
            self._failures -= 1
            raise _RateLimit()
        return LLMResponse(
            text=self._answers.get(request.prompt, request.prompt),
            model=self.model,
            provider=self.name,
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )


class _RateLimit(Exception):
    def __init__(self, retry_after: float | None = None) -> None:
        super().__init__("rate limit")
        self.status_code = 429
        self.retry_after = retry_after


class _Judge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def judge(self, task: EvalTask, answer: str) -> bool:
        self.calls.append(task.id)
        return task.id == "reason-1"


def _task(task_id: str, task_type: str, prompt: str, reference: str | None) -> EvalTask:
    return EvalTask(
        id=task_id,
        prompt=prompt,
        reference_answer=reference,
        type=task_type,
        notes=None if reference is not None else "sem dado",
    )


def test_runner_scores_a_few_tasks_and_prices_them() -> None:
    tasks = [
        _task("fact-1", "factual", "symbol", "Au"),
        _task("num-1", "numeric", "sum", "45"),
        _task("reason-1", "reasoning", "logic", "yes"),
        _task("pt-1", "portuguese", "capital", "Brasília"),
        _task("unans-1", "unanswerable", "missing", None),
        _task("ambig-1", "ambiguous", "bank", None),
    ]
    provider = _Scripted(
        {
            "symbol": "AU.",
            "sum": "45.0",
            "logic": "the answer is affirmative",
            "capital": "Rio",
            "missing": "I don't know",
            "bank": "12 meters",
        }
    )
    judge = _Judge()
    ticks = iter(float(index) for index in range(12))
    records = run_evaluation(provider, tasks, clock=lambda: next(ticks), judge=judge)

    by_id = {record.task_id: record for record in records}
    assert by_id["fact-1"].correct is True
    assert by_id["fact-1"].score_method == "normalized"
    assert by_id["fact-1"].estimated_cost_usd == "0.00125"
    assert by_id["fact-1"].input_tokens == 1000
    assert by_id["fact-1"].output_tokens == 500
    assert by_id["fact-1"].latency_ms == 1000.0
    assert by_id["num-1"].correct is True
    assert by_id["reason-1"].correct is True
    assert by_id["reason-1"].score_method == "judge"
    assert by_id["pt-1"].correct is False
    assert by_id["pt-1"].score_method == "judge"
    assert by_id["unans-1"].correct is True
    assert by_id["unans-1"].overconfident is False
    assert by_id["ambig-1"].correct is False
    assert by_id["ambig-1"].overconfident is True
    assert judge.calls == ["reason-1", "pt-1"]
    assert all(record.error is None for record in records)
    assert all(record.cached is False for record in records)


def test_one_task_error_does_not_stop_the_run() -> None:
    class _Boom:
        name = "fake"
        model = "gpt-5-mini"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            if request.prompt == "bad":
                raise RuntimeError("nope")
            return LLMResponse(
                text="Au",
                model=self.model,
                provider=self.name,
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )

    provider = _Boom()
    tasks = [
        _task("bad", "factual", "bad", "Au"),
        _task("ok", "factual", "good", "Au"),
    ]
    records = run_evaluation(
        provider, tasks, clock=lambda: 0.0, max_attempts=4, sleep=lambda _seconds: None
    )
    assert provider.calls == 2
    assert records[0].error == "RuntimeError: nope"
    assert records[0].correct is None
    assert records[0].overconfident is False
    assert records[0].answer is None
    assert records[0].estimated_cost_usd is None
    assert records[1].correct is True
    assert records[1].answer == "Au"


def test_rate_limit_retries_then_records_the_successful_attempt() -> None:
    provider = _Scripted({"symbol": "Au"}, failures=2)
    sleeps: list[float] = []
    ticks = iter((0.0, 1.0, 2.0, 3.0, 10.0, 10.25))
    records = run_evaluation(
        provider,
        [_task("fact-1", "factual", "symbol", "Au")],
        clock=lambda: next(ticks),
        max_attempts=3,
        base_delay_s=0.5,
        sleep=sleeps.append,
    )
    assert provider.calls == 3
    assert sleeps == [0.5, 1.0]
    assert records[0].error is None
    assert records[0].answer == "Au"
    assert records[0].latency_ms == 250.0
    assert records[0].correct is True


def test_exhausted_rate_limit_is_a_task_error() -> None:
    provider = _Scripted({}, failures=5)
    sleeps: list[float] = []
    records = run_evaluation(
        provider,
        [_task("fact-1", "factual", "symbol", "Au")],
        clock=lambda: 0.0,
        max_attempts=2,
        base_delay_s=0.0,
        sleep=sleeps.append,
    )
    assert provider.calls == 2
    assert sleeps == [0.0]
    assert records[0].answer is None
    assert records[0].correct is None
    assert records[0].error == "_RateLimit: rate limit"


def test_non_rate_limit_is_not_retried() -> None:
    provider = _Scripted({}, error=ProviderResponseError("response has no usage"))
    sleeps: list[float] = []
    records = run_evaluation(
        provider,
        [_task("fact-1", "factual", "symbol", "Au")],
        clock=lambda: 0.0,
        max_attempts=5,
        sleep=sleeps.append,
    )
    assert provider.calls == 1
    assert sleeps == []
    assert records[0].correct is None
    assert "ProviderResponseError" in (records[0].error or "")


def test_backoff_honors_retry_after_when_it_is_longer() -> None:
    limited = _RateLimit(retry_after=4)
    assert is_rate_limit_error(limited)
    assert is_rate_limit_error(ProviderResponseError("response has no usage")) is False
    assert is_rate_limit_error(RuntimeError("connection reset")) is False

    class _Named(Exception):
        pass

    _Named.__name__ = "RateLimitError"
    assert is_rate_limit_error(_Named("nope"))
    assert backoff_seconds(1, 0.5, _RateLimit()) == 0.5
    assert backoff_seconds(2, 0.5, _RateLimit()) == 1.0
    assert backoff_seconds(1, 0.5, limited) == 4.0
    assert backoff_seconds(1, 0.5, _RateLimit(retry_after=-1)) == 0.5


def test_cache_skips_the_provider_and_reprices_on_rerun(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    provider = _Scripted({"symbol": "Au"})
    task = _task("fact-1", "factual", "symbol", "Au")
    ticks = iter((10.0, 10.25))
    first = run_evaluation(provider, [task], clock=lambda: next(ticks), cache_dir=cache)
    assert provider.calls == 1
    assert first[0].cached is False
    assert first[0].latency_ms == 250.0
    assert first[0].estimated_cost_usd == "0.00125"

    custom = {
        "gpt-5-mini": ModelPrice(
            input_usd_per_million=Decimal("1"),
            output_usd_per_million=Decimal("0"),
            source="local override",
            note="teste",
        )
    }
    second = run_evaluation(
        provider,
        [task],
        clock=lambda: pytest.fail("cache hit must not read the clock"),
        cache_dir=cache,
        prices=custom,
    )
    assert provider.calls == 1
    assert second[0].cached is True
    assert second[0].answer == "Au"
    assert second[0].latency_ms == 250.0
    assert second[0].estimated_cost_usd == "0.001"
    assert second[0].correct is True


def test_corrupt_cache_and_other_model_do_not_reuse_the_answer(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    provider = _Scripted({"symbol": "Au"})
    task = _task("fact-1", "factual", "symbol", "Au")
    run_evaluation(provider, [task], clock=lambda: 0.0, cache_dir=cache)
    for path in cache.glob("*.json"):
        path.write_text("not-json", encoding="utf-8")
    run_evaluation(provider, [task], clock=lambda: 0.0, cache_dir=cache)
    assert provider.calls == 2

    provider.model = "other-model"
    run_evaluation(provider, [task], clock=lambda: 0.0, cache_dir=cache)
    assert provider.calls == 3


def test_judge_calls_are_cached_separately_from_the_candidate(tmp_path: Path) -> None:
    class _YesJudge:
        name = "fake"
        model = "judge-echo"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            return LLMResponse(
                text="yes",
                model=self.model,
                provider=self.name,
                usage=TokenUsage(input_tokens=4, output_tokens=1),
            )

    provider = _Scripted({"logic": "a paraphrase"})
    judge = _YesJudge()
    task = _task("reason-1", "reasoning", "logic", "yes")
    cache = tmp_path / "cache"
    first = run_evaluation(
        provider,
        [task],
        clock=lambda: 0.0,
        cache_dir=cache,
        judge_provider=judge,
    )
    second = run_evaluation(
        provider,
        [task],
        clock=lambda: 0.0,
        cache_dir=cache,
        judge_provider=judge,
    )
    assert first[0].correct is True
    assert first[0].score_method == "judge"
    assert first[0].estimated_cost_usd == "0.00125"
    assert provider.calls == 1
    assert judge.calls == 1
    assert second[0].cached is True
    assert second[0].correct is True
    assert provider.calls == 1
    assert judge.calls == 1


def test_shipped_v1_runs_offline_on_the_echo_provider() -> None:
    tasks = load_eval_tasks()
    records = run_evaluation(FakeProvider(), tasks, clock=lambda: 0.0)
    assert [record.task_id for record in records] == [task.id for task in tasks]
    assert len(records) == 52
    for record, task in zip(records, tasks, strict=True):
        assert record.error is None
        assert record.cached is False
        assert record.estimated_cost_usd is None
        assert record.answer == task.prompt
        assert record.provider == "fake"
        if task.type in NON_ANSWERABLE_TYPES:
            assert record.correct is False
            assert record.overconfident is True
            assert record.score_method == "abstention"
        else:
            assert record.correct is False
            assert record.overconfident is False
            assert record.score_method == "normalized"


def test_jsonl_round_trip_and_rejection(tmp_path: Path) -> None:
    provider = _Scripted({"symbol": "Au"})
    records = run_evaluation(
        provider,
        [_task("fact-1", "factual", "symbol", "Au")],
        clock=lambda: 0.0,
    )
    path = tmp_path / "out.jsonl"
    write_records(path, records)
    assert read_records(path) == records
    path.write_text(
        path.read_text(encoding="utf-8").replace("}", ', "extra": 1}', 1), encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="unknown fields"):
        read_records(path)
