"""Contrato do runner: registra modelo, resposta, latência, tokens e custo; não pontua."""

from llm_benchmark.dataset import Task, load_tasks
from llm_benchmark.providers.fake import FakeProvider
from llm_benchmark.runner import run_eval
from llm_benchmark.types import LLMResponse, TokenUsage


class _Flaky:
    name = "fake"
    model = "gpt-5-mini"

    def complete(self, request):
        if request.prompt == "bad":
            raise RuntimeError("nope")
        return LLMResponse(
            text="Au",
            model=self.model,
            provider=self.name,
            usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )


def test_runner_records_cost_latency_and_keeps_going_after_an_error() -> None:
    ticks = iter((10.0, 10.25, 20.0, 20.5))
    tasks = [
        Task(id="a", category="factual", language="en", prompt="gold", reference="Au"),
        Task(id="b", category="reasoning", language="pt", prompt="bad", reference=None),
    ]
    records = run_eval(_Flaky(), tasks, clock=lambda: next(ticks))

    ok, failed = records
    assert ok.task_id == "a"
    assert ok.answer == "Au"
    assert ok.model == "gpt-5-mini"
    assert ok.provider == "fake"
    assert ok.latency_ms == 250.0
    assert ok.input_tokens == 1000
    assert ok.output_tokens == 500
    assert ok.estimated_cost_usd == "0.00125"
    assert ok.error is None
    assert "reference" not in ok.to_dict()
    assert "score" not in ok.to_dict()

    assert failed.task_id == "b"
    assert failed.answer is None
    assert failed.input_tokens is None
    assert failed.output_tokens is None
    assert failed.estimated_cost_usd is None
    assert failed.latency_ms == 500.0
    assert failed.error == "RuntimeError: nope"


def test_empty_task_list_returns_no_records() -> None:
    assert run_eval(FakeProvider(), [], clock=lambda: 0.0) == []


def test_unpriced_model_keeps_the_answer_and_null_cost() -> None:
    provider = FakeProvider(model="fake-echo", usage=TokenUsage(input_tokens=8, output_tokens=2))
    task = Task(id="t", category="factual", language="en", prompt="ping")
    record = run_eval(provider, [task], clock=lambda: 0.0)[0]
    assert record.answer == "ping"
    assert record.input_tokens == 8
    assert record.estimated_cost_usd is None
    assert record.error is None


def test_local_name_collision_does_not_bill_cloud_rates() -> None:
    class _Local:
        name = "local"
        model = "gpt-5-mini"

        def complete(self, request):
            return LLMResponse(
                text="x",
                model="gpt-5-mini",
                provider="local",
                usage=TokenUsage(input_tokens=1_000_000, output_tokens=0),
            )

    task = Task(id="t", category="factual", language="en", prompt="ping")
    record = run_eval(_Local(), [task], clock=lambda: 0.0)[0]
    assert record.estimated_cost_usd == "0"
    assert record.input_tokens == 1_000_000


def test_placeholder_set_runs_on_the_fake_provider() -> None:
    tasks = load_tasks()
    records = run_eval(FakeProvider(), tasks)
    assert [record.task_id for record in records] == [task.id for task in tasks]
    assert all(record.error is None for record in records)
    assert all(record.provider == "fake" for record in records)
    assert all(record.latency_ms >= 0 for record in records)
    assert all(record.estimated_cost_usd is None for record in records)
    assert all(record.answer == task.prompt for record, task in zip(records, tasks, strict=True))
