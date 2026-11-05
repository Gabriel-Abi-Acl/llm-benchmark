"""Contrato da CLI: fake imprime um JSON por tarefa; provedor desconhecido sai com código 2."""

import json

from llm_benchmark.__main__ import main
from llm_benchmark.dataset import load_tasks


def test_cli_fake_prints_one_record_per_task(capsys) -> None:
    code = main(["--provider", "fake"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    lines = [line for line in captured.out.splitlines() if line.strip()]
    tasks = load_tasks()
    assert len(lines) == len(tasks)
    first = json.loads(lines[0])
    assert first["provider"] == "fake"
    assert first["model"] == "fake-echo"
    assert first["task_id"] == tasks[0].id
    assert first["error"] is None
    assert first["estimated_cost_usd"] is None
    assert "reference" not in first
    assert "score" not in first


def test_cli_unknown_provider_exits_2(capsys) -> None:
    code = main(["--provider", "anthropic"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "anthropic" in captured.err
