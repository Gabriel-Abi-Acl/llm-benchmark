"""Contrato da CLI de avaliação: offline, JSONL, tabela, sem chave no fake."""

import ast
import json
from pathlib import Path

from llm_benchmark.eval_run import EvalRecord, write_records
from llm_benchmark.evaluate import main

ROOT = Path(__file__).resolve().parents[1] / "src" / "llm_benchmark"


def _tasks(path: Path) -> None:
    lines = [
        {
            "id": "fact-gold",
            "prompt": "What is the chemical symbol of gold? Reply with the symbol only.",
            "reference_answer": "Au",
            "type": "factual",
        },
        {
            "id": "unans-city",
            "prompt": "What is the population of the city named in the previous sentence?",
            "reference_answer": None,
            "type": "unanswerable",
            "notes": "O prompt não nomeia cidade.",
        },
    ]
    path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )


def test_package_modules_do_not_import_provider_sdks() -> None:
    forbidden = {"openai", "google", "ollama"}
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not (names & forbidden), path.name


def test_cli_fake_writes_scored_jsonl_and_tables(tmp_path: Path, capsys) -> None:
    tasks = tmp_path / "tasks.jsonl"
    output = tmp_path / "out" / "fake.jsonl"
    cache = tmp_path / "cache"
    _tasks(tasks)
    code = main(
        [
            "--provider",
            "fake",
            "--tasks",
            str(tasks),
            "--output",
            str(output),
            "--cache",
            str(cache),
            "--no-cache",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert captured.out == ""
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["task_id"] for row in rows] == ["fact-gold", "unans-city"]
    assert rows[0]["answer"].startswith("What is the chemical symbol")
    assert rows[0]["correct"] is False
    assert rows[0]["score_method"] == "normalized"
    assert rows[0]["estimated_cost_usd"] is None
    assert rows[1]["overconfident"] is True
    assert rows[1]["score_method"] == "abstention"
    markdown = (tmp_path / "out" / "fake.md").read_text(encoding="utf-8")
    csv_text = (tmp_path / "out" / "fake.csv").read_text(encoding="utf-8")
    assert "Por provedor" in markdown
    assert "task_type,fake,fake-echo,unanswerable," in csv_text
    assert not cache.exists()


def test_cli_aggregates_existing_jsonl_without_a_provider(tmp_path: Path, capsys) -> None:
    source = tmp_path / "obs.jsonl"
    write_records(
        source,
        [
            _record("fake", "m", "factual"),
            _record("openai", "gpt-5-mini", "numeric", provider_ok=True),
        ],
    )
    markdown = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"
    code = main(["--from-jsonl", str(source), "--markdown", str(markdown), "--csv", str(csv_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    text = csv_path.read_text(encoding="utf-8")
    assert "provider,fake,m," in text
    assert "provider,openai,gpt-5-mini," in text
    assert "Origem" in markdown.read_text(encoding="utf-8")


def test_cli_rejects_bad_configuration(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "out.jsonl"
    assert main([]) == 2
    assert "missing --output" in capsys.readouterr().err

    assert main(["--output", str(output), "--max-attempts", "0"]) == 2
    assert "max-attempts" in capsys.readouterr().err
    assert not output.exists()

    assert main(["--from-jsonl", str(output), "--provider", "fake"]) == 2
    assert "aggregate-only" in capsys.readouterr().err

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--provider", "openai", "--output", str(output)]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not output.exists()

    prices = tmp_path / "prices.json"
    prices.write_text("{", encoding="utf-8")
    tasks = tmp_path / "tasks.jsonl"
    _tasks(tasks)
    assert (
        main(
            [
                "--provider",
                "fake",
                "--tasks",
                str(tasks),
                "--output",
                str(output),
                "--prices",
                str(prices),
            ]
        )
        == 2
    )
    assert "JSON" in capsys.readouterr().err


def _record(provider: str, model: str, task_type: str, *, provider_ok: bool = False) -> EvalRecord:
    return EvalRecord(
        task_id=f"{provider}-{task_type}",
        task_type=task_type,
        provider=provider,
        model=model,
        answer="x",
        latency_ms=1.0,
        input_tokens=1,
        output_tokens=1,
        estimated_cost_usd="0" if provider_ok else None,
        error=None,
        cached=False,
        abstained=False,
        correct=True,
        score_method="normalized",
        overconfident=False,
    )
