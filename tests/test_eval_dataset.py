"""Contrato do dataset v1: ids únicos, esquema, mistura, gabarito e ida-e-volta.

Tudo offline. Nenhum teste chama provedor.
"""

import json
from pathlib import Path

import pytest

from llm_benchmark.errors import DatasetError
from llm_benchmark.eval_dataset import (
    ANSWERABLE_TYPES,
    NON_ANSWERABLE_TYPES,
    V1_TYPE_COUNTS,
    default_eval_dataset_path,
    dump_eval_tasks,
    load_eval_tasks,
    main,
    validate_v1_mix,
)


def test_shipped_ids_are_unique() -> None:
    tasks = load_eval_tasks()
    assert len({task.id for task in tasks}) == len(tasks)
    assert len({task.prompt for task in tasks}) == len(tasks)


def test_type_distribution_matches_v1() -> None:
    assert set(V1_TYPE_COUNTS) == ANSWERABLE_TYPES | NON_ANSWERABLE_TYPES
    assert sum(V1_TYPE_COUNTS.values()) == 52
    assert validate_v1_mix(load_eval_tasks()) == V1_TYPE_COUNTS


def test_answerable_tasks_have_reference_answers() -> None:
    for task in load_eval_tasks():
        if task.type in ANSWERABLE_TYPES:
            assert isinstance(task.reference_answer, str)
            assert task.reference_answer.strip()
        else:
            assert task.type in NON_ANSWERABLE_TYPES
            assert task.reference_answer is None
            assert task.notes


def test_loader_round_trip(tmp_path: Path) -> None:
    source = default_eval_dataset_path()
    tasks = load_eval_tasks(source)
    dumped = dump_eval_tasks(tasks)
    assert dumped == source.read_text(encoding="utf-8")
    copy = tmp_path / "tasks.v1.jsonl"
    copy.write_text(dumped, encoding="utf-8")
    assert load_eval_tasks(copy) == tasks
    assert any(task.notes is None for task in tasks)
    assert any(task.reference_answer is None and task.notes for task in tasks)


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    text = _line("same") + "\n" + _line("same") + "\n"
    with pytest.raises(DatasetError, match="Duplicate"):
        load_eval_tasks(_write(tmp_path, text))


def test_schema_rejections(tmp_path: Path) -> None:
    cases = [
        (_line("a", type="coding") + "\n", "unknown type"),
        (_line("a", reference_answer=None) + "\n", "reference_answer"),
        (_line("a", surprise=1) + "\n", "unknown fields"),
        (_line("a", prompt="  x") + "\n", "invalid prompt"),
        ('{"id": "fact-a", "reference_answer": "4", "type": "factual"}\n', "missing"),
        ("[1]\n", "not an object"),
        ("{]\n", "not valid JSON"),
        ("\n\n", "blank line"),
    ]
    path = tmp_path / "tasks.jsonl"
    for text, match in cases:
        path.write_text(text, encoding="utf-8")
        with pytest.raises(DatasetError, match=match):
            load_eval_tasks(path)


def test_non_answerable_needs_null_reference_and_notes(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        _line("unans-a", type="unanswerable", reference_answer="n/a", notes="falta o dado") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="null reference_answer"):
        load_eval_tasks(path)

    path.write_text(
        _line("unans-a", type="ambiguous", reference_answer=None) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="needs notes"):
        load_eval_tasks(path)

    path.write_text(
        _line(
            "unans-a",
            type="unanswerable",
            reference_answer=None,
            notes="O enunciado não traz o dado.",
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_eval_tasks(path)
    assert loaded[0].reference_answer is None
    assert loaded[0].notes == "O enunciado não traz o dado."


def test_missing_file_and_invalid_utf8(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_eval_tasks(tmp_path / "absent.jsonl")
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b"\xff\n")
    with pytest.raises(DatasetError, match="UTF-8"):
        load_eval_tasks(path)


def test_v1_mix_rejects_a_short_list() -> None:
    with pytest.raises(DatasetError, match="type mix"):
        validate_v1_mix(load_eval_tasks()[:3])


def test_validation_script_accepts_shipped_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "ok: 52 tasks" in captured.out
    assert "factual=12" in captured.out
    assert "unanswerable=3" in captured.out


def test_validation_script_rejects_schema_and_mix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{}\n", encoding="utf-8")
    assert main([str(bad)]) == 1
    assert capsys.readouterr().err

    one = tmp_path / "one.jsonl"
    one.write_text(_line("fact-one") + "\n", encoding="utf-8")
    assert main([str(one)]) == 1
    assert "type mix" in capsys.readouterr().err


def _line(task_id: str = "fact-a", **overrides: object) -> str:
    row: dict[str, object] = {
        "id": task_id,
        "prompt": "What is 2 + 2? Reply with the number only.",
        "reference_answer": "4",
        "type": "factual",
    }
    row.update(overrides)
    return json.dumps(row, ensure_ascii=False)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tasks.jsonl"
    path.write_text(text, encoding="utf-8")
    return path
