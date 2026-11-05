"""Contrato do placeholder: 10 a 15 tarefas sintéticas, três famílias, sem id repetido."""

import json
from pathlib import Path

import pytest

from llm_benchmark.dataset import load_tasks
from llm_benchmark.errors import DatasetError


def test_placeholder_covers_factual_reasoning_and_portuguese() -> None:
    tasks = load_tasks()
    assert 10 <= len(tasks) <= 15
    assert {task.category for task in tasks} == {"factual", "reasoning", "portuguese"}
    assert any(task.language == "pt" for task in tasks)
    assert any(task.language == "en" for task in tasks)
    assert len({task.id for task in tasks}) == len(tasks)
    assert all(task.prompt.strip() for task in tasks)
    assert all(task.reference for task in tasks)


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    _task("same"),
                    _task("same"),
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="Duplicate"):
        load_tasks(path)


def test_unknown_category_and_empty_prompt_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps({"tasks": [_task("a", category="coding")]}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unknown category"):
        load_tasks(path)

    path.write_text(
        json.dumps({"tasks": [_task("a", prompt="  ")]}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="empty prompt"):
        load_tasks(path)


def test_missing_file_is_a_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_tasks(tmp_path / "absent.json")


def _task(
    task_id: str,
    *,
    category: str = "factual",
    prompt: str = "What is 2 + 2?",
) -> dict[str, str]:
    return {
        "id": task_id,
        "category": category,
        "language": "en",
        "prompt": prompt,
        "reference": "4",
    }
