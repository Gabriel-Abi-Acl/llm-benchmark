"""Carrega o conjunto de tarefas sintéticas.

O arquivo valida o esquema. A referência, quando existe, não é pontuada neste PR.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.errors import DatasetError

ALLOWED_CATEGORIES = frozenset({"factual", "reasoning", "portuguese"})
ALLOWED_LANGUAGES = frozenset({"en", "pt"})
_REQUIRED = ("id", "category", "language", "prompt")


@dataclass(frozen=True)
class Task:
    """Uma tarefa. `reference` é ilustrativa até o gabarito do PR2."""

    id: str
    category: str
    language: str
    prompt: str
    reference: str | None = None


def default_dataset_path() -> Path:
    """Caminho do placeholder. `LLM_BENCHMARK_TASKS` troca o arquivo sem mexer no código."""
    override = os.environ.get("LLM_BENCHMARK_TASKS")
    if override and override.strip():
        return Path(override.strip())
    return Path(__file__).resolve().parents[2] / "data" / "tasks.json"


def load_tasks(path: Path | None = None) -> list[Task]:
    """Lê e valida o JSON. Lista vazia, id duplicado ou categoria estranha falham."""
    dataset_path = path or default_dataset_path()
    try:
        raw_text = dataset_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(f"Dataset not found: {dataset_path}") from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Dataset is not valid JSON: {dataset_path}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise DatasetError("Dataset must be an object with a tasks list")

    tasks: list[Task] = []
    seen: set[str] = set()
    for index, item in enumerate(payload["tasks"]):
        task = _parse_task(item, index)
        if task.id in seen:
            raise DatasetError(f"Duplicate task id: {task.id}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise DatasetError("Dataset has no tasks")
    return tasks


def _parse_task(raw: object, index: int) -> Task:
    if not isinstance(raw, dict):
        raise DatasetError(f"Task at index {index} is not an object")
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise DatasetError(f"Task at index {index} missing {', '.join(missing)}")

    task_id = raw["id"]
    if not isinstance(task_id, str) or not task_id.strip():
        raise DatasetError(f"Task at index {index} has an empty id")
    task_id = task_id.strip()

    category = raw["category"]
    if category not in ALLOWED_CATEGORIES:
        raise DatasetError(f"Task {task_id} has unknown category: {category}")

    language = raw["language"]
    if language not in ALLOWED_LANGUAGES:
        raise DatasetError(f"Task {task_id} has unknown language: {language}")

    prompt = raw["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise DatasetError(f"Task {task_id} has an empty prompt")

    reference = raw.get("reference")
    if reference is not None and not isinstance(reference, str):
        raise DatasetError(f"Task {task_id} reference must be a string")

    return Task(
        id=task_id,
        category=category,
        language=language,
        prompt=prompt,
        reference=reference,
    )
