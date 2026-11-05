"""Esquema e loader do conjunto fechado v1.

Valida o JSONL no disco. Não chama provedor e não pontua resposta.
"""

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.errors import DatasetError

# Mistura publicada do arquivo v1. Trocar um número é outro arquivo, não um ajuste.
V1_TYPE_COUNTS: dict[str, int] = {
    "factual": 12,
    "reasoning": 12,
    "portuguese": 12,
    "numeric": 10,
    "unanswerable": 3,
    "ambiguous": 3,
}
ANSWERABLE_TYPES = frozenset({"factual", "reasoning", "portuguese", "numeric"})
NON_ANSWERABLE_TYPES = frozenset({"unanswerable", "ambiguous"})
_EVAL_TYPES = ANSWERABLE_TYPES | NON_ANSWERABLE_TYPES
_REQUIRED = ("id", "prompt", "reference_answer", "type")
_FIELDS = frozenset((*_REQUIRED, "notes"))
_ID_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


@dataclass(frozen=True)
class EvalTask:
    """Uma tarefa do gabarito v1.

    `reference_answer` é a string canônica, ou None quando o tipo não tem
    resposta única. Este PR não converte None em nota.
    """

    id: str
    prompt: str
    reference_answer: str | None
    type: str
    notes: str | None = None


def default_eval_dataset_path() -> Path:
    """Caminho do v1 no repositório. O nome do arquivo é a versão."""
    return Path(__file__).resolve().parents[2] / "data" / "eval" / "tasks.v1.jsonl"


def load_eval_tasks(path: Path | None = None) -> list[EvalTask]:
    """Lê o JSONL e aplica o esquema. Não abre rede e não pontua."""
    dataset_path = path or default_eval_dataset_path()
    try:
        raw_text = dataset_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetError(f"Dataset not found: {dataset_path}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(f"Dataset is not valid UTF-8: {dataset_path}") from exc
    except OSError as exc:
        raise DatasetError(f"Dataset not readable: {dataset_path}") from exc

    if raw_text == "":
        raise DatasetError("Dataset has no tasks")
    if not raw_text.endswith("\n"):
        raise DatasetError("Dataset must end with a newline")

    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        task = _parse_line(line, line_no)
        if task.id in seen:
            raise DatasetError(f"Duplicate task id: {task.id}")
        seen.add(task.id)
        tasks.append(task)
    if not tasks:
        raise DatasetError("Dataset has no tasks")
    return tasks


def dump_eval_tasks(tasks: Sequence[EvalTask]) -> str:
    """Regrava o JSONL canônico. A ida-e-volta preserva gabarito null e notes."""
    lines = [json.dumps(_task_payload(task), ensure_ascii=False) for task in tasks]
    return "\n".join(lines) + "\n"


def validate_v1_mix(tasks: Sequence[EvalTask]) -> dict[str, int]:
    """Confere a contagem publicada do v1. Outra mistura é outro arquivo."""
    counts = {name: 0 for name in V1_TYPE_COUNTS}
    for task in tasks:
        if task.type not in counts:
            raise DatasetError(f"Task {task.id} has unknown type: {task.type}")
        counts[task.type] += 1
    if counts != V1_TYPE_COUNTS:
        raise DatasetError(f"v1 type mix mismatch: expected {V1_TYPE_COUNTS}, found {counts}")
    return counts


def main(argv: list[str] | None = None) -> int:
    """Valida o v1 offline. Código 1 é esquema ou mistura; não há chamada de API."""
    parser = argparse.ArgumentParser(
        description="Validate the v1 eval dataset offline (schema and type mix)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="JSONL path. Default: data/eval/tasks.v1.jsonl",
    )
    args = parser.parse_args(argv)
    try:
        tasks = load_eval_tasks(args.path)
        counts = validate_v1_mix(tasks)
    except DatasetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    summary = ", ".join(f"{name}={counts[name]}" for name in sorted(counts))
    print(f"ok: {len(tasks)} tasks ({summary})")
    return 0


def _parse_line(line: str, line_no: int) -> EvalTask:
    if not line.strip():
        raise DatasetError(f"Line {line_no}: blank line")
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Line {line_no}: not valid JSON") from exc
    return _parse_task(raw, line_no)


def _parse_task(raw: object, line_no: int) -> EvalTask:
    def fail(message: str) -> DatasetError:
        return DatasetError(f"Line {line_no}: {message}")

    if not isinstance(raw, dict):
        raise fail("task is not an object")
    extra = sorted(set(raw) - _FIELDS)
    if extra:
        raise fail(f"unknown fields: {', '.join(extra)}")
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise fail(f"missing {', '.join(missing)}")

    task_id = _plain(raw["id"], what="id", fail=fail)
    if _ID_RE.fullmatch(task_id) is None:
        raise fail(f"invalid id: {task_id!r}")
    prompt = _plain(raw["prompt"], what="prompt", fail=fail)
    task_type = _plain(raw["type"], what="type", fail=fail)
    if task_type not in _EVAL_TYPES:
        raise fail(f"unknown type: {task_type}")

    # Respondível exige string. unanswerable/ambiguous ficam null e com notes:
    # uma frase de recusa aqui viraria gabarito antes do protocolo do PR3.
    reference = raw["reference_answer"]
    if task_type in NON_ANSWERABLE_TYPES:
        if reference is not None:
            raise fail(f"task {task_id} must have a null reference_answer")
        if "notes" not in raw:
            raise fail(f"task {task_id} needs notes")
    elif not isinstance(reference, str) or reference == "" or reference != reference.strip():
        raise fail(f"task {task_id} needs a reference_answer")

    notes = _plain(raw["notes"], what="notes", fail=fail) if "notes" in raw else None
    return EvalTask(
        id=task_id,
        prompt=prompt,
        reference_answer=reference if isinstance(reference, str) else None,
        type=task_type,
        notes=notes,
    )


def _plain(value: object, *, what: str, fail: Callable[[str], DatasetError]) -> str:
    if not isinstance(value, str) or value == "" or value != value.strip():
        raise fail(f"invalid {what}")
    return value


def _task_payload(task: EvalTask) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": task.id,
        "prompt": task.prompt,
        "reference_answer": task.reference_answer,
        "type": task.type,
    }
    if task.notes is not None:
        payload["notes"] = task.notes
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
