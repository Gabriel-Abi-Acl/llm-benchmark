"""Smoke do runner. O padrão é o provedor fake, para rodar sem chave e sem rede."""

import argparse
import json
import sys
from pathlib import Path

from llm_benchmark.dataset import load_tasks
from llm_benchmark.errors import DatasetError, UnknownProviderError
from llm_benchmark.providers import build_provider
from llm_benchmark.runner import run_eval


def main(argv: list[str] | None = None) -> int:
    """Imprime um JSON por tarefa. Erro de tarefa fica no registro; exit 2 é só config inválida."""
    parser = argparse.ArgumentParser(
        description="Run the placeholder task set against one provider."
    )
    parser.add_argument("--provider", default="fake")
    parser.add_argument("--tasks", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        provider = build_provider(args.provider)
        tasks = load_tasks(args.tasks)
    except (UnknownProviderError, DatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for record in run_eval(provider, tasks):
        print(json.dumps(record.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
