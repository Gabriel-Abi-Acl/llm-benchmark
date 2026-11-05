"""CLI da avaliação v1.

`python -m llm_benchmark` continua no placeholder, sem nota. Este módulo lê
o conjunto v1, grava o JSONL pontuado e a tabela. Não lê `.env`.
"""

import argparse
import sys
from pathlib import Path

from llm_benchmark.aggregate import render_csv, render_markdown, summarize
from llm_benchmark.errors import LLMBenchmarkError, MissingCredentialsError
from llm_benchmark.eval_dataset import load_eval_tasks
from llm_benchmark.eval_run import EvalRecord, read_records, run_evaluation, write_records
from llm_benchmark.pricing import load_price_table, overlay_prices
from llm_benchmark.providers import build_provider

_REQUIRED_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
_DEFAULT_ATTEMPTS = 5
_DEFAULT_BACKOFF = 1.0


def main(argv: list[str] | None = None) -> int:
    """Avalia um provedor ou só agrega JSONL já pontuado. Código 2 é configuração."""
    args = _parser().parse_args(argv)
    problem = _validate(args)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    try:
        if args.from_jsonl:
            return _aggregate(args)
        return _evaluate(args)
    except LLMBenchmarkError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v1 eval set for one provider, or aggregate scored JSONL. "
            "API keys come from the environment. This command does not read a .env file."
        )
    )
    parser.add_argument("--provider", default=None, help="Default when running: fake.")
    parser.add_argument("--model", default=None, help="Overrides the provider model id.")
    parser.add_argument(
        "--tasks", type=Path, default=None, help="Default: data/eval/tasks.v1.jsonl."
    )
    parser.add_argument("--output", type=Path, default=None, help="Scored JSONL path.")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--prices", type=Path, default=None, help="JSON overlay of USD per 1M tokens."
    )
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--max-attempts", type=int, default=_DEFAULT_ATTEMPTS)
    parser.add_argument("--retry-base-seconds", type=float, default=_DEFAULT_BACKOFF)
    parser.add_argument("--from-jsonl", type=Path, action="append", default=None)
    return parser


def _validate(args: argparse.Namespace) -> str | None:
    if args.max_attempts < 1:
        return "max-attempts must be >= 1"
    if args.retry_base_seconds < 0:
        return "retry-base-seconds must be >= 0"
    if args.from_jsonl:
        return _reject_run_flags(args)
    if args.output is None:
        return "missing --output"
    return None


def _reject_run_flags(args: argparse.Namespace) -> str | None:
    used = [
        flag
        for flag, value in (
            ("--output", args.output),
            ("--provider", args.provider),
            ("--model", args.model),
            ("--tasks", args.tasks),
            ("--prices", args.prices),
            ("--cache", args.cache),
            ("--no-cache", args.no_cache),
            ("--judge-provider", args.judge_provider),
            ("--judge-model", args.judge_model),
            ("--max-attempts", args.max_attempts != _DEFAULT_ATTEMPTS),
            ("--retry-base-seconds", args.retry_base_seconds != _DEFAULT_BACKOFF),
        )
        if value
    ]
    if used:
        return "aggregate-only does not accept " + ", ".join(used)
    return None


def _evaluate(args: argparse.Namespace) -> int:
    provider = build_provider(args.provider or "fake", model=args.model)
    _require_credentials(provider)
    judge_provider = None
    if args.judge_provider:
        judge_provider = build_provider(args.judge_provider, model=args.judge_model)
        _require_credentials(judge_provider)
    prices = overlay_prices(load_price_table(args.prices)) if args.prices else None
    tasks = load_eval_tasks(args.tasks)
    cache_dir = None if args.no_cache else (args.cache or Path(".cache/llm-benchmark"))
    records = run_evaluation(
        provider,
        tasks,
        prices=prices,
        cache_dir=cache_dir,
        max_attempts=args.max_attempts,
        base_delay_s=args.retry_base_seconds,
        judge_provider=judge_provider,
    )
    output: Path = args.output
    write_records(output, records)
    markdown, csv_path = _summary_paths(args, output)
    _write_summary(records, sources=[str(output)], markdown=markdown, csv_path=csv_path)
    return 0


def _aggregate(args: argparse.Namespace) -> int:
    records: list[EvalRecord] = []
    for path in args.from_jsonl:
        records.extend(read_records(path))
    markdown, csv_path = _summary_paths(args, args.from_jsonl[0])
    _write_summary(
        records,
        sources=[str(path) for path in args.from_jsonl],
        markdown=markdown,
        csv_path=csv_path,
    )
    return 0


def _summary_paths(args: argparse.Namespace, source: Path) -> tuple[Path, Path]:
    return args.markdown or source.with_suffix(".md"), args.csv or source.with_suffix(".csv")


def _write_summary(
    records: list[EvalRecord],
    *,
    sources: list[str],
    markdown: Path,
    csv_path: Path,
) -> None:
    rows = summarize(records)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(rows, sources=sources), encoding="utf-8")
    csv_path.write_text(render_csv(rows), encoding="utf-8")


def _require_credentials(provider: object) -> None:
    # Sem chave, cada tarefa falharia igual. O processo para antes da primeira chamada.
    env_var = _REQUIRED_KEYS.get(getattr(provider, "name", ""))
    if env_var is not None and not getattr(provider, "api_key", None):
        raise MissingCredentialsError(env_var)


if __name__ == "__main__":
    raise SystemExit(main())
