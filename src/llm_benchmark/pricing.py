"""Custo estimado em USD a partir de tabela pública, não de fatura.

Consultado em 2026-11-05. A tabela é entrada sem cache, faixa standard.
DeepSeek entra com a tarifa off-peak de cache miss (metade do pico).
Provedor `local` custa 0 de API; energia e hardware ficam de fora.
Modelo ausente devolve None: o estudo não inventa preço.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from llm_benchmark.errors import PriceTableError

AS_OF = "2026-11-05"
_MILLION = Decimal(1_000_000)
_QUANTUM = Decimal("0.00000001")


@dataclass(frozen=True)
class ModelPrice:
    """USD por 1 milhão de tokens, com a página que justifica a linha."""

    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    source: str
    note: str
    as_of: str = AS_OF


# Três defaults do PR1. Outro id de modelo fica sem linha até alguém citar a fonte.
PRICES: dict[str, ModelPrice] = {
    "gpt-5-mini": ModelPrice(
        input_usd_per_million=Decimal("0.25"),
        output_usd_per_million=Decimal("2.00"),
        source="https://developers.openai.com/api/docs/pricing",
        note="Standard, contexto curto, entrada sem cache. Cache, batch e ferramentas não entram.",
    ),
    "gemini-2.5-flash": ModelPrice(
        input_usd_per_million=Decimal("0.30"),
        output_usd_per_million=Decimal("2.50"),
        source="https://ai.google.dev/gemini-api/docs/pricing",
        note=(
            "Paid tier standard para texto, imagem e vídeo. Áudio, cache, batch e "
            "grounding não entram. Tokens de pensamento contam como saída."
        ),
    ),
    "deepseek-flash": ModelPrice(
        input_usd_per_million=Decimal("0.15"),
        output_usd_per_million=Decimal("0.60"),
        source="https://api-docs.deepseek.com/quick_start/pricing/",
        note="Off-peak e cache miss. Horário de pico (o dobro) e cache hit não são modelados.",
    ),
}


def format_usd(amount: Decimal) -> str:
    """String decimal estável para o registro, sem notação científica."""
    quantized = amount.quantize(_QUANTUM)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    provider: str | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
) -> Decimal | None:
    """Estima a chamada. `local` zera mesmo se o nome do modelo existir na tabela cloud.

    None significa "sem preço publicado nesta tabela", diferente de custo zero.
    `prices` substitui a tabela inteira desta chamada; None usa `PRICES`.
    """
    _require_token_count(input_tokens, "input_tokens")
    _require_token_count(output_tokens, "output_tokens")
    if provider == "local":
        return Decimal(0)
    table = PRICES if prices is None else prices
    price = table.get(model)
    if price is None:
        return None
    return (
        Decimal(input_tokens) * price.input_usd_per_million
        + Decimal(output_tokens) * price.output_usd_per_million
    ) / _MILLION


def load_price_table(path: Path) -> dict[str, ModelPrice]:
    """Lê um JSON de tarifas por id de modelo. Não consulta a rede.

    Decimal vai entre aspas. Um número JSON com ponto vira float e é recusado,
    para a tarifa não mudar no parse.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PriceTableError(f"Price table not readable: {path}") from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise PriceTableError(f"Price table is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise PriceTableError("Price table must be an object of models")
    table: dict[str, ModelPrice] = {}
    for model, item in payload.items():
        if not isinstance(model, str) or model == "" or model != model.strip():
            raise PriceTableError(f"Invalid model id: {model!r}")
        table[model] = _parse_price(model, item)
    return table


def overlay_prices(override: Mapping[str, ModelPrice]) -> dict[str, ModelPrice]:
    """O id que está no override substitui a linha embutida. O resto permanece."""
    merged = dict(PRICES)
    merged.update(override)
    return merged


def _require_token_count(value: int, label: str) -> None:
    # bool é int em Python; recusar evita contar True como 1 token.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")


_PRICE_FIELDS = frozenset(
    {"input_usd_per_million", "output_usd_per_million", "source", "note", "as_of"}
)


def _parse_price(model: str, item: object) -> ModelPrice:
    if not isinstance(item, dict):
        raise PriceTableError(f"Price for {model} must be an object")
    extra = sorted(set(item) - _PRICE_FIELDS)
    if extra:
        raise PriceTableError(f"Price for {model} has unknown fields: {', '.join(extra)}")
    missing = [
        key for key in ("input_usd_per_million", "output_usd_per_million") if key not in item
    ]
    if missing:
        raise PriceTableError(f"Price for {model} missing {', '.join(missing)}")
    source = item.get("source", "local override")
    note = item.get("note", "Tabela local. Não é a linha embutida.")
    as_of = item.get("as_of", AS_OF)
    if not isinstance(source, str) or source.strip() == "":
        raise PriceTableError(f"Price for {model} has an empty source")
    if not isinstance(note, str):
        raise PriceTableError(f"Price for {model} has a note that is not a string")
    if not isinstance(as_of, str) or as_of.strip() == "":
        raise PriceTableError(f"Price for {model} has an empty as_of")
    return ModelPrice(
        input_usd_per_million=_money(item["input_usd_per_million"], model=model, label="input"),
        output_usd_per_million=_money(item["output_usd_per_million"], model=model, label="output"),
        source=source.strip(),
        note=note.strip(),
        as_of=as_of.strip(),
    )


def _money(value: object, *, model: str, label: str) -> Decimal:
    # Float JSON perde o decimal citado. A tabela pede string ("0.25") ou inteiro.
    if isinstance(value, bool | float) or not isinstance(value, str | int):
        raise PriceTableError(
            f"Price for {model} {label} must be a string or int, not a JSON float"
        )
    try:
        number = Decimal(value if isinstance(value, int) else value.strip())
    except (InvalidOperation, ValueError) as exc:
        raise PriceTableError(f"Price for {model} {label} is not a decimal") from exc
    if number < 0:
        raise PriceTableError(f"Price for {model} {label} must be >= 0")
    return number
