"""Custo estimado em USD a partir de tabela pública, não de fatura.

Consultado em 2026-11-05. A tabela é entrada sem cache, faixa standard.
DeepSeek entra com a tarifa off-peak de cache miss (metade do pico).
Provedor `local` custa 0 de API; energia e hardware ficam de fora.
Modelo ausente devolve None: o estudo não inventa preço.
"""

from dataclasses import dataclass
from decimal import Decimal

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
) -> Decimal | None:
    """Estima a chamada. `local` zera mesmo se o nome do modelo existir na tabela cloud.

    None significa "sem preço publicado nesta tabela", diferente de custo zero.
    """
    _require_token_count(input_tokens, "input_tokens")
    _require_token_count(output_tokens, "output_tokens")
    if provider == "local":
        return Decimal(0)
    price = PRICES.get(model)
    if price is None:
        return None
    return (
        Decimal(input_tokens) * price.input_usd_per_million
        + Decimal(output_tokens) * price.output_usd_per_million
    ) / _MILLION


def _require_token_count(value: int, label: str) -> None:
    # bool é int em Python; recusar evita contar True como 1 token.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int")
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
