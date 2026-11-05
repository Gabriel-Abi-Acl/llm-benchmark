"""Contrato do custo: tabela citada, zero local, e None quando o modelo não tem preço."""

from decimal import Decimal

import pytest

from llm_benchmark.config import load_provider_settings
from llm_benchmark.pricing import PRICES, estimate_cost_usd, format_usd


def test_default_cloud_models_are_priced_and_sourced() -> None:
    for name in ("openai", "gemini", "deepseek"):
        model = load_provider_settings(name, {}).model
        price = PRICES[model]
        assert price.source.startswith("https://")
        assert price.as_of == "2026-11-05"
        assert price.input_usd_per_million > 0
        assert price.output_usd_per_million > 0


def test_known_model_arithmetic() -> None:
    # 1000 * 0.25 / 1e6 + 500 * 2.00 / 1e6 = 0.00125
    cost = estimate_cost_usd("gpt-5-mini", 1000, 500, provider="openai")
    assert cost == Decimal("0.00125")
    assert format_usd(cost) == "0.00125"


def test_full_million_matches_the_published_rate() -> None:
    assert estimate_cost_usd("gemini-2.5-flash", 1_000_000, 0) == Decimal("0.30")
    assert estimate_cost_usd("deepseek-flash", 0, 1_000_000) == Decimal("0.60")


def test_zero_tokens_on_a_known_model_is_zero() -> None:
    assert estimate_cost_usd("gpt-5-mini", 0, 0) == Decimal(0)
    assert format_usd(Decimal(0)) == "0"


def test_unknown_model_has_no_invented_price() -> None:
    assert estimate_cost_usd("not-a-listed-model", 1_000_000, 1_000_000) is None


def test_local_provider_is_zero_even_if_the_model_name_is_priced() -> None:
    assert estimate_cost_usd("gpt-5-mini", 1_000_000, 1_000_000, provider="local") == Decimal(0)


def test_negative_and_bool_tokens_are_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_cost_usd("gpt-5-mini", -1, 0)
    with pytest.raises(TypeError):
        estimate_cost_usd("gpt-5-mini", True, 0)  # noqa: FBT003
