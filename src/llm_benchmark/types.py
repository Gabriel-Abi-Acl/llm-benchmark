"""Contrato único entre adaptador e runner.

Custo e latência ficam de fora: o runner mede o relógio e aplica a tabela.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMRequest:
    """Uma chamada de geração. Temperatura 0 é o padrão do estudo."""

    prompt: str
    system: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass(frozen=True)
class TokenUsage:
    """Tokens reportados pelo provedor, não estimados pelo cliente."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """Resposta normalizada. `model` é o id enviado, não um snapshot da API."""

    text: str
    model: str
    provider: str
    usage: TokenUsage
