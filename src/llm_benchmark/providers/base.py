"""Peças comuns dos adaptadores.

Prompt vazio falha igual em todos, antes de rede ou de SDK.
"""

from typing import Protocol

from llm_benchmark.types import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    """Interface estável dos quatro provedores e do fake."""

    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def require_prompt(request: LLMRequest) -> None:
    """Recusa prompt vazio. O runner transforma isso em registro de erro."""
    if not isinstance(request.prompt, str) or request.prompt.strip() == "":
        raise ValueError("prompt must be non-empty")
