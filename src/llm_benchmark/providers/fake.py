"""Provedor determinístico para CI e para o smoke sem rede.

Não representa um modelo. A resposta padrão devolve o próprio prompt.
"""

from collections.abc import Mapping

from llm_benchmark.providers.base import require_prompt
from llm_benchmark.types import LLMRequest, LLMResponse, TokenUsage


class FakeProvider:
    """Eco estável. `answers` cobre o caso em que o teste precisa de um texto fixo."""

    name = "fake"

    def __init__(
        self,
        *,
        model: str = "fake-echo",
        answers: Mapping[str, str] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        self.model = model
        self._answers = dict(answers or {})
        self._usage = usage or TokenUsage(input_tokens=0, output_tokens=0)

    def complete(self, request: LLMRequest) -> LLMResponse:
        require_prompt(request)
        text = self._answers.get(request.prompt, request.prompt)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            usage=self._usage,
        )
