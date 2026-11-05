"""Adaptador local no protocolo do Ollama.

`stream=False` é obrigatório: no fluxo a resposta não traz a contagem final de tokens.
Custo de API é zero no runner porque `provider` sai como `local`.
"""

from collections.abc import Mapping

from llm_benchmark.errors import ProviderDependencyError, ProviderResponseError
from llm_benchmark.providers.base import require_prompt
from llm_benchmark.types import LLMRequest, LLMResponse, TokenUsage


class LocalProvider:
    """Cliente `ollama.Client`. O pacote só entra na primeira chamada sem cliente injetado."""

    name = "local"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        client: object | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        require_prompt(request)
        client = self._client_or_build()
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        options: dict[str, object] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        raw = client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options=options,
        )
        return _parse_ollama(raw, model=self.model, provider=self.name)

    def _client_or_build(self) -> object:
        if self._client is not None:
            return self._client
        try:
            import ollama
        except ImportError as exc:
            raise ProviderDependencyError("ollama") from exc
        self._client = ollama.Client(host=self.base_url)
        return self._client


def _lookup(obj: object, key: str) -> object:
    if isinstance(obj, Mapping):
        if key not in obj:
            raise ProviderResponseError(f"response missing {key}")
        return obj[key]
    if not hasattr(obj, key):
        raise ProviderResponseError(f"response missing {key}")
    return getattr(obj, key)


def _parse_ollama(response: object, *, model: str, provider: str) -> LLMResponse:
    message = _lookup(response, "message")
    content = _lookup(message, "content")
    if content is None:
        raise ProviderResponseError("response has no message content")
    if not isinstance(content, str):
        raise ProviderResponseError("response message content is not a string")
    try:
        input_tokens = int(_lookup(response, "prompt_eval_count"))
        output_tokens = int(_lookup(response, "eval_count"))
    except (TypeError, ValueError) as exc:
        raise ProviderResponseError("response token counts are incomplete") from exc
    if input_tokens < 0 or output_tokens < 0:
        raise ProviderResponseError("response usage has negative token counts")
    return LLMResponse(
        text=content,
        model=model,
        provider=provider,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
