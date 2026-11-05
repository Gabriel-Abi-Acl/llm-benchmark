"""Chat Completions no formato da OpenAI.

DeepSeek usa o mesmo formato e o mesmo pacote, com outro `base_url`.
O import de `openai` fica dentro da primeira chamada sem cliente injetado.
"""

from llm_benchmark.errors import (
    MissingCredentialsError,
    ProviderDependencyError,
    ProviderResponseError,
)
from llm_benchmark.providers.base import require_prompt
from llm_benchmark.types import LLMRequest, LLMResponse, TokenUsage


class OpenAICompatibleProvider:
    """Cliente chat.completions. Subclasses só fixam nome, variável e URL padrão."""

    name: str
    env_key_var: str
    default_base_url: str | None = None

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
        client: object | None = None,
    ) -> None:
        self.api_key = api_key
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
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        raw = client.chat.completions.create(**kwargs)
        return _parse_chat(raw, model=self.model, provider=self.name)

    def _client_or_build(self) -> object:
        # Cliente injetado no teste não exige chave nem SDK.
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise MissingCredentialsError(self.env_key_var)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderDependencyError("openai") from exc
        kwargs: dict[str, str] = {"api_key": self.api_key}
        base_url = self.base_url or self.default_base_url
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        return self._client


def _parse_chat(response: object, *, model: str, provider: str) -> LLMResponse:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ProviderResponseError("response has no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None:
        raise ProviderResponseError("response has no message content")
    usage = getattr(response, "usage", None)
    if usage is None:
        raise ProviderResponseError("response has no usage")
    try:
        input_tokens = int(usage.prompt_tokens)
        output_tokens = int(usage.completion_tokens)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProviderResponseError("response usage is incomplete") from exc
    if input_tokens < 0 or output_tokens < 0:
        raise ProviderResponseError("response usage has negative token counts")
    return LLMResponse(
        text=content,
        model=model,
        provider=provider,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
