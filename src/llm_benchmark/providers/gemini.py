"""Adaptador Gemini via `google-genai`.

O import acontece na primeira chamada sem cliente injetado.
Tokens de pensamento somam na saída porque a tabela da Google os cobra assim.
"""

from llm_benchmark.errors import (
    MissingCredentialsError,
    ProviderDependencyError,
    ProviderResponseError,
)
from llm_benchmark.providers.base import require_prompt
from llm_benchmark.types import LLMRequest, LLMResponse, TokenUsage


class GeminiProvider:
    """`client.models.generate_content`. Config vai como dict, formato aceito pelo SDK."""

    name = "gemini"
    env_key_var = "GEMINI_API_KEY"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        client: object | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        require_prompt(request)
        client = self._client_or_build()
        config: dict[str, object] = {"temperature": request.temperature}
        if request.system:
            config["system_instruction"] = request.system
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        raw = client.models.generate_content(
            model=self.model,
            contents=request.prompt,
            config=config,
        )
        return _parse_gemini(raw, model=self.model, provider=self.name)

    def _client_or_build(self) -> object:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise MissingCredentialsError(self.env_key_var)
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderDependencyError("google-genai") from exc
        self._client = genai.Client(api_key=self.api_key)
        return self._client


def _parse_gemini(response: object, *, model: str, provider: str) -> LLMResponse:
    text = getattr(response, "text", None)
    if text is None:
        raise ProviderResponseError("response has no text")
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        raise ProviderResponseError("response has no usage_metadata")
    try:
        prompt_tokens = meta.prompt_token_count
        if prompt_tokens is None:
            raise ProviderResponseError("response usage_metadata has no prompt_token_count")
        input_tokens = int(prompt_tokens)
        candidates = int(meta.candidates_token_count or 0)
        thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
    except ProviderResponseError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProviderResponseError("response usage_metadata is incomplete") from exc
    output_tokens = candidates + thoughts
    if input_tokens < 0 or output_tokens < 0:
        raise ProviderResponseError("response usage has negative token counts")
    return LLMResponse(
        text=text,
        model=model,
        provider=provider,
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
