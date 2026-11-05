"""Registro dos provedores. Importar este pacote não importa SDK de terceiros."""

from collections.abc import Mapping
from typing import Any

from llm_benchmark.config import load_provider_settings
from llm_benchmark.errors import UnknownProviderError
from llm_benchmark.providers.base import LLMProvider
from llm_benchmark.providers.deepseek import DeepSeekProvider
from llm_benchmark.providers.fake import FakeProvider
from llm_benchmark.providers.gemini import GeminiProvider
from llm_benchmark.providers.local import LocalProvider
from llm_benchmark.providers.openai import OpenAIProvider

PROVIDER_NAMES = ("openai", "gemini", "deepseek", "local", "fake")


def available_providers() -> tuple[str, ...]:
    """Nomes aceitos por `build_provider`, na ordem estável do estudo."""
    return PROVIDER_NAMES


def build_provider(
    name: str,
    *,
    env: Mapping[str, str] | None = None,
    client: Any = None,
    model: str | None = None,
) -> LLMProvider:
    """Monta o adaptador. Não abre conexão; `client` existe para o teste injetar um duplo.

    `model` troca o id desta execução. A chave continua vindo do ambiente.
    """
    settings = load_provider_settings(name, env)
    chosen = model.strip() if model is not None and model.strip() else settings.model
    key = name.strip().lower()
    if key == "openai":
        return OpenAIProvider(api_key=settings.api_key, model=chosen, client=client)
    if key == "gemini":
        return GeminiProvider(api_key=settings.api_key, model=chosen, client=client)
    if key == "deepseek":
        return DeepSeekProvider(
            api_key=settings.api_key,
            model=chosen,
            base_url=settings.base_url,
            client=client,
        )
    if key == "local":
        base_url = settings.base_url or "http://127.0.0.1:11434"
        return LocalProvider(model=chosen, base_url=base_url, client=client)
    if key == "fake":
        return FakeProvider(model=chosen)
    raise UnknownProviderError(name)
