"""Leitura de configuração só pelo ambiente do processo.

Arquivo `.env` no disco não é carregado: segredo não entra no import.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from llm_benchmark.errors import UnknownProviderError


@dataclass(frozen=True)
class ProviderSettings:
    """Parâmetros de um provedor. `api_key` fica None quando a variável está vazia."""

    model: str
    api_key: str | None = None
    base_url: str | None = None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_provider_settings(name: str, env: Mapping[str, str] | None = None) -> ProviderSettings:
    """Resolve modelo, chave e URL. `env` explícito existe para o teste não mexer no processo."""
    source: Mapping[str, str] = os.environ if env is None else env
    key = name.strip().lower()

    if key == "openai":
        return ProviderSettings(
            model=_clean(source.get("OPENAI_MODEL")) or "gpt-5-mini",
            api_key=_clean(source.get("OPENAI_API_KEY")),
        )
    if key == "gemini":
        return ProviderSettings(
            model=_clean(source.get("GEMINI_MODEL")) or "gemini-2.5-flash",
            api_key=_clean(source.get("GEMINI_API_KEY")),
        )
    if key == "deepseek":
        return ProviderSettings(
            model=_clean(source.get("DEEPSEEK_MODEL")) or "deepseek-flash",
            api_key=_clean(source.get("DEEPSEEK_API_KEY")),
            base_url=_clean(source.get("DEEPSEEK_BASE_URL")) or "https://api.deepseek.com",
        )
    if key == "local":
        # Sem chave: o servidor Ollama local não usa o esquema de API dos outros três.
        return ProviderSettings(
            model=_clean(source.get("OLLAMA_MODEL")) or "llama3.2",
            base_url=_clean(source.get("OLLAMA_BASE_URL")) or "http://127.0.0.1:11434",
        )
    if key == "fake":
        return ProviderSettings(model=_clean(source.get("FAKE_MODEL")) or "fake-echo")

    raise UnknownProviderError(name)
