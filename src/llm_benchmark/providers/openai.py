"""Adaptador OpenAI.

O pacote `openai` só é importado na primeira chamada real, sem cliente injetado.
"""

from llm_benchmark.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """API oficial. Sem `base_url` o SDK usa o host da OpenAI."""

    name = "openai"
    env_key_var = "OPENAI_API_KEY"
