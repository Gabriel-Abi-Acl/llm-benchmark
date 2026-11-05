"""Adaptador DeepSeek.

A API de chat é compatível com a OpenAI; o pacote importado é `openai`,
com `base_url` apontando para a DeepSeek. Sem isso o tráfego iria para a OpenAI.
"""

from llm_benchmark.providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """Mesmo cliente da OpenAI, outra base e outra variável de chave."""

    name = "deepseek"
    env_key_var = "DEEPSEEK_API_KEY"
    default_base_url = "https://api.deepseek.com"
