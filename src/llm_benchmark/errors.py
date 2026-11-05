"""Erros de fronteira do estudo.

A mensagem cita a variável ou o pacote, nunca o valor de uma chave.
"""


class LLMBenchmarkError(Exception):
    """Base dos erros que o runner e a CLI sabem registrar sem traceback."""


class UnknownProviderError(LLMBenchmarkError):
    """Nome fora do registro openai, gemini, deepseek, local, fake."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Unknown provider: {name}")


class MissingCredentialsError(LLMBenchmarkError):
    """Chave ausente. Deve estourar antes de importar o SDK."""

    def __init__(self, env_var: str) -> None:
        self.env_var = env_var
        super().__init__(f"Missing credential in {env_var}")


class ProviderDependencyError(LLMBenchmarkError):
    """SDK opcional não instalado. A CI não depende desses pacotes."""

    def __init__(self, package: str) -> None:
        self.package = package
        super().__init__(
            f"Package {package!r} is not installed. "
            "Install optional deps with: uv sync --extra providers"
        )


class ProviderResponseError(LLMBenchmarkError):
    """Resposta sem texto ou sem contagem de tokens.

    Sem `usage` o custo não é estimável: o adaptador não inventa token.
    """


class DatasetError(LLMBenchmarkError):
    """Conjunto de tarefas ilegível ou fora do esquema mínimo."""


class PriceTableError(LLMBenchmarkError):
    """Tabela de preços ilegível, ou tarifa que não dá para ler como decimal."""
