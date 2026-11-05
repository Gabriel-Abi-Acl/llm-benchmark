"""Estudo comparativo de LLMs.

Importar o pacote não chama API e não carrega os SDKs opcionais.
"""

from llm_benchmark.dataset import load_tasks
from llm_benchmark.pricing import estimate_cost_usd
from llm_benchmark.providers import build_provider
from llm_benchmark.runner import run_eval

__version__ = "0.1.0"

__all__ = [
    "build_provider",
    "estimate_cost_usd",
    "load_tasks",
    "run_eval",
]
