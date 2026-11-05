"""Cache em disco da resposta bem-sucedida.

A chave é o pedido (provedor, modelo, prompt), não o texto da resposta.
Erro não entra: a próxima execução ainda pode obter uma resposta.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_benchmark.types import LLMRequest

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CachedResponse:
    """O que o rerun precisa para não chamar o provedor de novo."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def request_key(
    *,
    provider: str,
    model: str,
    request: LLMRequest,
    namespace: str,
) -> dict[str, object]:
    """Identidade da chamada. `namespace` separa tarefa e juiz no mesmo diretório."""
    return {
        "v": 1,
        "namespace": namespace,
        "provider": provider,
        "model": model,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "system": request.system,
        "prompt": request.prompt,
    }


def load_cached(cache_dir: Path, key: Mapping[str, object]) -> CachedResponse | None:
    """Devolve a resposta ou None se não houver arquivo válido para esta chave."""
    path = _path_for(cache_dir, key)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("key") != dict(key):
        return None
    return _parse_cached(payload)


def store_cached(cache_dir: Path, key: Mapping[str, object], cached: CachedResponse) -> None:
    """Grava atômico. Arquivo pela metade não pode virar hit na próxima leitura."""
    path = _path_for(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "key": dict(key),
        "text": cached.text,
        "model": cached.model,
        "provider": cached.provider,
        "input_tokens": cached.input_tokens,
        "output_tokens": cached.output_tokens,
        "latency_ms": cached.latency_ms,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _path_for(cache_dir: Path, key: Mapping[str, object]) -> Path:
    encoded = json.dumps(dict(key), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if _DIGEST.fullmatch(digest) is None:
        raise ValueError("invalid cache digest")
    root = cache_dir.resolve()
    path = (root / f"{digest}.json").resolve()
    # O nome é só o hash. Recusar outro pai impede o arquivo sair do diretório.
    if path.parent != root:
        raise ValueError("cache path escaped the cache directory")
    return path


def _parse_cached(payload: Mapping[str, object]) -> CachedResponse | None:
    try:
        text = payload["text"]
        model = payload["model"]
        provider = payload["provider"]
        input_tokens = payload["input_tokens"]
        output_tokens = payload["output_tokens"]
        latency_ms = payload["latency_ms"]
    except KeyError:
        return None
    if not isinstance(text, str) or not isinstance(model, str) or not isinstance(provider, str):
        return None
    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 0:
        return None
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0:
        return None
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, int | float):
        return None
    return CachedResponse(
        text=text,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=float(latency_ms),
    )
