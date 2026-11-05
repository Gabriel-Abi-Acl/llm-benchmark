"""Contrato dos adaptadores: registro, import preguiçoso e normalização da resposta.

Nenhum teste chama a rede. Cliente vem injetado ou o SDK é um módulo falso.
"""

import ast
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_benchmark.errors import (
    MissingCredentialsError,
    ProviderDependencyError,
    ProviderResponseError,
    UnknownProviderError,
)
from llm_benchmark.providers import available_providers, build_provider
from llm_benchmark.providers.gemini import GeminiProvider
from llm_benchmark.providers.local import LocalProvider
from llm_benchmark.providers.openai import OpenAIProvider
from llm_benchmark.types import LLMRequest

PROVIDERS_DIR = Path(__file__).resolve().parents[1] / "src" / "llm_benchmark" / "providers"
_REAL_IMPORT = builtins.__import__


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _block_sdks(monkeypatch: pytest.MonkeyPatch, *blocked: str) -> None:
    blocked_roots = tuple(blocked)

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = name.split(".")[0]
        if root in blocked_roots or name in blocked_roots:
            raise ImportError(f"blocked {name}")
        if name == "google" and fromlist and "genai" in fromlist:
            raise ImportError("blocked google.genai")
        return _REAL_IMPORT(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_sdk_imports_are_not_at_module_scope() -> None:
    forbidden = {"openai", "google", "ollama"}
    for path in PROVIDERS_DIR.glob("*.py"):
        assert not (_top_level_imports(path) & forbidden), path.name


def test_importing_package_does_not_load_vendor_sdks() -> None:
    import llm_benchmark  # noqa: F401

    assert "openai" not in sys.modules
    assert "ollama" not in sys.modules
    assert "google.genai" not in sys.modules


def test_registry_builds_defaults_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_sdks(monkeypatch, "openai", "ollama", "google")
    expected = {
        "openai": "gpt-5-mini",
        "gemini": "gemini-2.5-flash",
        "deepseek": "deepseek-flash",
        "local": "llama3.2",
        "fake": "fake-echo",
    }
    assert available_providers() == tuple(expected)
    for name, model in expected.items():
        provider = build_provider(name, env={})
        assert provider.name == name
        assert provider.model == model


def test_unknown_provider_raises() -> None:
    with pytest.raises(UnknownProviderError) as exc:
        build_provider("anthropic", env={})
    assert exc.value.name == "anthropic"


def test_deepseek_and_local_default_urls() -> None:
    deepseek = build_provider("deepseek", env={})
    local = build_provider("local", env={})
    assert deepseek.base_url == "https://api.deepseek.com"
    assert local.base_url == "http://127.0.0.1:11434"


def test_env_overrides_model_and_deepseek_url() -> None:
    provider = build_provider(
        "deepseek",
        env={
            "DEEPSEEK_API_KEY": "test-deepseek-key",
            "DEEPSEEK_MODEL": "deepseek-flash",
            "DEEPSEEK_BASE_URL": "https://example.invalid/v1",
        },
    )
    assert provider.model == "deepseek-flash"
    assert provider.base_url == "https://example.invalid/v1"


class _ChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
        )


def test_openai_normalizes_injected_client_without_importing_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_sdks(monkeypatch, "openai")
    client = _ChatClient()
    provider = OpenAIProvider(api_key=None, model="gpt-5-mini", client=client)
    response = provider.complete(
        LLMRequest(prompt="ping", system="be brief", temperature=0, max_tokens=16)
    )
    assert response.text == "ok"
    assert response.model == "gpt-5-mini"
    assert response.provider == "openai"
    assert response.usage.input_tokens == 3
    assert response.usage.output_tokens == 4
    sent = client.calls[0]
    assert sent["model"] == "gpt-5-mini"
    assert sent["temperature"] == 0
    assert sent["max_tokens"] == 16
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "ping"},
    ]


def test_openai_omits_max_tokens_when_unset() -> None:
    client = _ChatClient()
    provider = OpenAIProvider(api_key=None, model="gpt-5-mini", client=client)
    provider.complete(LLMRequest(prompt="ping"))
    assert "max_tokens" not in client.calls[0]


def test_openai_builds_sdk_client_once_on_first_real_call(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            created["kwargs"] = kwargs
            created["count"] = int(created.get("count", 0)) + 1
            self.chat = _ChatClient().chat

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    provider = build_provider(
        "openai", env={"OPENAI_API_KEY": "test-openai-key", "OPENAI_MODEL": "gpt-5-mini"}
    )
    assert "kwargs" not in created
    provider.complete(LLMRequest(prompt="ping"))
    provider.complete(LLMRequest(prompt="ping"))
    assert created["kwargs"] == {"api_key": "test-openai-key"}
    assert created["count"] == 1


def test_deepseek_passes_base_url_to_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            created["kwargs"] = kwargs
            self.chat = _ChatClient().chat

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    provider = build_provider("deepseek", env={"DEEPSEEK_API_KEY": "test-deepseek-key"})
    response = provider.complete(LLMRequest(prompt="ping"))
    assert response.provider == "deepseek"
    assert created["kwargs"] == {
        "api_key": "test-deepseek-key",
        "base_url": "https://api.deepseek.com",
    }


def test_missing_key_does_not_import_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_sdks(monkeypatch, "openai", "google", "ollama")
    openai_provider = build_provider("openai", env={})
    with pytest.raises(MissingCredentialsError) as openai_exc:
        openai_provider.complete(LLMRequest(prompt="ping"))
    assert openai_exc.value.env_var == "OPENAI_API_KEY"

    gemini_provider = build_provider("gemini", env={"GEMINI_API_KEY": "  "})
    with pytest.raises(MissingCredentialsError) as gemini_exc:
        gemini_provider.complete(LLMRequest(prompt="ping"))
    assert gemini_exc.value.env_var == "GEMINI_API_KEY"


def test_missing_sdk_raises_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_sdks(monkeypatch, "openai", "google", "ollama")
    cases = (
        ("openai", {"OPENAI_API_KEY": "test-openai-key"}, "openai"),
        ("deepseek", {"DEEPSEEK_API_KEY": "test-deepseek-key"}, "openai"),
        ("gemini", {"GEMINI_API_KEY": "gk"}, "google-genai"),
        ("local", {}, "ollama"),
    )
    for name, env, package in cases:
        provider = build_provider(name, env=env)
        with pytest.raises(ProviderDependencyError) as exc:
            provider.complete(LLMRequest(prompt="ping"))
        assert exc.value.package == package


def test_chat_response_without_usage_is_rejected() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
                    usage=None,
                )
            )
        )
    )
    provider = OpenAIProvider(api_key=None, model="gpt-5-mini", client=client)
    with pytest.raises(ProviderResponseError):
        provider.complete(LLMRequest(prompt="ping"))


def test_empty_prompt_is_rejected_before_the_client() -> None:
    client = _ChatClient()
    provider = OpenAIProvider(api_key=None, model="gpt-5-mini", client=client)
    with pytest.raises(ValueError, match="prompt must be non-empty"):
        provider.complete(LLMRequest(prompt="  "))
    assert client.calls == []


def test_gemini_adds_thought_tokens_to_output() -> None:
    captured: dict[str, object] = {}

    def generate_content(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            text="42",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=4,
                thoughts_token_count=6,
            ),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GeminiProvider(api_key=None, model="gemini-2.5-flash", client=client)
    response = provider.complete(LLMRequest(prompt="quanto", system="curto", max_tokens=8))
    assert response.provider == "gemini"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 10
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["contents"] == "quanto"
    assert captured["config"] == {
        "temperature": 0.0,
        "system_instruction": "curto",
        "max_output_tokens": 8,
    }


def test_gemini_without_usage_metadata_is_rejected() -> None:
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(text="x", usage_metadata=None)
        )
    )
    provider = GeminiProvider(api_key=None, model="gemini-2.5-flash", client=client)
    with pytest.raises(ProviderResponseError):
        provider.complete(LLMRequest(prompt="ping"))


def test_gemini_builds_client_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            created["kwargs"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "google",
        SimpleNamespace(genai=SimpleNamespace(Client=FakeClient)),
    )
    monkeypatch.setitem(
        sys.modules,
        "google.genai",
        SimpleNamespace(Client=FakeClient),
    )
    provider = build_provider(
        "gemini", env={"GEMINI_API_KEY": "gk", "GEMINI_MODEL": "gemini-2.5-flash"}
    )
    provider._client_or_build()
    assert created["kwargs"] == {"api_key": "gk"}


def test_local_reads_dict_response_and_forces_stream_off() -> None:
    captured: dict[str, object] = {}

    def chat(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "message": {"content": "olá"},
            "prompt_eval_count": 5,
            "eval_count": 2,
        }

    provider = LocalProvider(
        model="llama3.2", base_url="http://127.0.0.1:11434", client=SimpleNamespace(chat=chat)
    )
    response = provider.complete(LLMRequest(prompt="oi", max_tokens=12))
    assert response.text == "olá"
    assert response.provider == "local"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 2
    assert captured["stream"] is False
    assert captured["model"] == "llama3.2"
    assert captured["options"] == {"temperature": 0.0, "num_predict": 12}


def test_local_reads_attribute_response() -> None:
    message = SimpleNamespace(content="ok")
    raw = SimpleNamespace(message=message, prompt_eval_count=1, eval_count=1)
    provider = LocalProvider(
        model="llama3.2",
        base_url="http://127.0.0.1:11434",
        client=SimpleNamespace(chat=lambda **kwargs: raw),
    )
    response = provider.complete(LLMRequest(prompt="oi"))
    assert response.text == "ok"
    assert response.usage.output_tokens == 1


def test_local_missing_token_count_is_rejected() -> None:
    provider = LocalProvider(
        model="llama3.2",
        base_url="http://127.0.0.1:11434",
        client=SimpleNamespace(chat=lambda **kwargs: {"message": {"content": "x"}}),
    )
    with pytest.raises(ProviderResponseError):
        provider.complete(LLMRequest(prompt="oi"))


def test_build_provider_model_override_wins_over_env() -> None:
    overridden = build_provider("fake", env={"FAKE_MODEL": "from-env"}, model="from-flag")
    assert overridden.model == "from-flag"
    fallback = build_provider("fake", env={}, model="  ")
    assert fallback.model == "fake-echo"


def test_local_builds_ollama_client_with_host(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeOllama:
        def __init__(self, **kwargs: object) -> None:
            created["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(Client=FakeOllama))
    provider = build_provider(
        "local", env={"OLLAMA_BASE_URL": "http://10.0.0.8:11434", "OLLAMA_MODEL": "qwen2.5"}
    )
    provider._client_or_build()
    assert provider.model == "qwen2.5"
    assert created["kwargs"] == {"host": "http://10.0.0.8:11434"}
