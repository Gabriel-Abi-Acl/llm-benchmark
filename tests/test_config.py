"""Contrato da config: ambiente explícito, chave em branco conta como ausente, `.env` não é lido."""

from pathlib import Path

from llm_benchmark.config import load_provider_settings


def test_blank_api_key_is_missing() -> None:
    settings = load_provider_settings(
        "openai", {"OPENAI_API_KEY": "  ", "OPENAI_MODEL": "gpt-5-mini"}
    )
    assert settings.api_key is None
    assert settings.model == "gpt-5-mini"


def test_config_does_not_read_a_dotenv_file(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    settings = load_provider_settings("openai", {})
    assert settings.api_key is None


def test_env_example_keeps_api_keys_empty() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / ".env.example").read_text(encoding="utf-8")
    assert "sk-" not in text
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.endswith("API_KEY"):
            assert value.strip() == ""


def test_gitignore_ignores_dotenv() -> None:
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored


def test_ci_runs_lint_and_tests_without_secrets() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv sync --frozen" in workflow
    assert "ruff check" in workflow
    assert "ruff format --check" in workflow
    assert "pytest" in workflow
    assert "API_KEY" not in workflow
