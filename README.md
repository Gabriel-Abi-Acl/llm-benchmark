# llm-benchmark

Estudo comparativo entre provedores de LLM: OpenAI, Gemini, DeepSeek e um modelo local (Ollama ou API compatível). O recorte é técnico. As grandezas são qualidade da resposta, latência e custo estimado, nas mesmas tarefas.

Este repositório está no PR1. Não há medição, ranking nem conclusão. Números de qualidade ainda não existem.

## Objetivo

Medir, no mesmo conjunto de tarefas, três coisas:

- qualidade da resposta, contra um gabarito e, opcionalmente, um juiz LLM;
- latência observada no cliente;
- custo estimado em USD, a partir dos tokens que o provedor reporta e de uma tabela de preços públicos.

Os dados das tarefas são públicos ou sintéticos. Chave de API fica em variável de ambiente e não entra no git.

## Hipótese

A testar no PR4, não neste PR:

1. O preço de tabela por token não ordena sozinho o custo por tarefa, porque o número de tokens (incluindo raciocínio) varia entre modelos.
2. O desempenho em tarefas escritas em português pode divergir do desempenho em tarefas factuais curtas em inglês, no mesmo modelo.

Enquanto o gabarito e o protocolo de avaliação não existem, isto é hipótese, não resultado.

## Metodologia planejada

Unidade experimental: uma tarefa, um provedor, um modelo, com temperatura 0.

Conjunto deste PR: placeholder sintético em `data/tasks.json`, com 12 tarefas nas categorias `factual`, `reasoning` e `portuguese`. As referências no JSON ilustram o esquema. Não são o gabarito e não entram em métrica. O conjunto fechado e o gabarito ficam no PR2.

Interface: cada provedor implementa `complete` e devolve texto, id do modelo e tokens. Os pacotes `openai`, `google-genai` e `ollama` são dependências opcionais, importadas só na primeira chamada real. Teste e CI usam o provedor `fake` ou um cliente injetado, então não precisam de chave.

DeepSeek fala o formato de chat da OpenAI. O adaptador importa o pacote `openai` e passa `base_url` (`https://api.deepseek.com` por padrão). O provedor `local` fala com um servidor Ollama (`OLLAMA_BASE_URL`, padrão `http://127.0.0.1:11434`) com `stream=False`, para a contagem de tokens vir na resposta.

Latência: `time.perf_counter` no runner, em volta da chamada, em milissegundos. É tempo de parede no cliente, não o tempo interno do servidor.

Tokens: os que a resposta traz. No Gemini, `thoughts_token_count` soma na saída, porque a tabela da Google cobra esses tokens como output. Resposta sem contagem vira erro. O código não inventa token.

Custo: estimativa em `src/llm_benchmark/pricing.py`, consultada em 2026-11-05.

| Modelo default | Entrada / 1M | Saída / 1M | Fonte | O que a conta não faz |
| --- | --- | --- | --- | --- |
| `gpt-5-mini` | USD 0,25 | USD 2,00 | [OpenAI Pricing](https://developers.openai.com/api/docs/pricing) | cache, batch, ferramentas; usa a faixa standard de contexto curto |
| `gemini-2.5-flash` | USD 0,30 | USD 2,50 | [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) | áudio, cache, batch, grounding; texto/imagem/vídeo no paid tier standard |
| `deepseek-flash` | USD 0,15 | USD 0,60 | [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/) | horário de pico (o dobro) e cache hit; a linha é off-peak com cache miss |

Provedor `local`: custo de API 0, mesmo se o nome do modelo coincidir com um id da tabela. Energia e hardware ficam de fora. Modelo ausente da tabela produz `estimated_cost_usd: null`. Isso não é zero e não é um preço chute.

Consequência honesta da simplificação: com cache hit a estimativa fica acima da fatura; no pico da DeepSeek, fica abaixo. A tabela não é invoice.

O que este PR não faz: pontuar resposta, comparar provedor, chamar API real na CI, carregar arquivo `.env`.

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m llm_benchmark --provider fake
```

O comando de cima imprime um JSON por tarefa. Com `fake`, a resposta é o próprio prompt, a latência é só o overhead local e o custo fica nulo porque `fake-echo` não está na tabela. Isso exercita o runner. Não é uma medição de modelo.

Para uma chamada real, instale os SDKs, exporte as variáveis e escolha o provedor. O processo lê o ambiente; não lê `.env` sozinho. O arquivo [`.env.example`](.env.example) lista os nomes.

```bash
uv sync --extra providers
export OPENAI_API_KEY="..."
uv run python -m llm_benchmark --provider openai
```

Provedores aceitos: `openai`, `gemini`, `deepseek`, `local`, `fake`.

A CI (`.github/workflows/ci.yml`) roda `uv sync --frozen`, `ruff check`, `ruff format --check` e `pytest`. O extra `providers` não entra nesse job.

## Roadmap

- PR2 — dataset completo e gabarito.
- PR3 — avaliação e métricas, com juiz LLM opcional.
- PR4 — experimentos e relatório.

## Layout

- `src/llm_benchmark/providers/` — adaptadores e o fake.
- `src/llm_benchmark/pricing.py` — tabela e estimativa.
- `src/llm_benchmark/runner.py` — um registro por tarefa.
- `data/tasks.json` — placeholder sintético.
- `tests/` — registro dos adaptadores, custo, runner com fake, conjunto e CLI.
