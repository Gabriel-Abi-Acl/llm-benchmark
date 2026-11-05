# llm-benchmark

Estudo comparativo entre provedores de LLM: OpenAI, Gemini, DeepSeek e um modelo local (Ollama ou API compatível). O recorte é técnico. As grandezas são qualidade da resposta, latência e custo estimado, nas mesmas tarefas.

O PR1 entregou adaptadores, estimativa de custo e o runner. O PR2 entrega o conjunto fechado e o gabarito. Não há medição, ranking nem conclusão. Números de qualidade ainda não existem.

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

O gabarito fechado já está no repositório (PR2). O protocolo que compara a resposta do modelo com esse gabarito ainda não existe (PR3). Enquanto isso, os dois pontos acima são hipótese, não resultado.

## Metodologia planejada

Unidade experimental: uma tarefa, um provedor, um modelo, com temperatura 0.

O runner ainda lê o placeholder `data/tasks.json` (12 tarefas sintéticas, categorias `factual`, `reasoning` e `portuguese`). Esse arquivo exercita o runner. A referência nele ilustra o esquema daquele JSON e não entra em métrica. O conjunto que o estudo vai pontuar é o v1, descrito abaixo. Trocar o runner para o v1 e calcular qualidade é o PR3.

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

O que o código ainda não faz: pontuar resposta, comparar provedor, chamar API real na CI, carregar arquivo `.env`.

## Dataset v1

O arquivo é `data/eval/tasks.v1.jsonl`: 52 tarefas, uma por linha, JSON em UTF-8. A versão é o nome do arquivo. Uma mudança de enunciado ou de gabarito que altere o que se mede cria `tasks.v2.jsonl`. O v1 não é reescrito depois que um experimento passar a usá-lo. Este repositório ainda não tem experimento.

Os itens são sintéticos ou de domínio público (símbolo químico, definição do SI, calendário, gramática, conta fechada). Não há texto de cliente, conversa privada nem chave real. O `id` é a chave estável para os registros futuros. Não se renumera um id já publicado.

Campos de cada linha:

| Campo | Papel |
| --- | --- |
| `id` | identificador único no arquivo, minúsculas e hífen |
| `prompt` | enunciado que o modelo recebe |
| `reference_answer` | gabarito canônico, sempre string, ou `null` |
| `type` | família da tarefa |
| `notes` | opcional; obrigatório quando o gabarito é `null` |

Número no gabarito continua string (`"45"`, `"0.75"`). O modelo devolve texto. Guardar o número como JSON number misturaria tipo de arquivo com tipo de resposta. Não há lista de sinônimos: o enunciado pede a forma ("só o número", "sim ou não", "só o nome") e a string é essa forma. Caixa, acento e unidade não são normalizados aqui. Quem for pontuar no PR3 decide isso em cima da string crua.

Famílias, contagem e desenho:

| `type` | n | O que entra | Gabarito |
| --- | --- | --- | --- |
| `factual` | 12 | fato público estável, pergunta curta em inglês | string |
| `reasoning` | 12 | inferência determinística em inglês, sem dado externo | string |
| `portuguese` | 12 | enunciado em português, resposta fechada | string |
| `numeric` | 10 | conta exata ou conversão de unidade | string |
| `unanswerable` | 3 | o dado não está no enunciado | `null` |
| `ambiguous` | 3 | mais de uma resposta é defensável | `null` |

`portuguese` é a língua do enunciado, não uma segunda etiqueta em cima de `factual`. Os tipos não se cruzam. As quatro primeiras famílias são respondíveis. As duas últimas existem para o PR3 não tratar abstenção e chute como a mesma coisa.

Itens sem gabarito não carregam uma frase modelo de recusa. Essa frase viraria resposta certa sem o protocolo existir. `notes` diz qual dado falta (`unanswerable`: cidade não nomeada, segredo que não está no prompt, preço futuro de um ticker fictício) ou qual leitura duplica (`ambiguous`: "bank", cidade mais próxima sem origem, data numérica sem ordem e sem dia). Quanto esses itens valem não está definido neste PR.

O esquema é a dataclass `EvalTask` em `src/llm_benchmark/eval_dataset.py`. Não há dependência nova. `load_eval_tasks` recusa id duplicado, tipo fora da lista, campo extra, gabarito vazio em tarefa respondível e gabarito preenchido em `unanswerable` ou `ambiguous`. `validate_v1_mix` confere a tabela acima. `dump_eval_tasks` regrava o JSONL canônico; o teste de ida-e-volta exige que o arquivo publicado sobreviva a carregar e gravar.

Validação offline, sem rede:

```bash
uv run python -m llm_benchmark.eval_dataset
```

O comando sai com 0 quando o esquema e a mistura passam, e com 1 quando não passam. `python -m llm_benchmark` continua no placeholder. Não lê o v1.

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m llm_benchmark.eval_dataset
uv run python -m llm_benchmark --provider fake
```

`uv run python -m llm_benchmark --provider fake` imprime um JSON por tarefa. Com `fake`, a resposta é o próprio prompt, a latência é só o overhead local e o custo fica nulo porque `fake-echo` não está na tabela. Isso exercita o runner. Não é uma medição de modelo.

Para uma chamada real, instale os SDKs, exporte as variáveis e escolha o provedor. O processo lê o ambiente; não lê `.env` sozinho. O arquivo [`.env.example`](.env.example) lista os nomes.

```bash
uv sync --extra providers
export OPENAI_API_KEY="..."
uv run python -m llm_benchmark --provider openai
```

Provedores aceitos: `openai`, `gemini`, `deepseek`, `local`, `fake`.

A CI (`.github/workflows/ci.yml`) roda `uv sync --frozen`, `ruff check`, `ruff format --check` e `pytest`. O extra `providers` não entra nesse job.

## Roadmap

- PR2 — dataset completo e gabarito. Feito: `data/eval/tasks.v1.jsonl`.
- PR3 — avaliação e métricas, com juiz LLM opcional.
- PR4 — experimentos e relatório.

## Layout

- `src/llm_benchmark/providers/` — adaptadores e o fake.
- `src/llm_benchmark/pricing.py` — tabela e estimativa.
- `src/llm_benchmark/runner.py` — um registro por tarefa.
- `src/llm_benchmark/eval_dataset.py` — esquema, loader e validação offline do v1.
- `data/eval/tasks.v1.jsonl` — conjunto fechado e gabarito.
- `data/tasks.json` — placeholder do runner, fora da métrica.
- `tests/` — adaptadores, custo, runner com fake, placeholder, dataset v1 e CLI.
