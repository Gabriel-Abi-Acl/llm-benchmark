# llm-benchmark

Estudo comparativo entre provedores de LLM: OpenAI, Gemini, DeepSeek e um modelo local (Ollama ou API compatível). O recorte é técnico. As grandezas são qualidade da resposta, latência e custo estimado, nas mesmas tarefas.

O PR1 entregou adaptadores, estimativa de custo e o runner do placeholder. O PR2 entregou o conjunto fechado e o gabarito. O PR3 entrega o harness que pontua esse gabarito. Não há medição de modelo, ranking nem conclusão. Números de qualidade ainda não existem: a CI não chama API real e este repositório não publica resultado.

## Objetivo

Medir, no mesmo conjunto de tarefas, três coisas:

- qualidade da resposta, contra um gabarito e, opcionalmente, um juiz LLM;
- latência observada no cliente;
- custo estimado em USD, a partir dos tokens que o provedor reporta e de uma tabela de preços.

Os dados das tarefas são públicos ou sintéticos. Chave de API fica em variável de ambiente e não entra no git. O código não lê arquivo `.env`.

## Hipótese

A testar no PR4, não neste PR:

1. O preço de tabela por token não ordena sozinho o custo por tarefa, porque o número de tokens (incluindo raciocínio) varia entre modelos.
2. O desempenho em tarefas escritas em português pode divergir do desempenho em tarefas factuais curtas em inglês, no mesmo modelo.

O protocolo que compara a resposta com o gabarito está neste PR. Enquanto nenhum experimento for gravado e revisto, os dois pontos acima continuam hipótese.

## Metodologia

Unidade experimental: uma tarefa, um provedor, um modelo, com temperatura 0.

Há dois comandos. `python -m llm_benchmark` continua no placeholder `data/tasks.json` (12 tarefas sintéticas). Esse arquivo exercita o adaptador. A referência dele não entra em métrica. O estudo pontua `data/eval/tasks.v1.jsonl` via `python -m llm_benchmark.evaluate`.

Interface: cada provedor implementa `complete` e devolve texto, id do modelo e tokens. Os pacotes `openai`, `google-genai` e `ollama` são dependências opcionais, importadas só na primeira chamada real. Teste e CI usam o provedor `fake` ou um cliente injetado, então não precisam de chave.

DeepSeek fala o formato de chat da OpenAI. O adaptador importa o pacote `openai` e passa `base_url` (`https://api.deepseek.com` por padrão). O provedor `local` fala com um servidor Ollama (`OLLAMA_BASE_URL`, padrão `http://127.0.0.1:11434`) com `stream=False`, para a contagem de tokens vir na resposta.

Latência: `time.perf_counter` em volta da tentativa que produziu a resposta, em milissegundos. É tempo de parede no cliente, não o tempo interno do servidor. Espera de rate limit fica de fora. Hit de cache reaproveita a latência gravada da chamada original; a linha sai com `cached: true`.

Tokens: os que a resposta traz. No Gemini, `thoughts_token_count` soma na saída, porque a tabela da Google cobra esses tokens como output. Resposta sem contagem vira erro. O código não inventa token.

Custo: estimativa em `src/llm_benchmark/pricing.py`, consultada em 2026-11-05. O valor gravado é só da resposta candidata. Chamada de juiz, quando existe, não entra nessa conta.

| Modelo default | Entrada / 1M | Saída / 1M | Fonte | O que a conta não faz |
| --- | --- | --- | --- | --- |
| `gpt-5-mini` | USD 0,25 | USD 2,00 | [OpenAI Pricing](https://developers.openai.com/api/docs/pricing) | cache, batch, ferramentas; usa a faixa standard de contexto curto |
| `gemini-2.5-flash` | USD 0,30 | USD 2,50 | [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) | áudio, cache, batch, grounding; texto/imagem/vídeo no paid tier standard |
| `deepseek-flash` | USD 0,15 | USD 0,60 | [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/) | horário de pico (o dobro) e cache hit; a linha é off-peak com cache miss |

Provedor `local`: custo de API 0, mesmo se o nome do modelo coincidir com um id da tabela. Energia e hardware ficam de fora. Modelo ausente da tabela produz `estimated_cost_usd: null`. Custo zero fica reservado ao provedor local e a uma tarifa explícita 0 na tabela.

Consequência honesta da simplificação: com cache hit a estimativa fica acima da fatura; no pico da DeepSeek, fica abaixo. A tabela não é invoice.

Um JSON opcional (`--prices`) cobre por cima a tabela embutida, um id de modelo por chave. A tarifa decimal vai entre aspas (`"0.25"`): número JSON com ponto é recusado, para o parse não trocar o valor. Id ausente do arquivo permanece na tabela embutida. `source` e `note` são opcionais; sem eles a linha fica marcada como override local.

```json
{
  "gpt-5-mini": {
    "input_usd_per_million": "0.25",
    "output_usd_per_million": "2.00",
    "source": "https://developers.openai.com/api/docs/pricing",
    "note": "Standard, contexto curto."
  }
}
```

O que o código ainda não faz: chamar API real na CI, carregar arquivo `.env`, rodar o experimento do PR4, publicar ranking.

## Dataset v1

O arquivo é `data/eval/tasks.v1.jsonl`: 52 tarefas, uma por linha, JSON em UTF-8. A versão é o nome do arquivo. Uma mudança de enunciado ou de gabarito que altere o que se mede cria `tasks.v2.jsonl`. O v1 não é reescrito depois que um experimento passar a usá-lo. Este repositório ainda não tem experimento.

Os itens são sintéticos ou de domínio público (símbolo químico, definição do SI, calendário, gramática, conta fechada). Não há texto de cliente, conversa privada nem chave real. O `id` é a chave estável para os registros. Não se renumera um id já publicado.

Campos de cada linha:

| Campo | Papel |
| --- | --- |
| `id` | identificador único no arquivo, minúsculas e hífen |
| `prompt` | enunciado que o modelo recebe |
| `reference_answer` | gabarito canônico, sempre string, ou `null` |
| `type` | família da tarefa |
| `notes` | opcional; obrigatório quando o gabarito é `null` |

Número no gabarito continua string (`"45"`, `"0.75"`). O modelo devolve texto. Não há lista de sinônimos no arquivo. A normalização acontece na hora de pontuar, descrita abaixo.

Famílias, contagem e desenho:

| `type` | n | O que entra | Gabarito |
| --- | --- | --- | --- |
| `factual` | 12 | fato público estável, pergunta curta em inglês | string |
| `reasoning` | 12 | inferência determinística em inglês, sem dado externo | string |
| `portuguese` | 12 | enunciado em português, resposta fechada | string |
| `numeric` | 10 | conta exata ou conversão de unidade | string |
| `unanswerable` | 3 | o dado não está no enunciado | `null` |
| `ambiguous` | 3 | mais de uma resposta é defensável | `null` |

`portuguese` é a língua do enunciado, não uma segunda etiqueta em cima de `factual`. Os tipos não se cruzam.

Itens sem gabarito não carregam uma frase modelo de recusa. `notes` diz qual dado falta ou qual leitura duplica. A nota desses itens é só abstenção, definida na seção seguinte.

O esquema é a dataclass `EvalTask` em `src/llm_benchmark/eval_dataset.py`. `load_eval_tasks` recusa id duplicado, tipo fora da lista, campo extra, gabarito vazio em tarefa respondível e gabarito preenchido em `unanswerable` ou `ambiguous`. `validate_v1_mix` confere a tabela acima. O harness aceita um recorte do mesmo esquema; a mistura de 52 itens continua sendo checada por:

```bash
uv run python -m llm_benchmark.eval_dataset
```

O comando sai com 0 quando o esquema e a mistura passam, e com 1 quando não passam.

## Avaliação (PR3)

`python -m llm_benchmark.evaluate` percorre as tarefas, chama o provedor e grava um JSONL pontuado mais uma tabela em Markdown e CSV. Temperatura 0. Erro de uma tarefa vira campo `error` e a execução segue. O processo sai com 0 nesse caso. Sai com 2 se a configuração é inválida (provedor desconhecido, arquivo ilegível, chave ausente, `--max-attempts` menor que 1). Chave ausente em `openai`, `gemini` ou `deepseek` para antes da primeira chamada.

Campos de cada linha do JSONL:

| Campo | Papel |
| --- | --- |
| `task_id`, `type` | tarefa |
| `provider`, `model` | quem respondeu |
| `answer` | texto, ou `null` em erro |
| `latency_ms` | milissegundos da tentativa final, ou o valor cacheado |
| `input_tokens`, `output_tokens` | contagem reportada, ou `null` em erro |
| `estimated_cost_usd` | string decimal, `0` no local, ou `null` sem tarifa |
| `error` | `Tipo: mensagem`, ou `null` |
| `cached` | a resposta saiu do disco |
| `abstained` | a resposta foi lida como recusa |
| `correct` | `true`, `false`, ou `null` se houve erro de transporte |
| `score_method` | `normalized`, `judge`, `judge_error`, `abstention`, ou `null` |
| `overconfident` | respondeu numa tarefa sem gabarito único |

`correct` nulo fica de fora da acurácia e de fora da taxa de excesso de confiança. Não é a mesma coisa que um erro.

### Match normalizado

`factual` e `numeric` fecham só por match normalizado. O juiz não entra, mesmo ligado: "one hundred" não vale "100".

A comparação aplica casefold, junta espaço, tira aspas externas e pontuação no fim. Acento permanece (`Brasilia` e `Brasília` não batem). Se os dois lados são números, a igualdade é numérica: `45.0` vale `45`, `0,75` vale `0.75`, `3,500` vale `3500`. `3.500` continua decimal (3,5), não milhar.

`reasoning` e `portuguese` usam o mesmo match. Sem juiz, a falha do match é erro. Com juiz, o juiz só é chamado quando o match falha.

### Abstenção

Em `unanswerable` e `ambiguous`, acertar é abster. Responder conta como erro e incrementa `overconfident`. A taxa de alucinação / excesso de confiança é `overconfident / non_answerable`, só entre essas tarefas que tiveram `correct` preenchido. Família sem item desse tipo deixa a taxa vazia. Vazio não é zero.

Abstenção, em `metrics.is_abstention`: texto normalizado sem dígito, sem marcador de palpite (`maybe`, `perhaps`, `probably`, `talvez`, `acho que`, `possivelmente`), com no máximo 40 palavras, e que é uma frase de recusa ou começa por ela. Um `sorry` ou `desculpe` na frente é ignorado. `I don't know.`, `Não sei.` e `unknown` abstêm. `I don't know, maybe 1822` não abstém.

O juiz não é chamado nesses tipos e não transforma palpite em acerto.

### Juiz opcional

`--judge-provider` (e `--judge-model`, se quiser outro id) monta um segundo adaptador. O pedido pede `yes` ou `no` na primeira palavra; `sim` e `não` também valem. Outra forma vira `score_method: judge_error` e `correct: false`. Falha do juiz não apaga a resposta do candidato e não vira erro de transporte. O custo do juiz não entra em `estimated_cost_usd`.

### Cache e rate limit

O cache mora num diretório (`--cache`, padrão `.cache/llm-benchmark`, ignorado pelo git). A chave é provedor, modelo, temperatura, system, prompt e um namespace (`task` ou `judge`). Só resposta bem-sucedida é gravada. Arquivo ilegível vira miss e a chamada segue. `--no-cache` desliga.

Rate limit (HTTP 429, classe com `RateLimit` no nome, ou frase como `rate limit` / `resource exhausted` / `quota exceeded`) repete com espera exponencial. `--max-attempts` inclui a primeira tentativa (padrão 5). `--retry-base-seconds` é a base (padrão 1). `Retry-After` numérico ganha se for maior que a espera calculada. Credencial ausente, resposta sem token e erro de prompt não repetem.

### Tabela

Ao lado do JSONL saem `{nome}.md` e `{nome}.csv`, ou os caminhos de `--markdown` e `--csv`. Há uma linha por provedor+modelo e uma linha por provedor+modelo+tipo. A linha `provider` soma todos os tipos, inclusive abstenção. A leitura por família está em `task_type`.

Colunas: `n`, `scored`, `correct`, `accuracy` (`correct/scored`, quatro casas), `non_answerable`, `overconfident`, `overconfidence_rate`, `errors`, `mean_latency_ms` (média de todas as observações do grupo, três casas), `total_cost_usd` (soma só dos custos preenchidos) e `unpriced` (sucesso sem tarifa). Denominador zero deixa a taxa vazia.

`--from-jsonl` (repetível) só agrega JSONL já pontuado. Não chama provedor. Não combina com as flags de execução.

O Markdown descreve o arquivo que foi somado. Não é ranking e não é resultado publicado.

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m llm_benchmark.eval_dataset
uv run python -m llm_benchmark --provider fake
uv run python -m llm_benchmark.evaluate \
  --provider fake \
  --tasks data/eval/tasks.v1.jsonl \
  --output /tmp/llm-benchmark-fake.jsonl \
  --no-cache
```

`uv run python -m llm_benchmark --provider fake` imprime um JSON por tarefa do placeholder. Com `fake`, a resposta é o próprio prompt, a latência é só o overhead local e o custo fica nulo porque `fake-echo` não está na tabela. Isso exercita o adaptador. Não é uma medição de modelo.

O `evaluate` com `--provider fake` faz o mesmo eco em cima do v1 e grava nota, latência e custo nulo. O arquivo mostra que o harness correu. Não é medição e não deve ser citado como resultado.

Para uma chamada real, instale os SDKs, exporte as variáveis e escolha o provedor. O processo lê o ambiente. O arquivo [`.env.example`](.env.example) lista os nomes. Não commite chave.

```bash
uv sync --extra providers
export OPENAI_API_KEY="..."
uv run python -m llm_benchmark.evaluate \
  --provider openai \
  --output runs/openai.jsonl \
  --cache .cache/llm-benchmark \
  --prices prices.json \
  --judge-provider openai
```

Provedores aceitos: `openai`, `gemini`, `deepseek`, `local`, `fake`. `--model` troca o id desta execução; a chave continua a do provedor.

Agregar dois JSONL já pontuados:

```bash
uv run python -m llm_benchmark.evaluate \
  --from-jsonl runs/openai.jsonl \
  --from-jsonl runs/gemini.jsonl \
  --markdown runs/summary.md \
  --csv runs/summary.csv
```

A CI (`.github/workflows/ci.yml`) roda `uv sync --frozen`, `ruff check`, `ruff format --check` e `pytest`. O extra `providers` não entra nesse job. A CI não chama API e não grava resultado.

## Roadmap

- PR2 — dataset completo e gabarito. Feito: `data/eval/tasks.v1.jsonl`.
- PR3 — avaliação e métricas, com juiz LLM opcional. Feito: `python -m llm_benchmark.evaluate`.
- PR4 — experimentos e relatório.

## Layout

- `src/llm_benchmark/providers/` — adaptadores e o fake.
- `src/llm_benchmark/pricing.py` — tabela embutida, overlay e estimativa.
- `src/llm_benchmark/runner.py` — placeholder, um registro por tarefa, sem nota.
- `src/llm_benchmark/eval_dataset.py` — esquema, loader e validação offline do v1.
- `src/llm_benchmark/evaluate.py` — CLI da avaliação e da agregação.
- `src/llm_benchmark/eval_run.py` — execução do v1, JSONL pontuado.
- `src/llm_benchmark/metrics.py` — match, abstenção e juiz.
- `src/llm_benchmark/aggregate.py` — tabela Markdown e CSV.
- `src/llm_benchmark/cache.py` — cache em disco da resposta bem-sucedida.
- `src/llm_benchmark/retry.py` — repetição só em rate limit.
- `data/eval/tasks.v1.jsonl` — conjunto fechado e gabarito.
- `data/tasks.json` — placeholder do runner, fora da métrica.
- `tests/` — adaptadores, custo, dataset, harness offline e CLI.
