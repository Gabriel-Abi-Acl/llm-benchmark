"""Pontua uma resposta contra o gabarito v1.

Factual e numeric fecham por match normalizado. Reasoning e portuguese
podem ir ao juiz se o match falhar. Unanswerable e ambiguous só contam
abstenção: o juiz não promove um palpite a acerto.
"""

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol

from llm_benchmark.eval_dataset import NON_ANSWERABLE_TYPES, EvalTask
from llm_benchmark.types import LLMRequest, LLMResponse

# Famílias em que uma paráfrase ainda pode ser a mesma resposta.
JUDGE_TYPES = frozenset({"reasoning", "portuguese"})
SCORE_METHODS = frozenset({"normalized", "judge", "judge_error", "abstention"})
_MAX_ABSTENTION_WORDS = 40

_EXACT_ABSTENTION = frozenset(
    {
        "unknown",
        "abstain",
        "unanswerable",
        "ambiguous",
        "indeterminado",
        "sem resposta",
        "no answer",
        "nao sei",
        "não sei",
    }
)
_ABSTENTION_PREFIXES = (
    "i don't know",
    "i do not know",
    "i cannot answer",
    "i can't answer",
    "i can not answer",
    "i cannot determine",
    "i can't determine",
    "i do not have enough",
    "cannot answer",
    "can't answer",
    "cannot be determined",
    "can't be determined",
    "not enough information",
    "insufficient information",
    "insufficient context",
    "não sei",
    "nao sei",
    "não posso responder",
    "nao posso responder",
    "não é possível responder",
    "nao e possivel responder",
    "impossível responder",
    "impossivel responder",
    "informação insuficiente",
    "informacao insuficiente",
    "sem informação",
    "sem informacao",
    "não há informação",
    "nao ha informacao",
    "dados insuficientes",
)
_GUESS = re.compile(
    r"\b(?:maybe|perhaps|probably|talvez|acho que|possivelmente)\b",
)
_QUOTES = "\"'“”«»"

JUDGE_SYSTEM = (
    "You grade one candidate answer against a reference. "
    "Reply with yes or no as the first word and nothing else. "
    "yes only if the candidate is semantically the same answer as the reference."
)


@dataclass(frozen=True)
class Score:
    """Resultado de uma observação. `correct` nulo fica de fora da acurácia."""

    abstained: bool
    correct: bool | None
    score_method: str | None
    overconfident: bool


class SupportsJudge(Protocol):
    """Juiz opcional. Quem pontua abstenção não chama este método."""

    def judge(self, task: EvalTask, answer: str) -> bool: ...


class SupportsComplete(Protocol):
    """O juiz só precisa de `complete`. O harness injeta retry e cache por fora."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def normalize_text(value: str) -> str:
    """Caixa, espaço e pontuação externa. Acento permanece: faz parte do português."""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) >= 2 and text[0] in _QUOTES and text[-1] in _QUOTES:
        text = text[1:-1].strip()
    while text and text[-1] in ".,;:!?":
        text = text[:-1].rstrip()
    return text.strip()


def answers_match(candidate: str, reference: str) -> bool:
    """Match normalizado. Número igual vale mesmo com zero à direita ou vírgula decimal."""
    left = normalize_text(candidate)
    right = normalize_text(reference)
    if left == right:
        return True
    left_number = _as_decimal(left)
    right_number = _as_decimal(right)
    if left_number is None or right_number is None:
        return False
    return left_number == right_number


def is_abstention(answer: str | None) -> bool:
    """Recusa curta, sem número e sem marcador de palpite.

    Frase exata ou começo de recusa. Um "sorry" ou "desculpe" na frente é ignorado.
    Quarenta palavras é o teto: um texto longo que menciona "não sei" não vira abstenção.
    """
    if answer is None:
        return False
    text = normalize_text(answer)
    if text == "" or re.search(r"\d", text) or _GUESS.search(text):
        return False
    if len(text.split()) > _MAX_ABSTENTION_WORDS:
        return False
    return _refusal(text) or _refusal(_strip_acknowledgment(text))


def score_answer(
    task: EvalTask,
    answer: str | None,
    *,
    error: str | None,
    judge: SupportsJudge | None,
) -> Score:
    """Aplica a regra da família. Erro de transporte não é acerto nem alucinação."""
    if error is not None or answer is None:
        return Score(abstained=False, correct=None, score_method=None, overconfident=False)
    abstained = is_abstention(answer)
    # Sem gabarito único, acertar é calar. Responder alimenta a taxa de excesso de confiança.
    if task.type in NON_ANSWERABLE_TYPES:
        return Score(
            abstained=abstained,
            correct=abstained,
            score_method="abstention",
            overconfident=not abstained,
        )
    reference = task.reference_answer
    if reference is None:
        return Score(abstained=abstained, correct=None, score_method=None, overconfident=False)
    if answers_match(answer, reference):
        return Score(
            abstained=abstained,
            correct=True,
            score_method="normalized",
            overconfident=False,
        )
    if judge is not None and task.type in JUDGE_TYPES:
        try:
            accepted = judge.judge(task, answer)
            if not isinstance(accepted, bool):
                raise TypeError("judge must return a bool")
        except Exception:
            # Falha do juiz não apaga a resposta nem vira erro de transporte.
            return Score(
                abstained=abstained,
                correct=False,
                score_method="judge_error",
                overconfident=False,
            )
        return Score(
            abstained=abstained,
            correct=accepted,
            score_method="judge",
            overconfident=False,
        )
    return Score(abstained=abstained, correct=False, score_method="normalized", overconfident=False)


def parse_verdict(text: str) -> bool:
    """Lê yes/no (ou sim/não) na primeira palavra. Qualquer outra forma é erro do juiz."""
    if text.strip() == "":
        raise ValueError("judge verdict is empty")
    first_line = normalize_text(text.splitlines()[0])
    token = first_line.split(" ", 1)[0].strip(".,;:!?") if first_line else ""
    if token in {"yes", "y", "sim", "s"}:
        return True
    if token in {"no", "n", "nao", "não"}:
        return False
    raise ValueError("judge verdict is not yes or no")


class LLMJudge:
    """Juiz via `complete`. Quem chama decide não usá-lo em abstenção."""

    def __init__(self, provider: SupportsComplete) -> None:
        self._provider = provider

    def judge(self, task: EvalTask, answer: str) -> bool:
        if task.reference_answer is None:
            raise ValueError("judge requires a reference answer")
        response = self._provider.complete(
            LLMRequest(
                prompt=judge_prompt(task, answer),
                system=JUDGE_SYSTEM,
                temperature=0.0,
            )
        )
        return parse_verdict(response.text)


def judge_prompt(task: EvalTask, answer: str) -> str:
    """Pedido estável. A mesma tríade (tarefa, gabarito, candidato) cai no mesmo cache."""
    return (
        f"type: {task.type}\n"
        f"prompt: {task.prompt}\n"
        f"reference: {task.reference_answer}\n"
        f"candidate: {answer}\n"
        "Is the candidate semantically equivalent to the reference? Reply yes or no."
    )


def _as_decimal(text: str) -> Decimal | None:
    compact = text.replace(" ", "")
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", compact):
        compact = compact.replace(",", "")
    elif re.fullmatch(r"[+-]?\d+,\d+", compact):
        compact = compact.replace(",", ".")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", compact) is None:
        return None
    if compact in {"+", "-", ".", "+.", "-."}:
        return None
    try:
        return Decimal(compact)
    except InvalidOperation:
        return None


def _refusal(text: str) -> bool:
    if text in _EXACT_ABSTENTION:
        return True
    # Prefixo só com fronteira: "não seio" não pode herdar a recusa de "não sei".
    return any(
        text == prefix or text.startswith(f"{prefix} ") or text.startswith(f"{prefix},")
        for prefix in _ABSTENTION_PREFIXES
    )


def _strip_acknowledgment(text: str) -> str:
    for lead in ("sorry,", "sorry", "desculpe,", "desculpe"):
        if text.startswith(lead):
            return text[len(lead) :].strip(" ,")
    return text
