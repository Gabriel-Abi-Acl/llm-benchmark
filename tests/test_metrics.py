"""Contrato da nota: match normalizado, juiz opcional e abstenção sem juiz."""

import pytest

from llm_benchmark.eval_dataset import EvalTask
from llm_benchmark.metrics import (
    LLMJudge,
    answers_match,
    is_abstention,
    parse_verdict,
    score_answer,
)
from llm_benchmark.types import LLMResponse, TokenUsage


class _Judge:
    def __init__(self, verdict: bool = True, *, explode: bool = False) -> None:
        self.verdict = verdict
        self.explode = explode
        self.calls: list[str] = []

    def judge(self, task: EvalTask, answer: str) -> bool:
        self.calls.append(task.id)
        if self.explode:
            raise RuntimeError("judge down")
        return self.verdict


def _task(task_type: str, reference: str | None, task_id: str = "t") -> EvalTask:
    return EvalTask(
        id=task_id,
        prompt="prompt",
        reference_answer=reference,
        type=task_type,
        notes=None if reference is not None else "sem dado",
    )


def test_normalized_match_folds_case_space_and_outer_punctuation() -> None:
    assert answers_match(" Au. ", "au")
    assert answers_match('"Yes"', "yes")
    assert answers_match("H2O", "h2o")
    assert answers_match("Bom dia", "bom dia")
    assert not answers_match("Brasilia", "Brasília")
    assert not answers_match("one hundred", "100")


def test_numeric_match_accepts_equivalent_decimals_and_rejects_a_different_value() -> None:
    assert answers_match("45.0", "45")
    assert answers_match("0,75", "0.75")
    assert answers_match("3,500", "3500")
    assert answers_match(".05", "0.05")
    assert not answers_match("46", "45")
    assert not answers_match("3.5", "3500")


def test_abstention_is_a_short_refusal_without_a_guess() -> None:
    assert is_abstention("I don't know.")
    assert is_abstention("Não sei.")
    assert is_abstention("unknown")
    assert is_abstention("sorry, I don't know")
    assert is_abstention("I don't know because the city was not named")
    assert not is_abstention(None)
    assert not is_abstention("   ")
    assert not is_abstention("Mercury")
    assert not is_abstention("I don't know, maybe 1822")
    assert not is_abstention("perhaps Lisbon")
    long_refusal = "i don't know " + " ".join(["word"] * 38)
    assert len(long_refusal.split()) == 41
    assert not is_abstention(long_refusal)


def test_factual_and_numeric_ignore_the_judge() -> None:
    judge = _Judge(True)
    factual = score_answer(_task("factual", "Au"), "AU.", error=None, judge=judge)
    assert factual.correct is True
    assert factual.score_method == "normalized"
    assert factual.overconfident is False

    spelled = score_answer(_task("factual", "100"), "one hundred", error=None, judge=judge)
    assert spelled.correct is False
    assert spelled.score_method == "normalized"

    numeric = score_answer(_task("numeric", "45"), "45.0", error=None, judge=judge)
    assert numeric.correct is True
    assert numeric.score_method == "normalized"
    assert judge.calls == []


def test_reasoning_uses_the_judge_only_after_a_normalized_miss() -> None:
    judge = _Judge(True)
    exact = score_answer(_task("reasoning", "yes"), "Yes.", error=None, judge=judge)
    assert exact.correct is True
    assert exact.score_method == "normalized"
    assert judge.calls == []

    paraphrase = score_answer(
        _task("reasoning", "yes", "reason-1"),
        "the answer is affirmative",
        error=None,
        judge=judge,
    )
    assert paraphrase.correct is True
    assert paraphrase.score_method == "judge"
    assert judge.calls == ["reason-1"]

    no_judge = score_answer(_task("reasoning", "yes"), "affirmative", error=None, judge=None)
    assert no_judge.correct is False
    assert no_judge.score_method == "normalized"


def test_judge_failure_does_not_award_the_point() -> None:
    judge = _Judge(explode=True)
    scored = score_answer(_task("portuguese", "sim"), "afirmativo", error=None, judge=judge)
    assert scored.correct is False
    assert scored.score_method == "judge_error"
    assert scored.overconfident is False
    assert judge.calls == ["t"]


def test_unanswerable_and_ambiguous_score_abstention_only() -> None:
    judge = _Judge(True, explode=True)
    refused = score_answer(
        _task("unanswerable", None, "unans-1"),
        "I don't know",
        error=None,
        judge=judge,
    )
    assert refused.correct is True
    assert refused.abstained is True
    assert refused.overconfident is False
    assert refused.score_method == "abstention"

    guessed = score_answer(
        _task("ambiguous", None, "ambig-1"),
        "123",
        error=None,
        judge=judge,
    )
    assert guessed.correct is False
    assert guessed.abstained is False
    assert guessed.overconfident is True
    assert guessed.score_method == "abstention"
    assert judge.calls == []


def test_transport_error_is_unscored() -> None:
    scored = score_answer(_task("unanswerable", None), None, error="RuntimeError: nope", judge=None)
    assert scored.correct is None
    assert scored.score_method is None
    assert scored.overconfident is False
    assert scored.abstained is False


def test_parse_verdict_accepts_yes_no_and_rejects_prose() -> None:
    assert parse_verdict("Yes") is True
    assert parse_verdict("no, they differ") is False
    assert parse_verdict("sim") is True
    assert parse_verdict("Não") is False
    with pytest.raises(ValueError, match="not yes or no"):
        parse_verdict("maybe")


def test_llm_judge_sends_the_reference_and_reads_the_verdict() -> None:
    seen: list[object] = []

    class _Provider:
        def complete(self, request):
            seen.append(request)
            return LLMResponse(
                text="yes",
                model="judge",
                provider="fake",
                usage=TokenUsage(input_tokens=1, output_tokens=1),
            )

    task = _task("reasoning", "south", "reason-turn")
    assert LLMJudge(_Provider()).judge(task, "facing south") is True
    request = seen[0]
    assert "south" in request.prompt
    assert "facing south" in request.prompt
    assert request.temperature == 0.0
    assert request.system
