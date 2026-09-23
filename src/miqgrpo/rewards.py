"""GRPO runtime rewards.

Scores *new* model completions against targets and constraints that were
computed offline and frozen into the dataset. This module never generates a
question, never resamples a property and never touches the benchmark.

The scoring path is deliberately identical to the official benchmark's:

    raw completion
      -> extract_moleculariq_answer()   (vendored from moleculariq-eval)
      -> evaluate_answer()              (moleculariq_core reward dispatcher)
      -> 0.0 / 1.0

so that "reward goes up" and "benchmark score goes up" mean the same thing.
Anything beyond that binary -- format shaping, SMILES validity -- is a small
additive bonus that can never outweigh being right.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

from moleculariq_core import evaluate_answer, valid_smiles

from .prompts import completion_to_text
from .vendor import extract_moleculariq_answer

__all__ = [
    "FormatStatus",
    "ParseStatus",
    "ScoredCompletion",
    "RewardConfig",
    "CorrectnessReward",
    "FormatReward",
    "ChemicalValidityReward",
    "build_reward_functions",
    "score_completion",
]


class ParseStatus:
    """Whether the *official* extractor + verifier produced a usable answer.

    Deliberately separate from :class:`FormatStatus`. The official extractor
    happily recovers a bare ```json block, so a completion with no ``<answer>``
    tags is a formatting miss, not a parse failure -- conflating the two made
    every early-training rollout look broken when the answers were in fact
    being read correctly.
    """

    OK = "OK"
    EMPTY = "EMPTY"
    MALFORMED = "MALFORMED"
    INVALID_SMILES = "INVALID_SMILES"
    VERIFIER_ERROR = "VERIFIER_ERROR"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"


class FormatStatus:
    """Whether the completion used the answer envelope the prompt asked for."""

    OK = "OK"
    NO_ANSWER_TAGS = "NO_ANSWER_TAGS"
    MULTIPLE_ANSWER_TAGS = "MULTIPLE_ANSWER_TAGS"
    CONFLICTING_ANSWER_TAGS = "CONFLICTING_ANSWER_TAGS"
    INVALID_JSON = "INVALID_JSON"
    EMPTY = "EMPTY"


_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

_COUNT_TASKS = frozenset({"single_count", "multi_count", "count"})
_INDEX_TASKS = frozenset({"single_index", "multi_index", "index"})
_GENERATION_TASKS = frozenset(
    {"constraint_generation", "single_constraint_generation", "multi_constraint_generation", "generation"}
)


@dataclass
class ScoredCompletion:
    """One completion's reward plus everything needed to debug it."""

    correctness: float = 0.0
    format_score: float = 0.0
    validity_score: float = 0.0
    status: str = ParseStatus.EMPTY
    format_status: str = FormatStatus.EMPTY
    extracted: str = ""
    answer_blocks: int = 0
    conflicting_answers: bool = False
    valid_smiles: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=8192)
def _loads(payload: str) -> Any:
    """Parse a stored JSON column.

    Cached on the JSON *text*, so a cache hit can only ever return the value
    that text encodes -- there is no key under which two different targets
    could collide.
    """
    return json.loads(payload)


def _normalise_task_type(task_type: str) -> str | None:
    task = (task_type or "").lower().replace("-", "_")
    if task in _COUNT_TASKS or task in _INDEX_TASKS:
        return task
    if task in _GENERATION_TASKS:
        return "constraint_generation"
    return None


def _answer_blocks(text: str) -> list[str]:
    return [block.strip() for block in _ANSWER_BLOCK.findall(text)]


#: Weights of the answer-shape components. They sum to 1.0, and the whole term
#: is then scaled by ``RewardConfig.format_weight``.
_SHAPE_ENVELOPE = 0.40
_SHAPE_JSON_OBJECT = 0.20
_SHAPE_KEYS = 0.25
_SHAPE_VALUE_TYPES = 0.15


def _as_json_object(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _value_types_ok(payload: dict[str, Any], family: str, expected: set[str]) -> bool:
    """Are the answer's values the kind of thing the task calls for?"""
    present = [payload[key] for key in expected if key in payload]
    if not present:
        return False
    if family == "count":
        return all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in present
        )
    if family == "index":
        return all(
            isinstance(v, list)
            and all(isinstance(i, int) and not isinstance(i, bool) for i in v)
            for v in present
        )
    return all(isinstance(v, str) and v.strip() for v in present)


def _shape_score(
    text: str,
    blocks: Sequence[str],
    extracted: Any,
    family: str,
    expected_keys: set[str],
) -> tuple[float, str]:
    """Graded reward for producing the answer *shape* the prompt asked for.

    This is the term that gives GRPO something to learn from in the first steps.
    Qwen2.5-0.5B answers essentially no MolecularIQ chemistry question correctly
    and, left alone, writes a ```json fence rather than the ``<answer>`` block
    the system prompt demands -- so both a binary format reward and the
    correctness reward are flat zero across a whole group, and GRPO has no
    advantage to propagate.

    Grading it into four pieces (envelope, valid JSON object, the exact keys the
    question named, plausible value types) makes rollouts differ from step one.
    None of it reveals the answer: the keys are printed in the question, and the
    value *type* is implied by the task. It is capped far below correctness, so
    a well-shaped wrong answer can never outrank a correct one.
    """
    if not text.strip():
        return 0.0, FormatStatus.EMPTY

    score = 0.0
    if len(blocks) == 1:
        score += _SHAPE_ENVELOPE
        status = FormatStatus.OK
    elif len(blocks) > 1:
        score += _SHAPE_ENVELOPE * 0.5
        status = (
            FormatStatus.CONFLICTING_ANSWER_TAGS
            if len(set(blocks)) > 1
            else FormatStatus.MULTIPLE_ANSWER_TAGS
        )
    else:
        status = FormatStatus.NO_ANSWER_TAGS

    payload = _as_json_object(blocks[-1] if blocks else extracted)
    if payload is None:
        payload = _as_json_object(extracted)
    if payload is None:
        if status == FormatStatus.OK:
            status = FormatStatus.INVALID_JSON
        return score, status

    score += _SHAPE_JSON_OBJECT
    if expected_keys and set(payload) == expected_keys:
        score += _SHAPE_KEYS
    if expected_keys and _value_types_ok(payload, family, expected_keys):
        score += _SHAPE_VALUE_TYPES
    return score, status


def _extract_smiles(extracted: Any) -> str | None:
    """Pull the SMILES out of whatever the official extractor returned."""
    value: Any = extracted
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{"):
            try:
                value = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return text
        else:
            return text
    if isinstance(value, dict):
        for key in value:
            if str(key).lower() == "smiles":
                return str(value[key]).strip()
        if len(value) == 1:
            return str(next(iter(value.values()))).strip()
        return None
    if isinstance(value, list) and len(value) == 1:
        return str(value[0]).strip()
    return None if value is None else str(value).strip()


def score_completion(
    text: str,
    task_type: str,
    target_json: str | None,
    constraints_json: str | None,
) -> ScoredCompletion:
    """Turn one raw completion into exactly one finite, diagnosed reward.

    Malformed model output is expected during RL and must never take the run
    down, so every failure mode resolves to a status plus reward 0.
    """
    result = ScoredCompletion()

    if not isinstance(text, str) or not text.strip():
        result.status = ParseStatus.EMPTY
        result.format_status = FormatStatus.EMPTY
        return result

    blocks = _answer_blocks(text)
    result.answer_blocks = len(blocks)
    result.conflicting_answers = len({b for b in blocks}) > 1

    normalised = _normalise_task_type(task_type)
    if normalised is None:
        result.status = ParseStatus.UNSUPPORTED_TASK
        result.format_status = FormatStatus.NO_ANSWER_TAGS
        return result

    extracted = extract_moleculariq_answer(text)

    # Keys the question explicitly told the model to use; for generation tasks
    # the prompt asks for `smiles`. Shape scoring only checks compliance with
    # what the prompt already says, never the answer itself.
    if normalised == "constraint_generation":
        family, expected_keys = "constraint_generation", {"smiles"}
    else:
        family = "count" if normalised in _COUNT_TASKS else "index"
        try:
            expected_keys = set(_loads(target_json)) if target_json else set()
        except (json.JSONDecodeError, ValueError):
            expected_keys = set()

    result.format_score, result.format_status = _shape_score(
        text, blocks, extracted, family, expected_keys
    )

    if extracted is None:
        result.status = ParseStatus.MALFORMED
        return result
    result.extracted = str(extracted)[:500]

    if normalised == "constraint_generation":
        smiles = _extract_smiles(extracted)
        result.valid_smiles = bool(smiles) and valid_smiles(smiles)
        result.validity_score = 1.0 if result.valid_smiles else 0.0
        if not result.valid_smiles:
            # Still run the verifier below so the reward stays the official
            # one; the status just says why it is going to be zero.
            result.status = ParseStatus.INVALID_SMILES

    try:
        if normalised == "constraint_generation":
            constraints = _loads(constraints_json) if constraints_json else None
            if not constraints:
                result.status = ParseStatus.UNSUPPORTED_TASK
                return result
            score = evaluate_answer(
                task_type="constraint_generation",
                predicted=extracted,
                constraints=constraints,
            )
        else:
            target = _loads(target_json) if target_json else None
            if target is None:
                result.status = ParseStatus.UNSUPPORTED_TASK
                return result
            score = evaluate_answer(
                task_type=normalised, predicted=extracted, target=target
            )
    except Exception:  # noqa: BLE001 - a verifier crash must not kill training
        result.status = ParseStatus.VERIFIER_ERROR
        return result

    if isinstance(score, dict):
        score = score.get("reward", 0.0)
    try:
        correctness = float(score)
    except (TypeError, ValueError):
        result.status = ParseStatus.VERIFIER_ERROR
        return result
    if correctness != correctness or correctness in (float("inf"), float("-inf")):
        result.status = ParseStatus.VERIFIER_ERROR
        return result

    result.correctness = correctness
    if result.status == ParseStatus.EMPTY:
        result.status = ParseStatus.OK
    return result


# ---------------------------------------------------------------------------
# TRL reward callables
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Reward weights and shaping switches, recorded in the experiment config.

    ``correctness_weight`` must dominate: with the defaults a perfectly
    formatted wrong answer scores 0.12 while a scruffy correct one scores at
    least 1.0, so formatting can never buy a rank over being right.
    """

    correctness_weight: float = 1.0
    format_weight: float = 0.1
    validity_weight: float = 0.0
    log_diagnostics: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _BaseReward:
    """Shared scoring + diagnostics for the TRL reward callables.

    All three callables need the same per-completion score, so it is computed
    once per batch by :class:`CorrectnessReward` and cached on a shared holder
    that the others read.
    """

    def __init__(self, cache: "_ScoreCache", config: RewardConfig) -> None:
        self._cache = cache
        self._config = config


@dataclass
class _ScoreCache:
    """Per-batch scores, shared between the reward callables of one trainer."""

    key: tuple[int, ...] = field(default_factory=tuple)
    scores: list[ScoredCompletion] = field(default_factory=list)

    def get(
        self,
        completions: Sequence[Any],
        task_type: Sequence[str],
        target_json: Sequence[str | None],
        constraints_json: Sequence[str | None],
    ) -> list[ScoredCompletion]:
        texts = [completion_to_text(c) for c in completions]
        key = tuple(hash((t, tt)) for t, tt in zip(texts, task_type))
        if key == self.key and len(self.scores) == len(texts):
            return self.scores
        self.scores = [
            score_completion(text, task, target, constraints)
            for text, task, target, constraints in zip(
                texts, task_type, target_json, constraints_json
            )
        ]
        self.key = key
        return self.scores


def _columns(
    n: int, kwargs: dict[str, Any], name: str
) -> list[Any]:
    """Fetch a dataset column TRL repeated across the generation group."""
    column = kwargs.get(name)
    if column is None:
        return [None] * n
    return list(column)


class CorrectnessReward(_BaseReward):
    """Binary official-verifier reward. This is the signal that matters."""

    def __call__(
        self,
        completions: Sequence[Any] | None = None,
        log_metric: Callable[[str, float], None] | None = None,
        log_extra: Callable[[str, list], None] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        completions = completions or []
        n = len(completions)
        task_type = _columns(n, kwargs, "task_type")
        scores = self._cache.get(
            completions,
            task_type,
            _columns(n, kwargs, "target_json"),
            _columns(n, kwargs, "constraints_json"),
        )

        if self._config.log_diagnostics and n:
            self._log(scores, log_metric, log_extra)

        weight = self._config.correctness_weight
        return [weight * s.correctness for s in scores]

    def _log(
        self,
        scores: Sequence[ScoredCompletion],
        log_metric: Callable[[str, float], None] | None,
        log_extra: Callable[[str, list], None] | None,
    ) -> None:
        n = len(scores)
        if log_metric is not None:
            ok = sum(1 for s in scores if s.status == ParseStatus.OK)
            log_metric("parse/failure_fraction", 1.0 - ok / n)
            log_metric(
                "parse/answer_tag_fraction",
                sum(1 for s in scores if s.answer_blocks) / n,
            )
            log_metric(
                "parse/well_formed_fraction",
                sum(1 for s in scores if s.format_status == FormatStatus.OK) / n,
            )
            log_metric(
                "parse/conflicting_answer_fraction",
                sum(1 for s in scores if s.conflicting_answers) / n,
            )
            log_metric(
                "verifier/error_fraction",
                sum(1 for s in scores if s.status == ParseStatus.VERIFIER_ERROR) / n,
            )
            log_metric(
                "reward/correctness_mean", sum(s.correctness for s in scores) / n
            )
            checked = [s for s in scores if s.valid_smiles is not None]
            if checked:
                log_metric(
                    "chem/invalid_smiles_fraction",
                    sum(1 for s in checked if not s.valid_smiles) / len(checked),
                )
        if log_extra is not None:
            log_extra("parse_status", [s.status for s in scores])
            log_extra("format_status", [s.format_status for s in scores])
            log_extra("extracted_answer", [s.extracted for s in scores])


class FormatReward(_BaseReward):
    """Small bonus for producing the requested ``<answer>{...}</answer>`` shape."""

    def __call__(
        self,
        completions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        completions = completions or []
        n = len(completions)
        scores = self._cache.get(
            completions,
            _columns(n, kwargs, "task_type"),
            _columns(n, kwargs, "target_json"),
            _columns(n, kwargs, "constraints_json"),
        )
        weight = self._config.format_weight
        return [weight * s.format_score for s in scores]


class ChemicalValidityReward(_BaseReward):
    """Small bonus for emitting a parseable molecule on generation tasks.

    Returns ``None`` for non-generation samples so TRL drops this term for them
    rather than scoring them zero, which would otherwise bias count/index
    rewards downward for no reason.
    """

    def __call__(
        self,
        completions: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> list[float | None]:
        completions = completions or []
        n = len(completions)
        scores = self._cache.get(
            completions,
            _columns(n, kwargs, "task_type"),
            _columns(n, kwargs, "target_json"),
            _columns(n, kwargs, "constraints_json"),
        )
        weight = self._config.validity_weight
        return [
            None if s.valid_smiles is None else weight * s.validity_score
            for s in scores
        ]


def build_reward_functions(config: RewardConfig) -> list[Callable[..., Any]]:
    """Build the reward callables for one experiment.

    Terms with zero weight are left out entirely so they do not clutter the
    logged metrics with a constant zero column.
    """
    cache = _ScoreCache()
    functions: list[Callable[..., Any]] = [CorrectnessReward(cache, config)]
    if config.format_weight:
        functions.append(FormatReward(cache, config))
    if config.validity_weight:
        functions.append(ChemicalValidityReward(cache, config))
    return functions
