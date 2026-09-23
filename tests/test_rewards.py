"""Adversarial tests for the GRPO reward.

The reward is the only thing standing between "the policy learned chemistry"
and "the policy learned to exploit the scorer". Each test below is a specific
way that could go wrong.
"""

from __future__ import annotations

import json

import pytest

from miqgrpo.rewards import (
    FormatStatus,
    ParseStatus,
    RewardConfig,
    build_reward_functions,
    score_completion,
)

COUNT_TARGET = json.dumps({"ring_count": 2})
INDEX_TARGET = json.dumps({"ring_index": [0, 1, 2, 3, 4, 5]})
CONSTRAINTS = json.dumps(
    [{"type": "aromatic_ring_count", "operator": "=", "value": 1}]
)


def correctness(text: str, task_type: str, target=None, constraints=None) -> float:
    return score_completion(text, task_type, target, constraints).correctness


# --- 1-4: symbolic correctness --------------------------------------------


def test_correct_count_scores_one():
    assert correctness('<answer>{"ring_count": 2}</answer>', "single_count", COUNT_TARGET) == 1.0


def test_wrong_count_scores_zero():
    assert correctness('<answer>{"ring_count": 3}</answer>', "single_count", COUNT_TARGET) == 0.0


def test_correct_indices_score_one():
    text = '<answer>{"ring_index": [0, 1, 2, 3, 4, 5]}</answer>'
    assert correctness(text, "single_index", INDEX_TARGET) == 1.0


def test_off_by_one_indices_score_zero():
    """The classic index bug: a shifted list must not be accepted."""
    text = '<answer>{"ring_index": [1, 2, 3, 4, 5, 6]}</answer>'
    assert correctness(text, "single_index", INDEX_TARGET) == 0.0


def test_index_order_does_not_matter():
    """Official semantics compare sets, so a permutation is still correct."""
    text = '<answer>{"ring_index": [5, 4, 3, 2, 1, 0]}</answer>'
    assert correctness(text, "single_index", INDEX_TARGET) == 1.0


# --- 5-7: degenerate and malformed output ---------------------------------


def test_empty_completion():
    result = score_completion("", "single_count", COUNT_TARGET, None)
    assert result.correctness == 0.0
    assert result.status == ParseStatus.EMPTY
    assert result.format_score == 0.0


def test_whitespace_only_completion():
    result = score_completion("   \n\t ", "single_count", COUNT_TARGET, None)
    assert result.correctness == 0.0
    assert result.status == ParseStatus.EMPTY


def test_malformed_structured_output_does_not_crash():
    text = '<answer>{"ring_count": </answer>'
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.correctness == 0.0
    assert result.format_status == FormatStatus.INVALID_JSON


def test_conflicting_answers_use_official_last_block_rule():
    """Two blocks: the official extractor reads the last one, so we do too."""
    text = '<answer>{"ring_count": 2}</answer> ... <answer>{"ring_count": 7}</answer>'
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.conflicting_answers is True
    assert result.correctness == 0.0
    assert result.format_status == FormatStatus.CONFLICTING_ANSWER_TAGS


def test_duplicate_identical_answers_are_not_conflicting():
    text = '<answer>{"ring_count": 2}</answer> <answer>{"ring_count": 2}</answer>'
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.conflicting_answers is False
    assert result.correctness == 1.0


# --- 8-10: constrained generation -----------------------------------------


def test_invalid_smiles_scores_zero():
    text = '<answer>{"smiles": "C1CC"}</answer>'
    result = score_completion(text, "constraint_generation", None, CONSTRAINTS)
    assert result.correctness == 0.0
    assert result.valid_smiles is False
    assert result.status == ParseStatus.INVALID_SMILES


def test_valid_smiles_violating_the_constraint_scores_zero():
    """Chemically fine, but it does not satisfy what was asked."""
    text = '<answer>{"smiles": "CCO"}</answer>'
    result = score_completion(text, "constraint_generation", None, CONSTRAINTS)
    assert result.valid_smiles is True
    assert result.correctness == 0.0


def test_any_molecule_satisfying_the_constraints_is_correct():
    """Correctness is constraint satisfaction, never equality to one reference."""
    for smiles in ("c1ccccc1", "Cc1ccccc1", "OCc1ccccc1O", "c1ccc(Cl)cc1"):
        text = f'<answer>{{"smiles": "{smiles}"}}</answer>'
        assert correctness(text, "constraint_generation", None, CONSTRAINTS) == 1.0, smiles


def test_multi_constraint_requires_all_constraints():
    constraints = json.dumps(
        [
            {"type": "aromatic_ring_count", "operator": "=", "value": 1},
            {"type": "halogen_atom_count", "operator": "=", "value": 1},
        ]
    )
    assert correctness('<answer>{"smiles": "c1ccccc1"}</answer>', "constraint_generation", None, constraints) == 0.0
    assert correctness('<answer>{"smiles": "Clc1ccccc1"}</answer>', "constraint_generation", None, constraints) == 1.0


# --- 11: formatting must never beat chemistry -----------------------------


def test_pretty_formatting_with_wrong_chemistry_loses_to_ugly_correct():
    config = RewardConfig()
    pretty_wrong = score_completion(
        '<answer>{"ring_count": 99}</answer>', "single_count", COUNT_TARGET, None
    )
    ugly_right = score_completion(
        'the answer is {"ring_count": 2}', "single_count", COUNT_TARGET, None
    )

    def total(result):
        return (
            config.correctness_weight * result.correctness
            + config.format_weight * result.format_score
        )

    assert pretty_wrong.format_score > ugly_right.format_score
    assert total(ugly_right) > total(pretty_wrong)


def test_shape_reward_is_graded_not_binary():
    """Early training needs reward variance; a binary shape term gives none."""
    scores = [
        score_completion(text, "single_count", COUNT_TARGET, None).format_score
        for text in (
            "there are some rings",
            '{"wrong_key": 2}',
            '{"ring_count": 2}',
            '<answer>{"ring_count": 2}</answer>',
        )
    ]
    assert scores == sorted(scores)
    assert len(set(scores)) >= 3


def test_shape_reward_never_exceeds_one():
    for text in (
        '<answer>{"ring_count": 2}</answer>',
        '<answer>{"ring_count": 2}</answer><answer>{"ring_count": 2}</answer>',
        '{"ring_count": 2}',
    ):
        assert 0.0 <= score_completion(text, "single_count", COUNT_TARGET, None).format_score <= 1.0


# --- 12: verifier robustness ----------------------------------------------


def test_verifier_exception_is_contained(monkeypatch):
    import miqgrpo.rewards as rewards

    def boom(*args, **kwargs):
        raise RuntimeError("verifier exploded")

    monkeypatch.setattr(rewards, "evaluate_answer", boom)
    result = rewards.score_completion(
        '<answer>{"ring_count": 2}</answer>', "single_count", COUNT_TARGET, None
    )
    assert result.correctness == 0.0
    assert result.status == ParseStatus.VERIFIER_ERROR


def test_unknown_task_type_is_reported_not_raised():
    result = score_completion('<answer>{"x": 1}</answer>', "elephant", COUNT_TARGET, None)
    assert result.status == ParseStatus.UNSUPPORTED_TASK
    assert result.correctness == 0.0


def test_missing_target_is_reported_not_raised():
    result = score_completion('<answer>{"ring_count": 2}</answer>', "single_count", None, None)
    assert result.correctness == 0.0
    assert result.status == ParseStatus.UNSUPPORTED_TASK


# --- fuzzing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '  <answer>{"ring_count": 2}</answer>  ',
        '<answer>\n{"ring_count": 2}\n</answer>',
        '```json\n<answer>{"ring_count": 2}</answer>\n```',
        'Let me think...\n<think>hmm</think>\n<answer>{"ring_count": 2}</answer>',
        'blah blah <answer>{"ring_count": 2}</answer> trailing prose',
        '<answer>{"ring_count" : 2}</answer>',
        '<answer>{"ring_count": 2.0}</answer>',
    ],
)
def test_correct_answer_survives_surrounding_noise(text):
    assert correctness(text, "single_count", COUNT_TARGET) == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "\x00\x01\x02",
        "<answer>" * 500,
        '<answer>{"ring_count": ' + "9" * 5000 + "}</answer>",
        "𝕬𝖓𝖘𝖜𝖊𝖗: two",
        '<answer>{"ring_count": 2}',  # truncated mid-tag
        "</answer>{\"ring_count\": 2}<answer>",
    ],
)
def test_hostile_input_produces_one_finite_reward(text):
    result = score_completion(text, "single_count", COUNT_TARGET, None)
    assert result.correctness in (0.0, 1.0)
    assert 0.0 <= result.format_score <= 1.0


# --- TRL integration -------------------------------------------------------


def _kwargs(n, task_type, target=None, constraints=None):
    return {
        "task_type": [task_type] * n,
        "target_json": [target] * n,
        "constraints_json": [constraints] * n,
    }


def test_reward_functions_return_one_value_per_completion():
    funcs = build_reward_functions(RewardConfig(format_weight=0.1, validity_weight=0.05))
    completions = [
        [{"role": "assistant", "content": '<answer>{"ring_count": 2}</answer>'}],
        [{"role": "assistant", "content": "nonsense"}],
        [{"role": "assistant", "content": '<answer>{"ring_count": 3}</answer>'}],
    ]
    for func in funcs:
        values = func(completions=completions, **_kwargs(3, "single_count", COUNT_TARGET))
        assert len(values) == 3
        assert all(v is None or isinstance(v, float) for v in values)


def test_validity_reward_is_none_for_non_generation_tasks():
    """Returning None makes TRL drop the term rather than score it zero."""
    funcs = build_reward_functions(RewardConfig(validity_weight=0.05))
    validity = funcs[-1]
    completions = [[{"role": "assistant", "content": '<answer>{"ring_count": 2}</answer>'}]]
    assert validity(completions=completions, **_kwargs(1, "single_count", COUNT_TARGET)) == [None]


def test_correctness_reward_matches_string_and_conversational_completions():
    funcs = build_reward_functions(RewardConfig(format_weight=0.0))
    text = '<answer>{"ring_count": 2}</answer>'
    as_string = funcs[0](completions=[text], **_kwargs(1, "single_count", COUNT_TARGET))
    as_messages = funcs[0](
        completions=[[{"role": "assistant", "content": text}]],
        **_kwargs(1, "single_count", COUNT_TARGET),
    )
    assert as_string == as_messages == [1.0]


def test_score_cache_does_not_leak_across_different_completions():
    """A cache keyed loosely could hand one completion another's reward."""
    funcs = build_reward_functions(RewardConfig(format_weight=0.0))
    right = '<answer>{"ring_count": 2}</answer>'
    wrong = '<answer>{"ring_count": 8}</answer>'
    assert funcs[0](completions=[right], **_kwargs(1, "single_count", COUNT_TARGET)) == [1.0]
    assert funcs[0](completions=[wrong], **_kwargs(1, "single_count", COUNT_TARGET)) == [0.0]
    assert funcs[0](completions=[right], **_kwargs(1, "single_count", COUNT_TARGET)) == [1.0]


def test_log_metric_hook_is_called():
    funcs = build_reward_functions(RewardConfig())
    seen: dict[str, float] = {}
    funcs[0](
        completions=[[{"role": "assistant", "content": '<answer>{"ring_count": 2}</answer>'}]],
        log_metric=lambda name, value: seen.__setitem__(name, value),
        **_kwargs(1, "single_count", COUNT_TARGET),
    )
    assert "reward/correctness_mean" in seen
    assert "parse/failure_fraction" in seen
