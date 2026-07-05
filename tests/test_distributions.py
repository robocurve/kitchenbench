"""Tests for the task-instance distribution types."""

from __future__ import annotations

import json

import numpy as np
import pytest

from kitchenbench.distributions import Categorical, Constant, Normal, Uniform


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_uniform_within_support_and_builtin_float() -> None:
    d = Uniform(8, 15)
    for s in range(20):
        v = d.sample(_rng(s))
        assert 8.0 <= v <= 15.0
        assert type(v) is float
    assert d.describe() == "Uniform[8, 15]"


def test_uniform_golden_determinism() -> None:
    a = Uniform(0.0, 1.0).sample(_rng(123))
    b = Uniform(0.0, 1.0).sample(_rng(123))
    assert a == b  # same seed -> identical draw


def test_categorical_preserves_int_dtype() -> None:
    d = Categorical((12, 14, 16))
    v = d.sample(_rng(1))
    assert v in (12, 14, 16)
    assert type(v) is int  # NOT np.str_ — the index-sampling guarantee
    assert d.describe() == "Categorical({12, 14, 16})"


def test_categorical_strings() -> None:
    d = Categorical(("bowl", "cup", "pot"))
    assert d.sample(_rng(2)) in ("bowl", "cup", "pot")
    assert d.describe() == "Categorical({bowl, cup, pot})"


def test_categorical_weighted_branch() -> None:
    # Heavily weight the last value; over a batch it should dominate.
    d = Categorical(("a", "b", "c"), weights=(0.0, 0.0, 1.0))
    draws = {d.sample(_rng(s)) for s in range(10)}
    assert draws == {"c"}


def test_categorical_variation_over_batch() -> None:
    d = Categorical(("bowl", "cup", "pot"))
    draws = {d.sample(_rng(s)) for s in range(20)}
    assert len(draws) > 1  # batch variation, not flaky pairwise


def test_categorical_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        Categorical(())


def test_categorical_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="match values"):
        Categorical(("a", "b"), weights=(1.0,))


def test_categorical_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Categorical(("a", "b"), weights=(1.0, -0.5))


def test_categorical_rejects_zero_sum_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        Categorical(("a", "b"), weights=(0.0, 0.0))


def test_normal_builtin_float_and_describe() -> None:
    d = Normal(0.0, 3.0)
    v = d.sample(_rng(0))
    assert type(v) is float
    assert d.describe() == "N(0, 3²)"


def test_constant_returns_value() -> None:
    d = Constant("dry_pasta")
    assert d.sample(_rng(5)) == "dry_pasta"
    assert d.describe() == "'dry_pasta'"


def test_samples_are_json_native() -> None:
    rng = _rng(7)
    payload = {
        "u": Uniform(0, 1).sample(rng),
        "c_int": Categorical((1, 2, 3)).sample(rng),
        "c_str": Categorical(("x", "y")).sample(rng),
        "n": Normal(0, 1).sample(rng),
        "k": Constant(5).sample(rng),
    }
    json.dumps(payload)  # must not raise
