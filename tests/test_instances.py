"""Tests for TaskInstance / Realization / Validation."""

from __future__ import annotations

import json

import pytest

from kitchenbench.distributions import Categorical, Normal, Uniform
from kitchenbench.instances import K_EXPERTS, Realization, TaskInstance, Validation


def _instance() -> TaskInstance:
    return TaskInstance(
        instance_id="demo/pour",
        goal="pour the dry pasta into the {vessel}",
        setup={
            "vessel": Categorical(("bowl", "cup", "pot")),
            "fill_g": Uniform(80, 200),
            "jitter_x": Normal(0.0, 3.0),
        },
        language_vars=("vessel",),
        target_kind="pour_into",
        static={"substance": "dry_pasta"},
    )


def test_realize_is_deterministic() -> None:
    inst = _instance()
    a = inst.realize(42)
    b = inst.realize(42)
    assert a == b
    assert isinstance(a, Realization)


def test_realize_fills_only_language_vars() -> None:
    r = _instance().realize(0)
    assert r.instruction.startswith("pour the dry pasta into the ")
    assert r.values["vessel"] in ("bowl", "cup", "pot")
    assert set(r.values) == {"vessel", "fill_g", "jitter_x"}


def test_realize_values_are_json_native() -> None:
    json.dumps(_instance().realize(3).values)  # must not raise


def test_realize_varies_over_seed_batch() -> None:
    inst = _instance()
    fills = {round(inst.realize(s).values["fill_g"], 6) for s in range(20)}
    assert len(fills) > 1  # batch variation, not flaky pairwise


def test_setup_spec_strings() -> None:
    spec = _instance().setup_spec()
    assert spec["vessel"] == "Categorical({bowl, cup, pot})"
    assert spec["fill_g"] == "Uniform[80, 200]"
    assert spec["jitter_x"] == "N(0, 3²)"


def test_setup_lines_sorted() -> None:
    lines = _instance().realize(1).setup_lines
    keys = [line.split(" = ")[0] for line in lines]
    assert keys == sorted(keys) == ["fill_g", "jitter_x", "vessel"]


def test_goal_placeholder_missing_from_language_vars_raises() -> None:
    bad = TaskInstance(
        instance_id="bad",
        goal="put the {item} down",
        setup={"item": Categorical(("a", "b"))},
        target_kind="x",
        language_vars=(),  # forgot to declare {item}
    )
    with pytest.raises(KeyError):
        bad.realize(0)


def test_validation_defaults_not_validated() -> None:
    assert Validation().validated is False
    assert Validation().source == "opus-draft"


def test_validation_accept_rule() -> None:
    ok = Validation(
        representativeness=(4, 5, 4),
        quality=(5, 4, 4),
    )
    assert ok.validated is True


def test_validation_rejects_low_score() -> None:
    low = Validation(representativeness=(4, 3, 5), quality=(5, 5, 5))
    assert low.validated is False


def test_validation_rejects_too_few_experts() -> None:
    few = Validation(representativeness=(5,) * (K_EXPERTS - 1), quality=(5,) * (K_EXPERTS - 1))
    assert few.validated is False


def test_instances_and_realizations_are_hashable() -> None:
    from kitchenbench.specs import SPECS

    inst = _instance()
    # TaskInstance should be hashable
    assert hash(inst) is not None
    s_inst = {inst}
    assert inst in s_inst

    # Realization should be hashable
    real = inst.realize(0)
    assert hash(real) is not None
    s_real = {real}
    assert real in s_real

    # TaskSpec should also be hashable
    spec = SPECS[0]
    assert hash(spec) is not None
    s_spec = {spec}
    assert spec in s_spec
