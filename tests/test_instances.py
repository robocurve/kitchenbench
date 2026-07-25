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

    # TaskInstance: equal objects must hash equal and support dict lookup
    inst_a = _instance()
    inst_b = _instance()
    assert inst_a == inst_b
    assert hash(inst_a) == hash(inst_b)
    assert {inst_a: "x"}[inst_b] == "x"

    # Realization: equal objects must hash equal and support dict lookup
    real_a = inst_a.realize(0)
    real_b = inst_b.realize(0)
    assert real_a == real_b
    assert hash(real_a) == hash(real_b)
    assert {real_a: "y"}[real_b] == "y"

    # TaskSpec: equal objects must hash equal and support dict lookup
    spec_a = SPECS[0]
    spec_b = SPECS[0]
    assert spec_a == spec_b
    assert hash(spec_a) == hash(spec_b)
    assert {spec_a: "z"}[spec_b] == "z"

    # Spec-wide invariant: every instance across SPECS must be hashable
    for spec in SPECS:
        for instance in spec.instances:
            assert hash(instance) is not None
