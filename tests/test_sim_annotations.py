"""The SimSpec annotation model: structural validation at instance construction."""

from __future__ import annotations

import pytest

from kitchenbench.distributions import Categorical, Uniform
from kitchenbench.instances import SimObject, SimSpec, TaskInstance, Var


def _instance(sim: SimSpec | None) -> TaskInstance:
    return TaskInstance(
        instance_id="test/example",
        goal="place the {thing} on the plate",
        setup={
            "thing": Categorical(("spoon", "fork")),
            "thing_x_cm": Uniform(-10, 10),
            "grip": Uniform(1, 5),
        },
        language_vars=("thing",),
        target_kind="place_on",
        static={"substance": "dry_pasta"},
        sim=sim,
    )


def _valid_spec() -> SimSpec:
    return SimSpec(
        objects=(
            SimObject(name="thing", asset=Var("thing"), role="item", x_cm=Var("thing_x_cm")),
            SimObject(name="plate", asset="plate", role="surface", x_cm=10.0),
        ),
        conditions=("grip",),
    )


def test_valid_annotation_constructs() -> None:
    assert _instance(_valid_spec()).sim is not None


def test_unannotated_instance_stays_legal() -> None:
    assert _instance(None).sim is None


def test_uncovered_setup_variable_rejected() -> None:
    spec = SimSpec(objects=_valid_spec().objects)  # 'grip' not covered anywhere
    with pytest.raises(ValueError, match=r"not covered.*grip"):
        _instance(spec)


def test_unknown_var_rejected() -> None:
    spec = SimSpec(
        objects=(SimObject(name="thing", asset=Var("nope"), role="item"),),
        conditions=("thing", "thing_x_cm", "grip"),
    )
    with pytest.raises(ValueError, match=r"unknown variable 'nope'"):
        _instance(spec)


def test_var_bound_by_two_object_fields_rejected() -> None:
    spec = SimSpec(
        objects=(
            SimObject(name="a", asset=Var("thing"), x_cm=Var("thing_x_cm")),
            SimObject(name="b", asset="plate", y_cm=Var("thing_x_cm")),
        ),
        conditions=("grip",),
    )
    with pytest.raises(ValueError, match="more than one object field"):
        _instance(spec)


def test_var_shared_between_object_and_success_is_allowed() -> None:
    # pour's fill_g pattern: spawned amount AND success total.
    spec = SimSpec(
        objects=(
            SimObject(name="thing", asset=Var("thing"), role="item", amount_g=Var("thing_x_cm")),
        ),
        success=(("total_g", Var("thing_x_cm")), ("substance", Var("substance"))),
        conditions=("grip",),
    )
    assert _instance(spec).sim is spec


def test_duplicate_object_name_rejected() -> None:
    spec = SimSpec(
        objects=(
            SimObject(name="thing", asset=Var("thing")),
            SimObject(name="thing", asset="plate", x_cm=Var("thing_x_cm")),
        ),
        conditions=("grip",),
    )
    with pytest.raises(ValueError, match="duplicate sim object name"):
        _instance(spec)


def test_unknown_parent_rejected() -> None:
    spec = SimSpec(
        objects=(
            SimObject(name="thing", asset=Var("thing"), parent="ghost", x_cm=Var("thing_x_cm")),
        ),
        conditions=("grip",),
    )
    with pytest.raises(ValueError, match=r"unknown parent 'ghost'"):
        _instance(spec)


def test_reserved_parent_allowed() -> None:
    spec = SimSpec(
        objects=(
            SimObject(name="thing", asset=Var("thing"), parent="bench", x_cm=Var("thing_x_cm")),
        ),
        conditions=("grip",),
    )
    assert _instance(spec).sim is spec


def test_split_without_count_rejected() -> None:
    spec = SimSpec(
        objects=(
            SimObject(name="thing", asset=Var("thing"), split=("spoon", "fork")),
        ),
        conditions=("thing_x_cm", "grip"),
    )
    with pytest.raises(ValueError, match="split but no count"):
        _instance(spec)


def test_success_referencing_unknown_var_rejected() -> None:
    spec = SimSpec(
        objects=_valid_spec().objects,
        success=(("target_g", Var("missing")),),
        conditions=("grip",),
    )
    with pytest.raises(ValueError, match="success references unknown variable"):
        _instance(spec)


def test_condition_must_be_a_setup_variable() -> None:
    spec = SimSpec(objects=_valid_spec().objects, conditions=("grip", "substance"))
    with pytest.raises(ValueError, match=r"condition 'substance' is not a setup variable"):
        _instance(spec)
