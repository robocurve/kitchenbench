"""Blueprint derivation over all 50 shipped instances + pinned bindings."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from inspect_robots.rollout import derive_seed
from numpy.typing import NDArray

from kitchenbench.sim import (
    ASSETS,
    SIM_CONTRACT_VERSION,
    build_blueprint,
    make_success_checker,
)
from kitchenbench.sim.blueprint import SIZE_ORDER_STEP, SPREAD_CLEARANCE_CM, _size_factors
from kitchenbench.specs import SPEC_BY_KEY
from kitchenbench.tasks import build_scenes, realize_scene

Box = tuple[NDArray[np.float64], NDArray[np.float64]]


class _AnyWorld:
    """Answers every query permissively — only used to construct checkers."""

    def aabb(self, name: str) -> Box:
        return np.zeros(3), np.ones(3)

    def contained_fraction(self, item: str, container: str) -> float:
        return 0.0

    def contained_mass_g(self, substance: str, container: str) -> float:
        return 0.0

    def opening_fraction(self, name: str) -> float:
        return 0.0


def _all_scene_seed_pairs() -> list[tuple[str, object, int]]:
    pairs: list[tuple[str, object, int]] = []
    for spec in SPEC_BY_KEY.values():
        for index, scene in enumerate(build_scenes(spec)):
            for epoch in range(3):
                pairs.append(
                    (f"{spec.key}[{index}]e{epoch}", scene, derive_seed(0, scene.init_seed, epoch))
                )
    return pairs


_PAIRS = _all_scene_seed_pairs()


def test_sweep_covers_all_fifty_instances() -> None:
    assert len(_PAIRS) == 10 * 5 * 3


@pytest.mark.parametrize(("label", "scene", "seed"), _PAIRS, ids=[p[0] for p in _PAIRS])
def test_every_shipped_instance_builds_a_sound_blueprint(
    label: str, scene: object, seed: int
) -> None:
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    assert blueprint.contract_version == SIM_CONTRACT_VERSION

    names = blueprint.object_names()
    assert len(names) == len(set(names)), "object names must be unique"
    for obj in blueprint.objects:
        assert obj.asset in ASSETS, f"asset {obj.asset!r} missing from ASSETS"

    # Required roles: constructing the kind's checker validates them.
    make_success_checker(blueprint, _AnyWorld())

    # Spawnability: no two same-parent objects coincide (within 1 mm).
    by_parent: dict[str | None, list[tuple[float, float]]] = {}
    for obj in blueprint.objects:
        by_parent.setdefault(obj.parent, []).append((obj.x_cm, obj.y_cm))
    for parent, placements in by_parent.items():
        rounded = {(round(x, 1), round(y, 1)) for x, y in placements}
        assert len(rounded) == len(placements), f"coincident placements under {parent!r}: {label}"

    # Rigid copies never interpenetrate at spawn: every expanded family is
    # spaced at least the widest copy's nominal footprint + clearance apart.
    families: dict[str, list[float]] = {}
    widest: dict[str, float] = {}
    for obj in blueprint.objects:
        stem, _, index = obj.name.rpartition("_")
        if stem and index.isdigit():
            families.setdefault(stem, []).append(obj.x_cm)
            widest[stem] = max(widest.get(stem, 0.0), ASSETS[obj.asset].footprint_w_cm)
    for stem, xs in families.items():
        for a, b in pairwise(sorted(xs)):
            assert b - a >= widest[stem] + SPREAD_CLEARANCE_CM - 1e-9, (
                f"copies of {stem!r} interpenetrate at spawn: {label}"
            )

    # Conditions are exactly the annotation's declared list, verbatim values —
    # compared against the ANNOTATION so an empty/mangled dict cannot pass.
    from kitchenbench.tasks import find_instance

    instance = find_instance(scene)  # type: ignore[arg-type]
    assert instance.sim is not None
    assert set(blueprint.conditions) == set(instance.sim.conditions)
    for key, value in blueprint.conditions.items():
        assert blueprint.values[key] == value


def _scene(task_key: str, instance_index: int) -> object:
    return build_scenes(SPEC_BY_KEY[task_key])[instance_index]


def test_pinned_place_cutlery_bindings_match_realization() -> None:
    scene = _scene("place_cutlery", 0)  # spoon-on-plate
    seed = derive_seed(0, 0, 0)
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    values = realize_scene(scene, seed).values  # type: ignore[arg-type]

    cutlery = next(o for o in blueprint.objects if o.name == "cutlery")
    dishware = next(o for o in blueprint.objects if o.name == "dishware")
    assert cutlery.asset == values["cutlery"]
    assert cutlery.x_cm == float(values["cutlery_x_cm"])
    assert cutlery.y_cm == float(values["jitter_y_cm"])
    assert dishware.asset == values["dishware"]
    assert dishware.x_cm == float(values["dishware_x_cm"])
    assert blueprint.roles["item"] == ("cutlery",)
    assert blueprint.roles["surface"] == ("dishware",)
    assert blueprint.instruction == realize_scene(scene, seed).instruction  # type: ignore[arg-type]


def test_pinned_stack_count_expansion_and_spread() -> None:
    scene = _scene("stack", 0)  # stack/cups
    seed = derive_seed(0, 0, 1)
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    values = realize_scene(scene, seed).values  # type: ignore[arg-type]

    stack = [o for o in blueprint.objects if o.role == "stack"]
    assert len(stack) == int(values["count"])
    assert [o.name for o in stack] == [f"cup_{k + 1}" for k in range(len(stack))]
    spacing = {round(b.x_cm - a.x_cm, 6) for a, b in pairwise(stack)}
    # Sampled spread, clamped so 9 cm cups never interpenetrate at spawn.
    expected = max(float(values["spread_cm"]), ASSETS["cup"].footprint_w_cm + SPREAD_CLEARANCE_CM)
    assert spacing == {round(expected, 6)}


def test_pinned_sort_total_count_round_robin() -> None:
    scene = _scene("sort_cutlery", 2)  # overlapping: only total_count sampled
    seed = derive_seed(0, 2, 0)
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    values = realize_scene(scene, seed).values  # type: ignore[arg-type]

    sortables = [o for o in blueprint.objects if o.role == "sortable"]
    assert len(sortables) == int(values["total_count"])
    assets = [o.asset for o in sortables]
    # Deterministic round-robin over the declared classes.
    assert assets == [("spoon", "fork", "knife")[k % 3] for k in range(len(sortables))]
    for cls in ("spoon", "fork", "knife"):
        assert cls in assets  # >=1 of each declared class (total_count >= 6)
    compartments = {o.name for o in blueprint.objects if o.role == "compartment"}
    assert compartments == {"compartment_spoon", "compartment_fork", "compartment_knife"}
    tray = next(o for o in blueprint.objects if o.name == "tray")
    for name in compartments:
        assert next(o for o in blueprint.objects if o.name == name).parent == tray.name


def test_pinned_pour_source_holds_the_pasta() -> None:
    scene = _scene("pour_pasta", 0)  # measuring-cup-to-bowl
    seed = derive_seed(0, 0, 0)
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    values = realize_scene(scene, seed).values  # type: ignore[arg-type]

    source = next(o for o in blueprint.objects if o.role == "source")
    pasta = next(o for o in blueprint.objects if o.role == "substance")
    assert pasta.parent == source.name
    assert pasta.amount_g == float(values["fill_g"])
    assert blueprint.success_params["total_g"] == values["fill_g"]
    assert blueprint.success_params["substance"] == pasta.name
    assert pasta.asset == "dry_pasta"  # via the static, slugged


def test_pinned_seal_lid_offset_binds_to_lid() -> None:
    scene = _scene("seal_container", 3)  # misaligned-lid
    seed = derive_seed(0, 3, 0)
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    values = realize_scene(scene, seed).values  # type: ignore[arg-type]

    lid = next(o for o in blueprint.objects if o.role == "lid")
    container = next(o for o in blueprint.objects if o.role == "container")
    assert lid.x_cm == float(values["lid_offset_cm"])
    assert lid.yaw_deg == float(values["lid_yaw_deg"])
    assert lid.parent == container.name


def test_pinned_scoop_instruction_exposes_target() -> None:
    scene = _scene("scoop_pasta", 0)
    seed = derive_seed(0, 0, 0)
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    values = realize_scene(scene, seed).values  # type: ignore[arg-type]
    assert f"about {float(values['fill_target_g']):.0f} g" in blueprint.instruction
    assert blueprint.success_params["target_g"] == values["fill_target_g"]


def test_pinned_handoff_receiving_arms() -> None:
    # utensil starts left (-x) -> receiving right; cross-body starts right -> left.
    for index, expected in ((0, "right"), (4, "left"), (1, "either")):
        scene = _scene("handoff", index)
        blueprint = build_blueprint(scene, derive_seed(0, index, 0))  # type: ignore[arg-type]
        assert blueprint.success_params["receiving_arm"] == expected


def test_pinned_size_order_mixed_sizes_both_orders() -> None:
    scene = _scene("stack", 3)  # mixed-sizes: size_order sampled
    # These fixed seeds sample "largest_first" and "shuffled" (pinned).
    seed_first = derive_seed(0, 3, 0)
    values_first = realize_scene(scene, seed_first).values  # type: ignore[arg-type]
    assert values_first["size_order"] == "largest_first"
    bp = build_blueprint(scene, seed_first)  # type: ignore[arg-type]
    sizes = [o.size_cm for o in bp.objects if o.role == "stack"]
    assert sizes == sorted(sizes, key=lambda s: -float(s))  # type: ignore[arg-type]
    base = float(sizes[0])  # type: ignore[arg-type]
    assert sizes == [base * f for f in _size_factors(len(sizes), "largest_first")]

    seed_shuffled = derive_seed(0, 3, 1)
    values_shuffled = realize_scene(scene, seed_shuffled).values  # type: ignore[arg-type]
    assert values_shuffled["size_order"] == "shuffled"
    bp2 = build_blueprint(scene, seed_shuffled)  # type: ignore[arg-type]
    sizes2 = [o.size_cm for o in bp2.objects if o.role == "stack"]
    base2 = float(max(s for s in sizes2 if s is not None))
    # The documented deterministic rotation, end to end through _expand.
    assert sizes2 == [base2 * f for f in _size_factors(len(sizes2), "shuffled")]


def test_size_factor_sequences() -> None:
    assert _size_factors(3, "largest_first") == [
        1.0,
        1.0 - SIZE_ORDER_STEP,
        1.0 - 2 * SIZE_ORDER_STEP,
    ]
    shuffled = _size_factors(3, "shuffled")
    assert sorted(shuffled, reverse=True) == _size_factors(3, "largest_first")
    assert shuffled != _size_factors(3, "largest_first")  # the fixed rotation moved something
    assert _size_factors(3, "shuffled") == shuffled  # deterministic


def test_seed_none_guards_to_zero() -> None:
    scene = _scene("place_cutlery", 0)
    assert build_blueprint(scene, None) == build_blueprint(scene, 0)  # type: ignore[arg-type]


def test_unannotated_instance_raises_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = SPEC_BY_KEY["place_cutlery"]
    stripped = replace(spec, instances=tuple(replace(inst, sim=None) for inst in spec.instances))
    monkeypatch.setitem(SPEC_BY_KEY, "place_cutlery", stripped)
    scene = build_scenes(stripped)[0]
    with pytest.raises(LookupError, match="no sim annotation"):
        build_blueprint(scene, 0)


def test_blueprint_is_pure_function_of_seed() -> None:
    scene = _scene("sort_cutlery", 2)
    a = build_blueprint(scene, 1234)  # type: ignore[arg-type]
    b = build_blueprint(scene, 1234)  # type: ignore[arg-type]
    assert a == b
    c = build_blueprint(scene, 1235)  # type: ignore[arg-type]
    assert a != c


def test_size_order_respects_bound_size() -> None:
    """size_cm bound AND size_order set: factors scale the sampled size."""
    from kitchenbench.instances import SimObject, Var
    from kitchenbench.sim.blueprint import _expand, _Resolver
    from kitchenbench.specs import SPEC_BY_KEY

    instance = SPEC_BY_KEY["stack"].instances[0]
    obj = SimObject(
        name="bowl",
        asset="bowl",
        role="stack",
        count=Var("count"),
        spread_cm=Var("spread_cm"),
        size_cm=Var("base_cm"),
        size_order=Var("order"),
    )
    resolver = _Resolver(
        instance, {"count": 3, "spread_cm": 6.0, "base_cm": 20.0, "order": "largest_first"}
    )
    expanded = _expand(obj, resolver)
    assert [o.size_cm for o in expanded] == [20.0, 20.0 * 0.88, 20.0 * 0.76]


def test_size_factors_floored_above_zero() -> None:
    # A large count can never yield a zero/negative size.
    assert all(f >= SIZE_ORDER_STEP for f in _size_factors(20, "largest_first"))


def test_expand_clamps_spread_to_footprint() -> None:
    """A sampled spread narrower than the copies themselves cannot be honored."""
    from kitchenbench.instances import SimObject, Var
    from kitchenbench.sim.blueprint import _expand, _Resolver

    instance = SPEC_BY_KEY["stack"].instances[0]
    obj = SimObject(name="cup", asset="cup", role="stack", count=Var("count"), spread_cm=3.0)
    copies = _expand(obj, _Resolver(instance, {"count": 3}))
    # 9 cm cups at a sampled 3 cm spread -> clamped to 10 cm spacing.
    assert [o.x_cm for o in copies] == [0.0, 10.0, 20.0]

    # A spread wider than the footprint passes through unchanged.
    wide = SimObject(name="cup", asset="cup", role="stack", count=Var("count"), spread_cm=14.0)
    copies = _expand(wide, _Resolver(instance, {"count": 2}))
    assert [o.x_cm for o in copies] == [0.0, 14.0]


def test_expand_count_below_one_raises() -> None:
    from kitchenbench.instances import SimObject, Var
    from kitchenbench.sim.blueprint import _expand, _Resolver

    instance = SPEC_BY_KEY["stack"].instances[0]
    obj = SimObject(name="cup", asset="cup", count=Var("count"))
    with pytest.raises(ValueError, match="counts must be >= 1"):
        _expand(obj, _Resolver(instance, {"count": 0}))


def test_var_count_resolving_to_one_still_numbers() -> None:
    """Sampled counts number their copies even at 1: names stay stable across epochs."""
    from kitchenbench.instances import SimObject, Var
    from kitchenbench.sim.blueprint import _expand, _Resolver

    instance = SPEC_BY_KEY["stack"].instances[0]
    obj = SimObject(
        name="cutlery", asset="cutlery", role="sortable", count=Var("count"), split=("spoon",)
    )
    (only,) = _expand(obj, _Resolver(instance, {"count": 1}))
    assert only.name == "cutlery_1"
    assert only.asset == "spoon"  # split applies to the numbered copy too

    # A literal count of 1 keeps the bare annotation name.
    (bare,) = _expand(SimObject(name="fork", asset="fork"), _Resolver(instance, {}))
    assert bare.name == "fork"
    assert bare.asset == "fork"


def test_pinned_conditions_survive_verbatim() -> None:
    scene = _scene("place_cutlery", 1)  # from-drawer: approach_angle_deg condition
    seed = derive_seed(0, 1, 0)
    blueprint = build_blueprint(scene, seed)  # type: ignore[arg-type]
    values = realize_scene(scene, seed).values  # type: ignore[arg-type]
    assert dict(blueprint.conditions) == {"approach_angle_deg": values["approach_angle_deg"]}
