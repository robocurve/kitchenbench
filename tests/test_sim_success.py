"""Per-target-kind success criteria against a fake world (no physics)."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from numpy.typing import NDArray

from kitchenbench.distributions import Scalar
from kitchenbench.sim import (
    CONTAIN_FRACTION,
    SIM_CONTRACT_VERSION,
    SceneBlueprint,
    SceneObject,
    make_success_checker,
)

Box = tuple[NDArray[np.float64], NDArray[np.float64]]


def _box(x: float, y: float, z: float, w: float, d: float, h: float) -> Box:
    """AABB from center-bottom (x, y, z) and extents (w, d, h), meters."""
    low = np.array([x - w / 2, y - d / 2, z], dtype=np.float64)
    high = np.array([x + w / 2, y + d / 2, z + h], dtype=np.float64)
    return low, high


class FakeWorld:
    def __init__(
        self,
        boxes: dict[str, Box] | None = None,
        fractions: dict[tuple[str, str], float] | None = None,
        masses: dict[tuple[str, str], float] | None = None,
        openings: dict[str, float] | None = None,
    ) -> None:
        self.boxes = boxes or {}
        self.fractions = fractions or {}
        self.masses = masses or {}
        self.openings = openings or {}

    def aabb(self, name: str) -> Box:
        return self.boxes[name]

    def contained_fraction(self, item: str, container: str) -> float:
        return self.fractions[(item, container)]

    def contained_mass_g(self, substance: str, container: str) -> float:
        return self.masses[(substance, container)]

    def opening_fraction(self, name: str) -> float:
        return self.openings[name]


def _blueprint(
    kind: str,
    objects: tuple[SceneObject, ...],
    success: dict[str, Scalar] | None = None,
) -> SceneBlueprint:
    roles: dict[str, list[str]] = {}
    for obj in objects:
        if obj.role is not None:
            roles.setdefault(obj.role, []).append(obj.name)
    return SceneBlueprint(
        contract_version=SIM_CONTRACT_VERSION,
        scene_id="test-scene",
        instruction="do the thing",
        target_kind=kind,
        objects=objects,
        roles=MappingProxyType({k: tuple(v) for k, v in roles.items()}),
        success_params=MappingProxyType(success or {}),
        conditions=MappingProxyType({}),
        values=MappingProxyType({}),
    )


def _obj(name: str, asset: str | None = None, role: str | None = None) -> SceneObject:
    return SceneObject(name=name, asset=asset or name, role=role, x_cm=0.0, y_cm=0.0, yaw_deg=0.0)


# --------------------------------------------------------------------------- #
# place_on
# --------------------------------------------------------------------------- #
def _place_on_bp() -> SceneBlueprint:
    return _blueprint("place_on", (_obj("spoon", role="item"), _obj("plate", role="surface")))


def test_place_on_success_and_failures() -> None:
    plate = _box(0, 0, 0, 0.26, 0.26, 0.03)
    checker = make_success_checker(_place_on_bp(), FakeWorld())
    on = FakeWorld({"spoon": _box(0.02, 0, 0.03, 0.04, 0.18, 0.02), "plate": plate})
    assert checker(on).success

    beside = FakeWorld({"spoon": _box(0.30, 0, 0.0, 0.04, 0.18, 0.02), "plate": plate})
    assert not checker(beside).success
    assert "not over" in checker(beside).explanation

    hovering = FakeWorld({"spoon": _box(0, 0, 0.10, 0.04, 0.18, 0.02), "plate": plate})
    assert not checker(hovering).success
    assert "not resting" in checker(hovering).explanation


def test_place_on_boundary_contact_tolerance() -> None:
    plate = _box(0, 0, 0, 0.26, 0.26, 0.03)
    checker = make_success_checker(_place_on_bp(), FakeWorld())
    just_within = FakeWorld({"spoon": _box(0, 0, 0.0499, 0.04, 0.18, 0.02), "plate": plate})
    assert checker(just_within).success  # gap 0.0199 < CONTACT_TOL_M
    just_beyond = FakeWorld({"spoon": _box(0, 0, 0.0501, 0.04, 0.18, 0.02), "plate": plate})
    assert not checker(just_beyond).success  # gap 0.0201 > CONTACT_TOL_M


def test_missing_object_fails_soft() -> None:
    checker = make_success_checker(_place_on_bp(), FakeWorld())
    verdict = checker(FakeWorld({"plate": _box(0, 0, 0, 0.26, 0.26, 0.03)}))
    assert not verdict.success
    assert "unknown object 'spoon'" in verdict.explanation


# --------------------------------------------------------------------------- #
# stack (nesting-aware)
# --------------------------------------------------------------------------- #
def _stack_bp() -> SceneBlueprint:
    return _blueprint("stack", (_obj("cup_1", "cup", "stack"), _obj("cup_2", "cup", "stack")))


def test_stack_nested_cups_pass() -> None:
    # cup_2's bottom sits INSIDE cup_1 (below its rim): nesting must count.
    world = FakeWorld(
        {"cup_1": _box(0, 0, 0, 0.09, 0.09, 0.10), "cup_2": _box(0, 0, 0.06, 0.09, 0.09, 0.10)}
    )
    assert make_success_checker(_stack_bp(), world)(world).success


def test_stack_plates_on_top_pass_and_side_by_side_fail() -> None:
    stacked = FakeWorld(
        {"cup_1": _box(0, 0, 0, 0.09, 0.09, 0.10), "cup_2": _box(0, 0, 0.11, 0.09, 0.09, 0.10)}
    )
    assert make_success_checker(_stack_bp(), stacked)(stacked).success

    apart = FakeWorld(
        {"cup_1": _box(0, 0, 0, 0.09, 0.09, 0.10), "cup_2": _box(0.2, 0, 0, 0.09, 0.09, 0.10)}
    )
    verdict = make_success_checker(_stack_bp(), apart)(apart)
    assert not verdict.success


def test_stack_right_height_but_laterally_offset_fails() -> None:
    # Correct stacking height, 20 cm to the side: the xy-overlap guard
    # (not the z-interval check) must reject this.
    offset = FakeWorld(
        {"cup_1": _box(0, 0, 0, 0.09, 0.09, 0.10), "cup_2": _box(0.2, 0, 0.11, 0.09, 0.09, 0.10)}
    )
    verdict = make_success_checker(_stack_bp(), offset)(offset)
    assert not verdict.success
    assert "not over" in verdict.explanation


def test_stack_floating_fails() -> None:
    floating = FakeWorld(
        {"cup_1": _box(0, 0, 0, 0.09, 0.09, 0.10), "cup_2": _box(0, 0, 0.20, 0.09, 0.09, 0.10)}
    )
    assert not make_success_checker(_stack_bp(), floating)(floating).success


# --------------------------------------------------------------------------- #
# place_in / pour_into / open / scoop_transfer (query-backed kinds)
# --------------------------------------------------------------------------- #
def test_place_in_threshold() -> None:
    bp = _blueprint("place_in", (_obj("plate", role="item"), _obj("rack", "dish_rack", "rack")))
    checker = make_success_checker(bp, FakeWorld())
    assert checker(FakeWorld(fractions={("plate", "rack"): CONTAIN_FRACTION})).success
    assert not checker(FakeWorld(fractions={("plate", "rack"): 0.79})).success


def test_pour_into_uses_mass_against_total() -> None:
    bp = _blueprint(
        "pour_into",
        (_obj("bowl", role="vessel"),),
        {"substance": "pasta", "total_g": 200.0},
    )
    checker = make_success_checker(bp, FakeWorld())
    assert checker(FakeWorld(masses={("pasta", "bowl"): 160.0})).success  # 80% arrives
    assert not checker(FakeWorld(masses={("pasta", "bowl"): 159.0})).success


def test_open_threshold() -> None:
    bp = _blueprint("open", (_obj("jar", role="container"),))
    checker = make_success_checker(bp, FakeWorld())
    assert checker(FakeWorld(openings={"jar": 0.7})).success
    assert not checker(FakeWorld(openings={"jar": 0.69})).success


def test_scoop_transfer_targets_amount_not_pile() -> None:
    bp = _blueprint(
        "scoop_transfer",
        (_obj("container", role="container"),),
        {"substance": "pile", "target_g": 80.0, "tol_g": 10.0},
    )
    checker = make_success_checker(bp, FakeWorld())
    assert checker(FakeWorld(masses={("pile", "container"): 88.0})).success
    # Dumping the whole pile must FAIL: the criterion is the target amount.
    assert not checker(FakeWorld(masses={("pile", "container"): 400.0})).success
    assert not checker(FakeWorld(masses={("pile", "container"): 69.0})).success


def test_scoop_transfer_default_tolerance() -> None:
    bp = _blueprint(
        "scoop_transfer",
        (_obj("container", role="container"),),
        {"substance": "pile", "target_g": 80.0},
    )
    checker = make_success_checker(bp, FakeWorld())
    assert checker(FakeWorld(masses={("pile", "container"): 94.0})).success  # SCOOP_TOL_G=15
    assert not checker(FakeWorld(masses={("pile", "container"): 96.0})).success


# --------------------------------------------------------------------------- #
# fold: initial-state capture + nominal baseline
# --------------------------------------------------------------------------- #
def test_fold_captures_initial_footprint() -> None:
    bp = _blueprint("fold", (_obj("cloth", role="cloth"),), {"fold_count": 1})
    flat = FakeWorld({"cloth": _box(0, 0, 0, 0.4, 0.4, 0.01)})
    checker = make_success_checker(bp, flat)
    assert not checker(flat).success  # unfolded: area ratio 1.0
    folded = FakeWorld({"cloth": _box(0, 0, 0, 0.4, 0.2, 0.02)})
    assert checker(folded).success  # half the footprint <= 0.6 ratio

    # Capture matters: constructing against the folded state moves the baseline.
    late_checker = make_success_checker(bp, folded)
    assert not late_checker(folded).success


def test_fold_nominal_baseline_for_crumpled_start() -> None:
    bp = _blueprint(
        "fold",
        (SceneObject("cloth", "dish_towel", "cloth", 0.0, 0.0, 0.0),),
        {"fold_count": 1, "baseline": "nominal"},
    )
    # dish_towel nominal 45x65 cm -> baseline 0.2925 m2; threshold 0.6x.
    crumpled = FakeWorld({"cloth": _box(0, 0, 0, 0.3, 0.35, 0.05)})
    checker = make_success_checker(bp, crumpled)
    # The crumpled INITIAL state has a small footprint but is 5 cm tall:
    # success must NOT fire at t=0 (the flatness gate, not the area, rejects).
    verdict = checker(crumpled)
    assert not verdict.success
    assert "not lying flat" in verdict.explanation
    folded_flat = FakeWorld({"cloth": _box(0, 0, 0, 0.3, 0.35, 0.02)})
    assert checker(folded_flat).success  # 0.105 <= 0.1755, and flat
    spread_out = FakeWorld({"cloth": _box(0, 0, 0, 0.45, 0.65, 0.01)})
    assert not checker(spread_out).success


def test_fold_lifted_cloth_does_not_count() -> None:
    # A grasped, hanging cloth has a tiny footprint by definition — the
    # flatness gate must reject it, or every fold trial "succeeds" at pickup.
    bp = _blueprint("fold", (_obj("cloth", role="cloth"),), {"fold_count": 1})
    flat = FakeWorld({"cloth": _box(0, 0, 0, 0.4, 0.4, 0.01)})
    checker = make_success_checker(bp, flat)
    hanging = FakeWorld({"cloth": _box(0, 0, 0.05, 0.03, 0.4, 0.35)})
    verdict = checker(hanging)
    assert not verdict.success
    assert "not lying flat" in verdict.explanation
    folded = FakeWorld({"cloth": _box(0, 0, 0, 0.4, 0.2, 0.02)})
    assert checker(folded).success


def test_fold_slack_loosens_threshold() -> None:
    tight = _blueprint("fold", (_obj("cloth", role="cloth"),), {"fold_count": 2})
    slacked = _blueprint("fold", (_obj("cloth", role="cloth"),), {"fold_count": 2, "slack": 0.4})
    flat = FakeWorld({"cloth": _box(0, 0, 0, 0.4, 0.4, 0.01)})
    # 0.36 ratio boundary: 0.6^2=0.36; with slack 0.4 -> 0.504
    at_040 = FakeWorld({"cloth": _box(0, 0, 0, 0.4, 0.16, 0.02)})  # ratio 0.40
    assert not make_success_checker(tight, flat)(at_040).success
    assert make_success_checker(slacked, flat)(at_040).success


# --------------------------------------------------------------------------- #
# seal
# --------------------------------------------------------------------------- #
def test_seal_seated_vs_offset() -> None:
    bp = _blueprint("seal", (_obj("lid", role="lid"), _obj("pot", role="container")))
    pot = _box(0, 0, 0, 0.24, 0.24, 0.14)
    checker = make_success_checker(bp, FakeWorld())
    seated = FakeWorld({"lid": _box(0, 0, 0.12, 0.20, 0.20, 0.03), "pot": pot})
    assert checker(seated).success  # screw lids seat below the rim: SEAL_TOL_M
    misaligned = FakeWorld({"lid": _box(0.2, 0, 0.14, 0.20, 0.20, 0.03), "pot": pot})
    assert not checker(misaligned).success
    floating = FakeWorld({"lid": _box(0, 0, 0.30, 0.20, 0.20, 0.03), "pot": pot})
    assert not checker(floating).success


# --------------------------------------------------------------------------- #
# handoff: stateful transfer detection
# --------------------------------------------------------------------------- #
def _handoff_bp(receiving: str) -> SceneBlueprint:
    return _blueprint("handoff", (_obj("cup", role="item"),), {"receiving_arm": receiving})


def _grip_world(item_x: float, left_x: float = -0.3, right_x: float = 0.3) -> FakeWorld:
    return FakeWorld(
        {
            "cup": _box(item_x, 0, 0.2, 0.09, 0.09, 0.10),
            "gripper_left": _box(left_x, 0, 0.2, 0.06, 0.06, 0.12),
            "gripper_right": _box(right_x, 0, 0.2, 0.06, 0.06, 0.12),
        }
    )


def test_handoff_left_to_right_transfer() -> None:
    checker = make_success_checker(_handoff_bp("right"), FakeWorld())
    held_left = _grip_world(item_x=-0.28)
    assert not checker(held_left).success  # establishes holder=left
    mid_transfer = FakeWorld(
        {
            # Item already nearer the receiving hand, but the giver still
            # grips it (both within GRASP_RADIUS_M): must not fire yet.
            "cup": _box(0.02, 0, 0.2, 0.09, 0.09, 0.10),
            "gripper_left": _box(-0.05, 0, 0.2, 0.06, 0.06, 0.12),
            "gripper_right": _box(0.05, 0, 0.2, 0.06, 0.06, 0.12),
        }
    )
    verdict = checker(mid_transfer)
    assert not verdict.success
    assert "mid-transfer" in verdict.explanation
    held_right = _grip_world(item_x=0.28)
    verdict = checker(held_right)
    assert verdict.success
    assert "handed off to right" in verdict.explanation


def test_handoff_wrong_direction_fails() -> None:
    checker = make_success_checker(_handoff_bp("left"), FakeWorld())
    assert not checker(_grip_world(item_x=-0.28)).success  # holder=left established
    verdict = checker(_grip_world(item_x=0.28))  # went to right
    assert not verdict.success
    assert "not left" in verdict.explanation


def test_handoff_either_accepts_any_direction() -> None:
    checker = make_success_checker(_handoff_bp("either"), FakeWorld())
    assert not checker(_grip_world(item_x=0.28)).success  # holder=right
    assert checker(_grip_world(item_x=-0.28)).success


def test_handoff_same_holder_and_unheld_do_not_fire() -> None:
    checker = make_success_checker(_handoff_bp("either"), FakeWorld())
    assert not checker(_grip_world(item_x=-0.28)).success
    assert not checker(_grip_world(item_x=-0.28)).success  # still left
    assert not checker(_grip_world(item_x=0.0)).success  # on the bench, unheld
    assert "not held" in checker(_grip_world(item_x=0.0)).explanation


def test_handoff_missing_gripper_fails_soft() -> None:
    checker = make_success_checker(_handoff_bp("either"), FakeWorld())
    verdict = checker(FakeWorld({"cup": _box(0, 0, 0.2, 0.09, 0.09, 0.10)}))
    assert not verdict.success
    assert "unknown object" in verdict.explanation


def test_handoff_bench_mediated_regrasp_does_not_count() -> None:
    # left holds -> item set down on the bench -> right picks it up: that is
    # placing, not a handoff. The set-down resets the tracked holder.
    checker = make_success_checker(_handoff_bp("either"), FakeWorld())
    assert not checker(_grip_world(item_x=-0.28)).success  # holder=left
    assert not checker(_grip_world(item_x=0.0)).success  # set down: holder reset
    verdict = checker(_grip_world(item_x=0.28))  # right regrasps from the bench
    assert not verdict.success
    assert "held by gripper_right" in verdict.explanation  # re-established, no transfer
    # A subsequent DIRECT transfer still succeeds.
    assert checker(_grip_world(item_x=-0.28)).success


def test_handoff_wrong_arm_transfer_reestablishes_holder() -> None:
    # The wrong gripper grasping first is not a life sentence: after a
    # completed (failing) transfer to the wrong arm, a transfer back to the
    # required arm succeeds.
    checker = make_success_checker(_handoff_bp("right"), FakeWorld())
    assert not checker(_grip_world(item_x=0.28)).success  # right established first
    wrong = checker(_grip_world(item_x=-0.28))  # right -> left completes
    assert not wrong.success
    assert "not right" in wrong.explanation
    assert checker(_grip_world(item_x=0.28)).success  # left -> right: success


def test_handoff_long_tool_end_grasp_counts() -> None:
    # Grasp distance is measured to the item's AABB, not its center: an end
    # grasp of a 40 cm tool (20 cm from center) is a grasp, and a natural
    # end-to-end handoff succeeds.
    def tool_world(left_x: float, right_x: float) -> FakeWorld:
        return FakeWorld(
            {
                "tool": _box(0.0, 0, 0.2, 0.40, 0.04, 0.03),  # spans x in [-0.2, 0.2]
                "gripper_left": _box(left_x, 0, 0.2, 0.06, 0.06, 0.12),
                "gripper_right": _box(right_x, 0, 0.2, 0.06, 0.06, 0.12),
            }
        )

    bp = _blueprint("handoff", (_obj("tool", "utensil", "item"),), {"receiving_arm": "right"})
    checker = make_success_checker(bp, FakeWorld())
    assert not checker(tool_world(left_x=-0.2, right_x=0.5)).success  # left end grasp: held
    both = checker(tool_world(left_x=-0.2, right_x=0.2))  # both ends: tie, no success
    assert not both.success
    assert "ambiguous" in both.explanation
    verdict = checker(tool_world(left_x=-0.5, right_x=0.2))  # left retreats
    assert verdict.success
    assert "handed off to right" in verdict.explanation


def test_handoff_equidistant_grip_is_ambiguous_and_keeps_holder() -> None:
    checker = make_success_checker(_handoff_bp("right"), FakeWorld())
    assert not checker(_grip_world(item_x=-0.28)).success  # holder=left
    tie = FakeWorld(
        {
            # Both grippers exactly 5 cm from the item: in range, no strict
            # nearer one. Must read as mid-transfer, not as a set-down.
            "cup": _box(0.0, 0, 0.2, 0.09, 0.09, 0.10),
            "gripper_left": _box(-0.05, 0, 0.2, 0.06, 0.06, 0.12),
            "gripper_right": _box(0.05, 0, 0.2, 0.06, 0.06, 0.12),
        }
    )
    verdict = checker(tie)
    assert not verdict.success
    assert "ambiguous" in verdict.explanation
    # The established holder survived the tie: completing the transfer fires.
    assert checker(_grip_world(item_x=0.28)).success


# --------------------------------------------------------------------------- #
# sort
# --------------------------------------------------------------------------- #
def _sort_bp() -> SceneBlueprint:
    return _blueprint(
        "sort",
        (
            _obj("compartment_spoon", "compartment", "compartment"),
            _obj("compartment_fork", "compartment", "compartment"),
            SceneObject("cutlery_1", "spoon", "sortable", 0.0, 0.0, 0.0),
            SceneObject("cutlery_2", "fork", "sortable", 3.0, 0.0, 0.0),
        ),
    )


def test_sort_all_pieces_in_their_compartments() -> None:
    checker = make_success_checker(_sort_bp(), FakeWorld())
    good = FakeWorld(
        fractions={
            ("cutlery_1", "compartment_spoon"): 0.9,
            ("cutlery_2", "compartment_fork"): 0.85,
        }
    )
    assert checker(good).success

    misplaced = FakeWorld(
        fractions={
            ("cutlery_1", "compartment_spoon"): 0.9,
            ("cutlery_2", "compartment_fork"): 0.2,
        }
    )
    verdict = checker(misplaced)
    assert not verdict.success
    assert "cutlery_2" in verdict.explanation


def test_sort_missing_compartment_class_fails() -> None:
    bp = _blueprint(
        "sort",
        (
            _obj("compartment_spoon", "compartment", "compartment"),
            SceneObject("cutlery_1", "knife", "sortable", 0.0, 0.0, 0.0),
        ),
    )
    verdict = make_success_checker(bp, FakeWorld())(FakeWorld())
    assert not verdict.success
    assert "no compartment for 'knife'" in verdict.explanation


# --------------------------------------------------------------------------- #
# construction errors
# --------------------------------------------------------------------------- #
def test_unknown_kind_rejected() -> None:
    bp = _blueprint("teleport", (_obj("cup", role="item"),))
    with pytest.raises(ValueError, match="unknown target kind"):
        make_success_checker(bp, FakeWorld())


def test_missing_role_rejected() -> None:
    bp = _blueprint("place_on", (_obj("spoon", role="item"),))  # no surface
    with pytest.raises(LookupError, match="exactly one 'surface'"):
        make_success_checker(bp, FakeWorld())


def test_missing_multi_role_rejected() -> None:
    bp = _blueprint("stack", (_obj("spoon", role="item"),))
    with pytest.raises(LookupError, match="needs the 'stack' role"):
        make_success_checker(bp, FakeWorld())
