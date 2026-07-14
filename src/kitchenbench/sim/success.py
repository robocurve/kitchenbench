"""Quantitative success criteria — the benchmark's official sim semantics.

A simulator (or a pose-tracked real rig) exposes four queries through
:class:`WorldState`; :func:`make_success_checker` turns a
:class:`~kitchenbench.sim.blueprint.SceneBlueprint` into a per-step checker
returning a :class:`Verdict`. Checkers **never raise** on missing objects —
a partially spawned scene fails with an explanation, it doesn't crash the
rollout. All world geometry is meters (SI); thresholds below are the
benchmark parameters of ``SIM_CONTRACT_VERSION`` 1.

An embodiment that sees ``checker(world).success`` terminates the trial with
``termination_reason="success"`` and the existing ``task_success`` scorer
fires unchanged — no scoring code knows about simulators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from kitchenbench.sim.blueprint import ASSETS, SceneBlueprint

# --------------------------------------------------------------------------- #
# Benchmark parameters (sim contract v1)
# --------------------------------------------------------------------------- #
CONTACT_TOL_M = 0.02  # "resting on": bottom within 2 cm of the support's top
SEAL_TOL_M = 0.04  # looser: screw lids seat below the rim by thread depth
CONTAIN_FRACTION = 0.8  # rigid item "in" a rack/compartment
TRANSFER_FRACTION = 0.8  # poured substance mass fraction that must arrive
OPEN_FRACTION = 0.7  # articulation opening fraction counting as "open"
FOLD_RATIO_PER_FOLD = 0.6  # footprint-area ratio per completed fold
FOLD_MAX_HEIGHT_M = 0.04  # folded cloth lies flat: z-extent above this is a
#   grasped/hanging or still-crumpled cloth, whose footprint shrinks for the
#   wrong reason — footprint alone must never fire the fold criterion
GRASP_RADIUS_M = 0.12  # gripper-center-to-item-AABB distance counting as "held"
#   (surface distance, not center distance: an end grasp of a 40 cm tool is a
#   grasp; measuring to the item's center would rule it out by geometry)
SCOOP_TOL_G = 15.0  # default scoop tolerance; dominates instruction rounding

_GRIPPERS = ("gripper_left", "gripper_right")


class WorldState(Protocol):
    """What a simulator must answer, keyed by blueprint object names.

    Implementations raise ``KeyError`` for names they did not register;
    checkers convert that into a failing :class:`Verdict`.
    """

    def aabb(self, name: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axis-aligned bounds as ``(min_xyz, max_xyz)`` in meters."""
        ...

    def contained_fraction(self, item: str, container: str) -> float:
        """Fraction (0..1) of ``item``'s volume inside ``container``."""
        ...

    def contained_mass_g(self, substance: str, container: str) -> float:
        """Grams of ``substance`` currently inside ``container``."""
        ...

    def opening_fraction(self, name: str) -> float:
        """Articulation state of ``name``: 0 fully closed .. 1 fully open."""
        ...


@dataclass(frozen=True)
class Verdict:
    """Result of a quantitative check, paired with a human-readable reason."""

    success: bool
    explanation: str


SuccessChecker = Callable[[WorldState], Verdict]


def _center(aabb: tuple[NDArray[np.float64], NDArray[np.float64]]) -> NDArray[np.float64]:
    total: NDArray[np.float64] = np.asarray(aabb[0], dtype=np.float64) + np.asarray(
        aabb[1], dtype=np.float64
    )
    return total / 2.0


def _xy_inside(
    inner: tuple[NDArray[np.float64], NDArray[np.float64]],
    outer: tuple[NDArray[np.float64], NDArray[np.float64]],
) -> bool:
    cx, cy = _center(inner)[:2]
    (ox0, oy0, _), (ox1, oy1, _) = np.asarray(outer[0]), np.asarray(outer[1])
    return bool(ox0 <= cx <= ox1 and oy0 <= cy <= oy1)


def _footprint_area(aabb: tuple[NDArray[np.float64], NDArray[np.float64]]) -> float:
    low, high = np.asarray(aabb[0]), np.asarray(aabb[1])
    return float((high[0] - low[0]) * (high[1] - low[1]))


def _height(aabb: tuple[NDArray[np.float64], NDArray[np.float64]]) -> float:
    return float(np.asarray(aabb[1])[2] - np.asarray(aabb[0])[2])


def _distance_to_box(
    point: NDArray[np.float64], aabb: tuple[NDArray[np.float64], NDArray[np.float64]]
) -> float:
    """Euclidean distance from ``point`` to the nearest point of ``aabb`` (0 inside)."""
    low, high = np.asarray(aabb[0], dtype=np.float64), np.asarray(aabb[1], dtype=np.float64)
    gap = np.maximum(np.maximum(low - point, point - high), 0.0)
    return float(np.linalg.norm(gap))


def _one(blueprint: SceneBlueprint, role: str) -> str:
    names = blueprint.roles.get(role, ())
    if len(names) != 1:
        raise LookupError(f"target kind {blueprint.target_kind!r} needs exactly one {role!r} role")
    return names[0]


def _many(blueprint: SceneBlueprint, role: str) -> tuple[str, ...]:
    names = blueprint.roles.get(role, ())
    if not names:
        raise LookupError(f"target kind {blueprint.target_kind!r} needs the {role!r} role")
    return names


class _Checker:
    """Wraps a criterion so unknown-object queries fail soft, never raise."""

    def __init__(self, criterion: Callable[[WorldState], Verdict]) -> None:
        self._criterion = criterion

    def __call__(self, world: WorldState) -> Verdict:
        try:
            return self._criterion(world)
        except KeyError as exc:
            return Verdict(False, f"unknown object {exc.args[0]!r}")


def _check_place_on(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    item, surface = _one(blueprint, "item"), _one(blueprint, "surface")

    def criterion(world: WorldState) -> Verdict:
        item_box, surface_box = world.aabb(item), world.aabb(surface)
        if not _xy_inside(item_box, surface_box):
            return Verdict(False, f"{item} is not over {surface}")
        gap = float(np.asarray(item_box[0])[2] - np.asarray(surface_box[1])[2])
        if abs(gap) > CONTACT_TOL_M:
            return Verdict(False, f"{item} is not resting on {surface} (gap {gap:.3f} m)")
        return Verdict(True, f"{item} rests on {surface}")

    return criterion


def _check_stack(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    names = _many(blueprint, "stack")

    def criterion(world: WorldState) -> Verdict:
        boxes = {name: world.aabb(name) for name in names}
        ordered = sorted(names, key=lambda n: float(np.asarray(boxes[n][0])[2]))
        for lower, upper in pairwise(ordered):
            if not _xy_inside(boxes[upper], boxes[lower]):
                return Verdict(False, f"{upper} is not over {lower}")
            upper_bottom = float(np.asarray(boxes[upper][0])[2])
            lower_bottom = float(np.asarray(boxes[lower][0])[2])
            lower_top = float(np.asarray(boxes[lower][1])[2])
            # Nesting-aware: an upper cup may sit *inside* the lower one
            # (bottom below the lower rim), a plate sits on top.
            if not lower_bottom < upper_bottom <= lower_top + CONTACT_TOL_M:
                return Verdict(False, f"{upper} is not stacked on {lower}")
        return Verdict(True, f"stacked: {' -> '.join(ordered)}")

    return criterion


def _check_place_in(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    item, rack = _one(blueprint, "item"), _one(blueprint, "rack")

    def criterion(world: WorldState) -> Verdict:
        fraction = world.contained_fraction(item, rack)
        if fraction >= CONTAIN_FRACTION:
            return Verdict(True, f"{item} is in {rack} ({fraction:.0%})")
        return Verdict(False, f"{item} is not in {rack} ({fraction:.0%})")

    return criterion


def _check_pour_into(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    vessel = _one(blueprint, "vessel")
    substance = str(blueprint.success_params["substance"])
    total_g = float(blueprint.success_params["total_g"])

    def criterion(world: WorldState) -> Verdict:
        arrived = world.contained_mass_g(substance, vessel)
        needed = TRANSFER_FRACTION * total_g
        if arrived >= needed:
            return Verdict(True, f"{arrived:.0f} g of {substance} in {vessel}")
        return Verdict(False, f"only {arrived:.0f}/{needed:.0f} g of {substance} in {vessel}")

    return criterion


def _check_open(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    container = _one(blueprint, "container")

    def criterion(world: WorldState) -> Verdict:
        fraction = world.opening_fraction(container)
        if fraction >= OPEN_FRACTION:
            return Verdict(True, f"{container} is open ({fraction:.0%})")
        return Verdict(False, f"{container} is not open ({fraction:.0%})")

    return criterion


def _check_fold(blueprint: SceneBlueprint, world: WorldState) -> Callable[[WorldState], Verdict]:
    cloth = _one(blueprint, "cloth")
    fold_count = int(float(blueprint.success_params.get("fold_count", 1)))
    slack = float(blueprint.success_params.get("slack", 0.0))
    if blueprint.success_params.get("baseline") == "nominal":
        # Crumpled start: the initial footprint is already shrunk, so the
        # denominator is the asset's nominal flat area (asset contract).
        asset = next(obj.asset for obj in blueprint.objects if obj.name == cloth)
        spec = ASSETS[asset]
        baseline = (spec.footprint_w_cm / 100.0) * (spec.footprint_h_cm / 100.0)
    else:
        baseline = _footprint_area(world.aabb(cloth))
    threshold = (FOLD_RATIO_PER_FOLD**fold_count) * (1.0 + slack) * baseline

    def criterion(world: WorldState) -> Verdict:
        box = world.aabb(cloth)
        # Footprint alone shrinks for the wrong reasons — a grasped, hanging
        # cloth or a still-crumpled ball both have small footprints. A folded
        # cloth also lies FLAT: gate on z-extent first.
        height = _height(box)
        if height > FOLD_MAX_HEIGHT_M:
            return Verdict(False, f"{cloth} is not lying flat (height {height * 100:.0f} cm)")
        area = _footprint_area(box)
        if area <= threshold:
            return Verdict(True, f"{cloth} folded (footprint {area * 1e4:.0f} cm2)")
        return Verdict(False, f"{cloth} not folded ({area * 1e4:.0f} > {threshold * 1e4:.0f} cm2)")

    return criterion


def _check_seal(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    lid, container = _one(blueprint, "lid"), _one(blueprint, "container")

    def criterion(world: WorldState) -> Verdict:
        lid_box, container_box = world.aabb(lid), world.aabb(container)
        if not _xy_inside(lid_box, container_box):
            return Verdict(False, f"{lid} is not over {container}")
        gap = float(np.asarray(lid_box[0])[2] - np.asarray(container_box[1])[2])
        if abs(gap) > SEAL_TOL_M:
            return Verdict(False, f"{lid} is not seated on {container} (gap {gap:.3f} m)")
        return Verdict(True, f"{container} sealed with {lid}")

    return criterion


class _HandoffChecker:
    """Stateful: success only when an established holder *directly* hands over.

    "Handoff" means the item moves gripper-to-gripper. Three rules keep that
    honest with proximity-only holding detection:

    - A set-down resets the tracked holder: once the item is away from both
      grippers, placing it on the bench and regrasping with the other arm is
      *placing*, not a handoff (and a transient false "hold" — a gripper merely
      passing near the item — can't poison the rest of the trial).
    - Both grippers in range without a strict nearer one is ambiguous
      (mid-transfer); the tracked holder is kept, never re-established.
    - A completed transfer to the *wrong* arm fails but re-establishes the
      holder, so a subsequent transfer back to the required arm can still
      succeed — the first grasp of a trial is not a life sentence.
    """

    def __init__(self, item: str, receiving_arm: str) -> None:
        self._item = item
        self._receiving = receiving_arm
        self._holder: str | None = None

    def _current_holder(self, world: WorldState) -> tuple[str | None, dict[str, float]]:
        # Distance from each gripper's center to the item's AABB *surface*, not
        # its center: an end grasp of a long tool is still a grasp.
        item_box = world.aabb(self._item)
        distances = {
            gripper: _distance_to_box(_center(world.aabb(gripper)), item_box)
            for gripper in _GRIPPERS
        }
        near, far = sorted(_GRIPPERS, key=lambda g: distances[g])
        if distances[near] <= GRASP_RADIUS_M and distances[near] < distances[far]:
            return near, distances
        return None, distances

    def __call__(self, world: WorldState) -> Verdict:
        try:
            holder, distances = self._current_holder(world)
        except KeyError as exc:
            return Verdict(False, f"unknown object {exc.args[0]!r}")
        if holder is None:
            if min(distances.values()) > GRASP_RADIUS_M:
                # Away from both grippers: a set-down, not part of a handoff.
                self._holder = None
                return Verdict(False, f"{self._item} is not held")
            # Both grippers in range, neither strictly nearer: ambiguous
            # (mid-transfer); keep the established holder.
            return Verdict(False, f"{self._item} grip is ambiguous (both grippers in range)")
        if self._holder is None:
            self._holder = holder
            return Verdict(False, f"{self._item} held by {holder}")
        if holder == self._holder:
            return Verdict(False, f"{self._item} still held by {holder}")
        # The previous holder must have actually released (both-hands-on
        # mid-transfer must not count as success).
        if distances[self._holder] <= GRASP_RADIUS_M:
            return Verdict(False, f"{self._item} held by both grippers (mid-transfer)")
        # A completed gripper-to-gripper transfer: the taker is the holder now,
        # whether or not it was the required arm.
        self._holder = holder
        received_side = holder.removeprefix("gripper_")
        if self._receiving != "either" and received_side != self._receiving:
            return Verdict(
                False, f"{self._item} went to the {received_side} arm, not {self._receiving}"
            )
        return Verdict(True, f"{self._item} handed off to {received_side} arm")


def _check_sort(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    sortables = _many(blueprint, "sortable")
    compartments = _many(blueprint, "compartment")
    asset_by_name = {obj.name: obj.asset for obj in blueprint.objects}
    compartment_for = {name.removeprefix("compartment_"): name for name in compartments}

    def criterion(world: WorldState) -> Verdict:
        for name in sortables:
            compartment = compartment_for.get(asset_by_name[name])
            if compartment is None:
                return Verdict(False, f"no compartment for {asset_by_name[name]!r}")
            fraction = world.contained_fraction(name, compartment)
            if fraction < CONTAIN_FRACTION:
                return Verdict(False, f"{name} is not in {compartment} ({fraction:.0%})")
        return Verdict(True, f"all {len(sortables)} pieces sorted")

    return criterion


def _check_scoop_transfer(blueprint: SceneBlueprint) -> Callable[[WorldState], Verdict]:
    container = _one(blueprint, "container")
    substance = str(blueprint.success_params["substance"])
    target_g = float(blueprint.success_params["target_g"])
    tol_g = float(blueprint.success_params.get("tol_g", SCOOP_TOL_G))

    def criterion(world: WorldState) -> Verdict:
        arrived = world.contained_mass_g(substance, container)
        if abs(arrived - target_g) <= tol_g:
            return Verdict(
                True, f"{arrived:.0f} g in {container} (target {target_g:.0f}±{tol_g:.0f})"
            )
        return Verdict(
            False, f"{arrived:.0f} g in {container}, target {target_g:.0f}±{tol_g:.0f} g"
        )

    return criterion


def make_success_checker(blueprint: SceneBlueprint, world: WorldState) -> SuccessChecker:
    """Build the per-step success checker for a blueprint.

    ``world`` is consulted at construction only where a criterion needs the
    *initial* state (fold's baseline footprint); stateful criteria (handoff)
    accumulate across calls. Raises ``LookupError`` for a blueprint missing the
    kind's required roles (an annotation bug, caught by the instance sweep) and
    ``KeyError``/``ValueError`` for unknown kinds or missing success params.
    """
    kind = blueprint.target_kind
    if kind == "place_on":
        return _Checker(_check_place_on(blueprint))
    if kind == "stack":
        return _Checker(_check_stack(blueprint))
    if kind == "place_in":
        return _Checker(_check_place_in(blueprint))
    if kind == "pour_into":
        return _Checker(_check_pour_into(blueprint))
    if kind == "open":
        return _Checker(_check_open(blueprint))
    if kind == "fold":
        return _Checker(_check_fold(blueprint, world))
    if kind == "seal":
        return _Checker(_check_seal(blueprint))
    if kind == "handoff":
        receiving = str(blueprint.success_params.get("receiving_arm", "either"))
        return _HandoffChecker(_one(blueprint, "item"), receiving)
    if kind == "sort":
        return _Checker(_check_sort(blueprint))
    if kind == "scoop_transfer":
        return _Checker(_check_scoop_transfer(blueprint))
    raise ValueError(f"unknown target kind {kind!r}")
