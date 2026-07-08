"""Scene blueprints — a realization resolved into a spawnable object list.

``build_blueprint(scene, seed)`` recovers the instance behind an Inspect Robots
``Scene`` (same seam as :func:`kitchenbench.tasks.realize_scene`), realizes it,
and resolves the instance's :class:`~kitchenbench.instances.SimSpec` annotation
into concrete objects, roles, success parameters, and conditions. Everything is
a pure function of the realization values, so a simulator and a human operator
given the same seed set up the *same* scene.

Bench frame convention: +x is to the robot's right; ``gripper_left`` parks at
-x, ``gripper_right`` at +x. Placements are centimeters and degrees; the
:class:`~kitchenbench.sim.success.WorldState` queries are meters (SI), matching
simulator conventions on each side of the boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from kitchenbench.distributions import Scalar
from kitchenbench.instances import SimObject, TaskInstance, Var
from kitchenbench.tasks import find_instance, realize_scene

if TYPE_CHECKING:
    from inspect_robots import Scene

#: Version of the sim contract (blueprint semantics + success criteria +
#: threshold constants). Stamped on every blueprint; embodiments should log it.
SIM_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class AssetSpec:
    """Nominal dimensions of an asset class (cm) — the shared asset contract.

    Adapters build catalogs against these; success checkers read them where a
    criterion needs a baseline the scene cannot provide (a crumpled cloth's
    flat area, ``size_order`` base sizes).
    """

    footprint_w_cm: float
    footprint_h_cm: float
    height_cm: float


#: The closed asset vocabulary of the shipped annotations (asset -> nominal dims).
ASSETS: Mapping[str, AssetSpec] = MappingProxyType(
    {
        "spoon": AssetSpec(4, 18, 3),
        "fork": AssetSpec(3, 19, 2),
        "knife": AssetSpec(2, 21, 2),
        "cutlery": AssetSpec(3, 19, 2),  # split-expanded generic; spawned as its split class
        "plate": AssetSpec(26, 26, 3),
        "bowl": AssetSpec(16, 16, 8),
        "cup": AssetSpec(9, 9, 10),
        "pot": AssetSpec(24, 24, 14),
        "napkin": AssetSpec(20, 20, 1),
        "cloth": AssetSpec(45, 45, 1),
        "dish_towel": AssetSpec(45, 65, 1),
        "jar": AssetSpec(9, 9, 12),
        "bottle": AssetSpec(7, 7, 24),
        "food_container": AssetSpec(18, 12, 8),
        "container": AssetSpec(14, 14, 10),  # generic receiving container (scoop target)
        "lid": AssetSpec(18, 12, 3),
        "tray": AssetSpec(30, 40, 5),
        "compartment": AssetSpec(9, 36, 4),
        "dish_rack": AssetSpec(35, 25, 12),
        "measuring_cup": AssetSpec(10, 10, 9),
        "pasta_box": AssetSpec(8, 4, 26),
        "penne": AssetSpec(1, 1, 4),
        "rigatoni": AssetSpec(2, 2, 4),
        "dry_pasta": AssetSpec(1, 1, 5),
        "utensil": AssetSpec(4, 25, 3),
        "produce_item": AssetSpec(8, 8, 8),
        "distractor": AssetSpec(8, 8, 8),
    }
)

#: ``size_order`` per-copy size factors: "largest_first" shrinks each copy by
#: this step; "shuffled" applies a fixed deterministic rotation of the same
#: sequence (no RNG — reproducible from the annotation alone).
SIZE_ORDER_STEP = 0.12

#: Minimum edge-to-edge clearance (cm) between expanded copies. A sampled
#: spread narrower than a copy's own footprint would make rigid bodies
#: interpenetrate at spawn — a pose no simulator can honor, so each engine's
#: penetration resolution would scatter the objects differently and break the
#: same-seed-same-scene guarantee. Spacing is clamped to
#: ``widest copy footprint + SPREAD_CLEARANCE_CM``; the sampled value stays
#: verbatim in ``SceneBlueprint.values``.
SPREAD_CLEARANCE_CM = 1.0


@dataclass(frozen=True)
class SceneObject:
    """One concrete object a simulator spawns."""

    name: str
    asset: str
    role: str | None
    x_cm: float
    y_cm: float
    yaw_deg: float
    parent: str | None = None  # None = bench frame; else the named object's frame
    size_cm: float | None = None
    amount_g: float | None = None


@dataclass(frozen=True)
class SceneBlueprint:
    """A realized, machine-readable scene: spawn list + checker inputs."""

    contract_version: int
    scene_id: str
    instruction: str
    target_kind: str
    objects: tuple[SceneObject, ...]
    roles: Mapping[str, tuple[str, ...]]
    success_params: Mapping[str, Scalar]
    conditions: Mapping[str, Scalar]
    values: Mapping[str, Scalar]

    def object_names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objects)


def _slug_asset(value: str) -> str:
    """Sampled asset values may contain spaces ("dish towel"); assets are snake_case."""
    return value.strip().lower().replace(" ", "_")


def _size_factors(count: int, order: str) -> list[float]:
    """Deterministic per-copy size factors for ``size_order`` expansions.

    Floored at ``SIZE_ORDER_STEP`` so a large count can never yield a zero or
    negative size (today's counts max out at 5, nowhere near the floor).
    """
    factors = [max(1.0 - SIZE_ORDER_STEP * k, SIZE_ORDER_STEP) for k in range(count)]
    if order == "shuffled":
        # Fixed rotation, not an RNG shuffle: reproducible everywhere.
        pivot = count // 2
        factors = factors[pivot:] + factors[:pivot]
    return factors


class _Resolver:
    """Resolves ``Var`` references against realization values then statics."""

    def __init__(self, instance: TaskInstance, values: Mapping[str, Scalar]) -> None:
        self._values = values
        self._statics = instance.static

    def scalar(self, value: Scalar | Var) -> Scalar:
        if not isinstance(value, Var):
            return value
        if value.name in self._values:
            return self._values[value.name]
        static: Scalar = self._statics[value.name]
        return static

    def number(self, value: float | int | Var) -> float:
        resolved = self.scalar(value)
        assert not isinstance(resolved, str), f"numeric binding got string {resolved!r}"
        return float(resolved)

    def optional_number(self, value: float | int | Var | None) -> float | None:
        return None if value is None else self.number(value)


def _expand(obj: SimObject, resolver: _Resolver) -> list[SceneObject]:
    """One annotation object -> one or ``count`` concrete objects, laid out.

    A ``count`` bound to a ``Var`` always expands to numbered copies
    (``name_1..name_n``), even when it resolves to 1, so object names stay
    stable across epochs of the same instance. Only a literal ``count=1``
    keeps the bare annotation name.
    """
    count = int(resolver.number(obj.count))
    if count < 1:
        raise ValueError(f"object {obj.name!r} resolved count {count}; counts must be >= 1")
    base_asset = _slug_asset(str(resolver.scalar(obj.asset)))
    x = resolver.number(obj.x_cm)
    y = resolver.number(obj.y_cm)
    yaw = resolver.number(obj.yaw_deg)
    spread = resolver.number(obj.spread_cm)
    size = resolver.optional_number(obj.size_cm)
    amount = resolver.optional_number(obj.amount_g)

    if count == 1 and not isinstance(obj.count, Var):
        return [
            SceneObject(
                name=obj.name,
                asset=base_asset,
                role=obj.role,
                x_cm=x,
                y_cm=y,
                yaw_deg=yaw,
                parent=obj.parent,
                size_cm=size,
                amount_g=amount,
            )
        ]

    factors: list[float] | None = None
    if obj.size_order is not None:
        order = str(resolver.scalar(obj.size_order))
        factors = _size_factors(count, order)
        if size is None:
            size = ASSETS[base_asset].footprint_w_cm

    # Rigid copies must not interpenetrate at spawn (see SPREAD_CLEARANCE_CM).
    classes = [_slug_asset(c) for c in obj.split] if obj.split else [base_asset]
    min_spacing = max(ASSETS[c].footprint_w_cm for c in classes) + SPREAD_CLEARANCE_CM
    spacing = max(spread, min_spacing)

    expanded: list[SceneObject] = []
    for k in range(count):
        asset = _slug_asset(obj.split[k % len(obj.split)]) if obj.split else base_asset
        expanded.append(
            SceneObject(
                name=f"{obj.name}_{k + 1}",
                asset=asset,
                role=obj.role,
                x_cm=x + k * spacing,
                y_cm=y,
                yaw_deg=yaw,
                parent=obj.parent,
                size_cm=size * factors[k] if size is not None and factors is not None else size,
                amount_g=amount,
            )
        )
    return expanded


def build_blueprint(scene: Scene, seed: int | None) -> SceneBlueprint:
    """Resolve the scene's sim annotation for ``seed`` (``None`` guards to 0).

    Raises ``LookupError`` if the instance has no ``sim`` annotation — the
    benchmark cannot guess scene semantics (plan 0003).
    """
    instance = find_instance(scene)
    spec = instance.sim
    if spec is None:
        raise LookupError(
            f"instance {instance.instance_id!r} has no sim annotation; "
            "sim embodiments need TaskInstance.sim (see plan 0003)"
        )
    realization = realize_scene(scene, seed)
    resolver = _Resolver(instance, realization.values)

    objects: list[SceneObject] = []
    for sim_obj in spec.objects:
        objects.extend(_expand(sim_obj, resolver))

    roles: dict[str, list[str]] = {}
    for obj in objects:
        if obj.role is not None:
            roles.setdefault(obj.role, []).append(obj.name)

    success_params = {key: resolver.scalar(value) for key, value in spec.success}
    conditions = {name: realization.values[name] for name in spec.conditions}

    return SceneBlueprint(
        contract_version=SIM_CONTRACT_VERSION,
        scene_id=scene.id,
        instruction=realization.instruction,
        target_kind=instance.target_kind,
        objects=tuple(objects),
        roles=MappingProxyType({role: tuple(names) for role, names in roles.items()}),
        success_params=MappingProxyType(success_params),
        conditions=MappingProxyType(conditions),
        values=MappingProxyType(dict(realization.values)),
    )
