"""Task instances — the unit of the physical-automation methodology.

A **task instance** is a self-contained scenario: a *stochastic environment
setup* (named :class:`~kitchenbench.distributions.Distribution` variables) plus a
*goal* (a natural-language success criterion that may reference sampled
variables). A **realization** samples every variable to produce one concrete
environment; the methodology runs ``K_REALIZATIONS`` realizations of each of
``K_INSTANCES`` instances per task and estimates the instance success probability
as the mean binary outcome.

This module is pure (no Inspect Robots import); :mod:`kitchenbench.tasks` maps instances
onto Inspect Robots ``Scene``/``Epochs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kitchenbench.distributions import Distribution, Scalar

#: Methodology recommendations (see the reference docs).
K_REALIZATIONS = 5  # independent rollouts per instance -> success probability
K_INSTANCES = 5  # validated instances per task
K_EXPERTS = 3  # validators per instance


@dataclass(frozen=True)
class Validation:
    """Per-expert validation ratings carried with an instance.

    Empty until the human commissioning pipeline is run; :attr:`validated` follows
    the methodology accept rule (``K_EXPERTS`` ratings, all >= 4 on both axes).
    """

    representativeness: tuple[int, ...] = ()
    quality: tuple[int, ...] = ()
    difficulty: tuple[int, ...] = ()  # optional ("nice to have" in the methodology)
    source: str = "opus-draft"  # provenance: "opus-draft" (AI) vs "human"

    @property
    def validated(self) -> bool:
        """Whether the ratings meet the methodology's expert count and accept threshold."""
        return (
            len(self.representativeness) >= K_EXPERTS
            and len(self.quality) >= K_EXPERTS
            and all(r >= 4 for r in self.representativeness)
            and all(q >= 4 for q in self.quality)
        )


@dataclass(frozen=True)
class Realization:
    """One concrete environment sampled from a :class:`TaskInstance`."""

    seed: int
    values: dict[str, Scalar]
    instruction: str
    setup_lines: tuple[str, ...]

    def __hash__(self) -> int:
        """Compute a hash of the realization, converting dict to a sorted items tuple."""
        return hash((
            self.seed,
            tuple(sorted(self.values.items())),
            self.instruction,
            self.setup_lines,
        ))



@dataclass(frozen=True)
class Var:
    """A reference to a setup variable (or static), resolved at realization time."""

    name: str


@dataclass(frozen=True)
class SimObject:
    """One object (or numbered family of objects) a simulator must spawn.

    Placements are centimeters/degrees in the bench frame, or in the parent
    object's frame when :attr:`parent` is set (compartments ride their tray;
    a substance spawns inside its source vessel). A ``count`` bound to a
    ``Var`` — and any literal ``count > 1`` — expands to ``name_1..name_n``
    (numbered even when a sampled count resolves to 1, so names stay stable
    across epochs) laid out deterministically: copy ``k`` (0-based) sits at
    ``(x_cm + k * spacing, y_cm)`` where spacing is ``spread_cm`` clamped up
    to the widest copy's nominal footprint plus a clearance, so rigid copies
    never interpenetrate at spawn. Counts below 1 are a blueprint-time
    error. ``split`` round-robins asset classes over the copies in declared
    order. ``size_order`` ("largest_first"/"shuffled") maps to a documented
    deterministic per-copy size sequence.
    """

    name: str
    asset: str | Var
    role: str | None = None
    x_cm: float | Var = 0.0
    y_cm: float | Var = 0.0
    yaw_deg: float | Var = 0.0
    parent: str | None = None
    count: int | Var = 1
    split: tuple[str, ...] = ()
    spread_cm: float | Var = 0.0
    size_cm: float | Var | None = None
    size_order: Var | None = None
    amount_g: float | Var | None = None


@dataclass(frozen=True)
class SimSpec:
    """Machine-readable sim semantics for one instance (see plan 0003).

    ``success`` holds per-target-kind parameters as ``(key, value)`` pairs
    (tuple-of-pairs keeps the dataclass hashable); ``conditions`` names setup
    variables that are physical/task conditions a simulator applies or reports
    as unsupported. Validation enforces the **coverage invariant**: every setup
    variable is referenced somewhere (object binding, success, or conditions),
    and at most once among object bindings.
    """

    objects: tuple[SimObject, ...]
    success: tuple[tuple[str, Scalar | Var], ...] = ()
    conditions: tuple[str, ...] = ()


#: Rig/scene names a simulator must register besides the blueprint objects.
RESERVED_SIM_NAMES = frozenset({"gripper_left", "gripper_right", "bench"})

_OBJECT_BINDING_FIELDS = (
    "asset",
    "x_cm",
    "y_cm",
    "yaw_deg",
    "count",
    "spread_cm",
    "size_cm",
    "size_order",
    "amount_g",
)


def _validate_sim_spec(instance: TaskInstance) -> None:
    """Structural validation of an instance's ``sim`` annotation (import-time)."""
    spec = instance.sim
    if spec is None:
        return
    iid = instance.instance_id
    known = set(instance.setup) | set(instance.static)

    names: set[str] = set()
    bound: list[str] = []
    for obj in spec.objects:
        if obj.name in names:
            raise ValueError(f"{iid}: duplicate sim object name {obj.name!r}")
        names.add(obj.name)
        for field_name in _OBJECT_BINDING_FIELDS:
            value = getattr(obj, field_name)
            if isinstance(value, Var):
                if value.name not in known:
                    raise ValueError(
                        f"{iid}: object {obj.name!r} binds unknown variable {value.name!r}"
                    )
                bound.append(value.name)
    duplicates = {name for name in bound if bound.count(name) > 1}
    if duplicates:
        raise ValueError(f"{iid}: variables bound by more than one object field: {duplicates}")

    declared: set[str] = set()
    for obj in spec.objects:
        if obj.parent is not None and obj.parent not in declared | RESERVED_SIM_NAMES:
            # Parents must be declared earlier in the tuple: typos surface at
            # import, and cycles are unrepresentable by construction.
            raise ValueError(f"{iid}: object {obj.name!r} has unknown parent {obj.parent!r}")
        declared.add(obj.name)
        if obj.split and not isinstance(obj.count, Var) and obj.count <= 1:
            raise ValueError(f"{iid}: object {obj.name!r} has a split but no count")

    success_refs = {v.name for _, v in spec.success if isinstance(v, Var)}
    for ref in success_refs:
        if ref not in known:
            raise ValueError(f"{iid}: success references unknown variable {ref!r}")
    for cond in spec.conditions:
        if cond not in instance.setup:
            raise ValueError(f"{iid}: condition {cond!r} is not a setup variable")

    covered = set(bound) | success_refs | set(spec.conditions)
    uncovered = set(instance.setup) - covered
    if uncovered:
        raise ValueError(
            f"{iid}: setup variables not covered by the sim annotation: {sorted(uncovered)}"
        )


@dataclass(frozen=True)
class TaskInstance:
    """A stochastic scenario: named distributions + a goal."""

    instance_id: str
    goal: str  # template; ``{var}`` placeholders must be in ``language_vars``
    setup: dict[str, Distribution]
    target_kind: str
    language_vars: tuple[str, ...] = ()
    static: dict[str, Any] = field(default_factory=dict)
    validation: Validation = field(default_factory=Validation)
    sim: SimSpec | None = None

    def __post_init__(self) -> None:
        _validate_sim_spec(self)

    def realize(self, seed: int) -> Realization:
        """Sample every setup variable (sorted-key order) for one concrete environment."""
        rng = np.random.default_rng(seed)
        values: dict[str, Scalar] = {key: self.setup[key].sample(rng) for key in sorted(self.setup)}
        instruction = self.goal.format(**{k: values[k] for k in self.language_vars})
        setup_lines = tuple(f"{key} = {values[key]}" for key in sorted(self.setup))
        return Realization(
            seed=seed, values=values, instruction=instruction, setup_lines=setup_lines
        )

    def setup_spec(self) -> dict[str, str]:
        """A JSON-native description of the setup (``{var: distribution.describe()}``)."""
        return {key: self.setup[key].describe() for key in sorted(self.setup)}

    def __hash__(self) -> int:
        """Compute a hash of the instance, converting dict fields to sorted items tuples."""
        return hash((
            self.instance_id,
            self.goal,
            tuple(sorted(self.setup.items())),
            self.target_kind,
            self.language_vars,
            tuple(sorted(self.static.items())),
            self.validation,
            self.sim,
        ))
